import os
import json
import datetime
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

import src.data_manager as dm
import src.model_trainer as mt
import src.optimizer as opt

app = Flask(__name__)

# Ensure data directory exists
os.makedirs(dm.DATA_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    metrics_path = os.path.join(dm.DATA_DIR, "metrics_report.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'Metrics not available yet. Model must be trained.'}), 404

@app.route('/api/retrain', methods=['POST'])
def retrain():
    try:
        metrics = mt.train_models()
        return jsonify({
            'success': True,
            'message': 'Model retrained successfully!',
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error during training: {str(e)}'
        }), 500

@app.route('/api/forecast', methods=['POST'])
def forecast():
    try:
        data = request.json or {}
        
        # Parse inputs
        target_date_str = data.get('date') # Format: YYYY-MM-DD
        lat = float(data.get('lat', dm.LAT))
        lon = float(data.get('lon', dm.LON))
        api_key = data.get('openweather_api_key', None)
        if not api_key:
            api_key = dm.OPENWEATHER_KEY
        
        # Market factors
        factors = {
            'Gas_Price': float(data.get('gas_price', 35.0)),
            'Nuclear_Outage': float(data.get('nuclear_outage', 0.15)),
            'Solar_Strike': float(data.get('solar_strike', 0.0)),
            'Market_Coeff': float(data.get('market_coeff', 1.0)),
            'VDR_Volume': float(data.get('vdr_volume', 1.0)),
            'Grid_Import_Export': float(data.get('grid_import_export', 0.0))
        }
        
        # Battery options
        battery_params = {
            'battery_capacity': float(data.get('battery_capacity', 1000.0)),
            'max_charge_power': float(data.get('max_charge_power', 250.0)),
            'max_discharge_power': float(data.get('max_discharge_power', 250.0)),
            'charge_efficiency': float(data.get('charge_efficiency', 95.0)) / 100.0,
            'discharge_efficiency': float(data.get('discharge_efficiency', 95.0)) / 100.0,
            'initial_soc': float(data.get('initial_soc', 20.0)) / 100.0,
            'min_soc': float(data.get('min_soc', 10.0)) / 100.0,
            'max_soc': float(data.get('max_soc', 90.0)) / 100.0,
            'max_cycles_per_day': float(data.get('max_cycles_per_day', 1.5)),
            'degradation_cost': float(data.get('degradation_cost', 1.20)),
            'transmission_tariff': float(data.get('transmission_tariff', 528.57)),
            'distribution_tariff': float(data.get('distribution_tariff', 1500.0)),
            'dispatch_tariff': float(data.get('dispatch_tariff', 104.57)),
            'supplier_margin': float(data.get('supplier_margin', 100.0)),
            'mode': data.get('mode', 'arbitrage')
        }
        
        # Determine if selected date is in our historical dataset
        target_date = datetime.datetime.strptime(target_date_str, '%Y-%m-%d').date()
        today = datetime.date.today()
        
        # Automatically sync current month prices and weather in real-time
        dm.sync_realtime_data()
        
        # Make sure merged data exists
        if not os.path.exists(dm.MERGED_DATA_PATH):
            print("Merged historical data not found. Loading and building...")
            dm.get_combined_historical_data()
            
        df_hist = pd.read_csv(dm.MERGED_DATA_PATH)
        df_hist['Datetime'] = pd.to_datetime(df_hist['Datetime'])
        
        # Verify database completeness
        data_status = dm.verify_data_completeness()
        
        # Find if we have actual data for the target date
        day_mask = df_hist['Datetime'].dt.date == target_date
        df_day = df_hist[day_mask].sort_values('Datetime')
        
        is_historical = len(df_day) == 24
        
        # 1. Weather Data for the target date
        if is_historical:
            # Load actual weather from our historical dataset
            weather_forecast = pd.DataFrame({
                'Temperature': df_day['Temperature'].values,
                'Cloud_Cover': df_day['Cloud_Cover'].values,
                'Wind_Speed': df_day['Wind_Speed'].values,
                'Shortwave_Radiation': df_day['Shortwave_Radiation'].values
            })
            actual_prices = df_day['Price'].tolist()
            print(f"Loading historical prices and weather for {target_date_str}")
        else:
            # Fetch live weather forecast
            print(f"Fetching weather forecast for {target_date_str}")
            weather_forecast = dm.fetch_weather_forecast(lat, lon, api_key)
            actual_prices = None
            
        # 2. Get past prices to construct lags
        # We need the last 168 hours of prices prior to target_date
        # Find target_date first hour
        target_dt_start = pd.to_datetime(target_date)
        hist_before_target = df_hist[df_hist['Datetime'] < target_dt_start].sort_values('Datetime')
        
        if len(hist_before_target) >= 168:
            last_prices = hist_before_target['Price'].iloc[-168:].tolist()
        else:
            # Fallback if target date is early in 2025 or we don't have enough history
            print("Warning: Insufficient history before target date. Using prices of target date or default padding.")
            last_prices = df_hist['Price'].iloc[:168].tolist()
            
        # 3. Predict prices using XGBoost and Neural Network
        prediction_results = mt.predict_next_day(target_date, weather_forecast, last_prices, factors)
        
        # Use XGBoost as the main prediction for the optimizer,
        # but let the user select between XGBoost and MLP (Neural Network) predictions to run the solver!
        selected_model = data.get('selected_model', 'xgboost')
        predicted_prices = prediction_results[selected_model]
        
        # 4. Run optimization
        optimization_results = opt.optimize_battery_schedule(predicted_prices, **battery_params)
        
        # If we have actual prices, run optimization on ACTUAL prices too for comparison!
        actual_optimization = None
        if actual_prices:
            actual_optimization = opt.optimize_battery_schedule(actual_prices, **battery_params)
            
        # 5. Explain price drivers & surplus
        surplus_hours = []
        high_price_hours = []
        explanation_bullets = []
        
        # Look for surplus conditions in forecast
        surplus_hours_idx = []
        for h in range(24):
            xgb_p = prediction_results['xgboost'][h]
            if xgb_p <= 15.0: # hit the floor
                surplus_hours.append(h + 1)
                surplus_hours_idx.append(h)
            elif xgb_p >= 6000.0: # peak pricing
                high_price_hours.append(h + 1)
                
        # Build explanation text
        if surplus_hours:
            explanation_bullets.append(
                f"**Енергетичний профіцит** очікується в годинах: {', '.join(map(str, surplus_hours))}. "
                f"У цей період прогнозується падіння ціни до **10 грн/МВт-год**. Основні фактори: "
                f"висока сонячна радіація (макс. {weather_forecast['Shortwave_Radiation'].max():.1f} Вт/м²), "
                f"низька хмарність ({weather_forecast['Cloud_Cover'].iloc[surplus_hours_idx[0]]:.1f}%) та низьке споживання (вихідний день або низький ринковий коефіцієнт)."
            )
        else:
            explanation_bullets.append(
                "**Профіцит  енергії (ціна 10 грн) не очікується.** Для його виникнення необхідні: "
                "сонячна радіація > 500 Вт/м², хмарність < 25%, відсутність пошкоджень СЕС та низьке базове споживання (наприклад, вихідний день)."
            )
            
        if high_price_hours:
            explanation_bullets.append(
                f"**Пікові ціни (вище 6000 грн)** прогнозуються в годинах: {', '.join(map(str, high_price_hours))}. "
                f"Це пов'язано з високою вартістю теплової генерації (газ TTF = {factors['Gas_Price']} EUR/MWh) та високим дефіцитом потужності (частка виведених АЕС = {factors['Nuclear_Outage']*100:.1f}%)."
            )
        else:
            max_p_hour = np.argmax(prediction_results['xgboost'])
            max_p = np.max(prediction_results['xgboost'])
            explanation_bullets.append(
                f"Максимальна ціна прогнозується в {max_p_hour + 1} годині на рівні **{max_p:.2f} грн/МВт-год**. "
                f"Вона обумовлена вечірнім піком споживання та цінами на газ."
            )
            
        if factors['Solar_Strike'] > 0.0:
            explanation_bullets.append(
                f"**Фактор прильотів по СЕС ({factors['Solar_Strike']*100:.0f}%):** Знижує загальну сонячну генерацію в енергосистемі. "
                "Це зменшує ймовірність профіциту в денний час та утримує денні ціни вище регуляторного мінімуму."
            )
            
        if factors['Nuclear_Outage'] > 0.30:
            explanation_bullets.append(
                f"**Критичний дефіцит АЕС ({factors['Nuclear_Outage']*100:.0f}%):** Виведення базових  енергоблоків "
                "призводить до зростання цін у всі години доби, змушуючи вмикати дорогі маневрові ТЕС/ТЕЦ."
            )
            
        actual_factors = None
        if is_historical:
            actual_factors = {
                'gas_price': float(df_day['Gas_Price'].mean()),
                'nuclear_outage': float(df_day['Nuclear_Outage'].mean() * 100.0), # as percent
                'solar_strike': float(df_day['Solar_Strike'].mean() * 100.0), # as percent
                'market_coeff': float(df_day['Market_Coeff'].mean()),
                'vdr_volume': float(df_day['VDR_Volume'].mean()),
                'grid_import_export': float(df_day['Grid_Import_Export'].mean())
            }

        response = {
            'date': target_date_str,
            'is_historical': is_historical,
            'actual_factors': actual_factors,
            'data_status': data_status,
            'weather': {
                'hours': list(range(24)),
                'temp': weather_forecast['Temperature'].tolist(),
                'clouds': weather_forecast['Cloud_Cover'].tolist(),
                'wind': weather_forecast['Wind_Speed'].tolist(),
                'radiation': weather_forecast['Shortwave_Radiation'].tolist(),
            },
            'forecast': {
                'hours': prediction_results['hours'],
                'xgboost': prediction_results['xgboost'],
                'mlp': prediction_results['mlp'],
                'actual': actual_prices # None if future
            },
            'optimization': optimization_results,
            'actual_optimization': actual_optimization,
            'explanations': explanation_bullets
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # We will run the server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
