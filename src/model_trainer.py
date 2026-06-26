import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import src.data_manager as dm

DATA_DIR = "/home/oleg/antigravity/data"
XGB_MODEL_PATH = os.path.join(DATA_DIR, "model_xgboost.pkl")
MLP_MODEL_PATH = os.path.join(DATA_DIR, "model_mlp.pkl")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")
DEFAULT_MODEL_PATH = "/home/oleg/antigravity/model.pkl"

FEATURES = [
    'Hour', 'Month', 'DayOfWeek', 'Is_Weekend', 'Is_Holiday', 'Is_Weekend_Or_Holiday',
    'Temperature', 'Cloud_Cover', 'Wind_Speed', 'Shortwave_Radiation',
    'Gas_Price', 'Nuclear_Outage', 'Solar_Strike', 'Market_Coeff',
    'VDR_Volume', 'Grid_Import_Export',
    'Solar_Gen', 'Wind_Gen', 'Nuclear_Gen',
    'Hour_Sin', 'Hour_Cos', 'Month_Sin', 'Month_Cos',
    'DayOfYear_Sin', 'DayOfYear_Cos',
    'Is_Night', 'Is_Morning_Peak', 'Is_Daytime', 'Is_Evening_Peak',
    'Price_Lag_24', 'Price_Lag_48', 'Price_Lag_168', 'Price_Mean_24h',
    'Temp_Lag_3', 'Temp_Lag_6',
    'Cloud_Lag_3', 'Cloud_Lag_6',
    'Radiation_Lag_3', 'Radiation_Lag_6'
]

def is_ukrainian_holiday(dt):
    # Standard fixed Ukrainian holidays
    fixed_holidays = [
        (1, 1),   # New Year
        (1, 7),   # Christmas
        (3, 8),   # Women's Day
        (5, 1),   # Labour Day
        (5, 9),   # Victory Day
        (6, 28),  # Constitution Day
        (8, 24),  # Independence Day
        (10, 14), # Defenders Day
        (12, 25)  # Christmas
    ]
    if (dt.month, dt.day) in fixed_holidays:
        return 1
    return 0

def prepare_features(df):
    """
    Constructs features and target from the historical dataframe.
    Reindexes data to a continuous hourly range to prevent lag alignment shift errors!
    """
    df = df.copy()
    
    # 1. Set Datetime as index and sort
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').set_index('Datetime')
    
    # 2. Reindex to a complete continuous hourly range to fix alignment!
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    df = df.reindex(full_range)
    df.index.name = 'Datetime'
    
    # 3. Interpolate values for the gaps
    df['Price'] = df['Price'].interpolate(method='linear').bfill().ffill()
    df['Temperature'] = df['Temperature'].interpolate(method='linear').bfill().ffill()
    df['Cloud_Cover'] = df['Cloud_Cover'].interpolate(method='linear').fillna(50.0)
    df['Wind_Speed'] = df['Wind_Speed'].interpolate(method='linear').fillna(15.0)
    df['Shortwave_Radiation'] = df['Shortwave_Radiation'].interpolate(method='linear').fillna(0.0)
    
    # Re-calculate simulated generation columns if they have NaNs
    df['Solar_Strike'] = df['Solar_Strike'].fillna(0.0)
    df['Nuclear_Outage'] = df['Nuclear_Outage'].fillna(0.15)
    
    df['Solar_Gen'] = df['Solar_Gen'].interpolate(method='linear')
    df['Wind_Gen'] = df['Wind_Gen'].interpolate(method='linear')
    df['Nuclear_Gen'] = df['Nuclear_Gen'].interpolate(method='linear')
    if 'Hydro_Gen' in df.columns:
        df['Hydro_Gen'] = df['Hydro_Gen'].interpolate(method='linear')
    if 'Thermal_Gen' in df.columns:
        df['Thermal_Gen'] = df['Thermal_Gen'].interpolate(method='linear')
    
    df['Gas_Price'] = df['Gas_Price'].interpolate(method='linear').fillna(35.0)
    df['Market_Coeff'] = df['Market_Coeff'].fillna(1.0)
    df['VDR_Volume'] = df['VDR_Volume'].interpolate(method='linear').fillna(1.0)
    df['Grid_Import_Export'] = df['Grid_Import_Export'].interpolate(method='linear').fillna(0.0)
    
    # Reset index to make Datetime a column again
    df = df.reset_index()
    
    # 4. Time and Calendar Features
    df['Hour'] = df['Datetime'].dt.hour
    df['Month'] = df['Datetime'].dt.month
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['Is_Holiday'] = df['Datetime'].apply(is_ukrainian_holiday)
    df['Is_Weekend_Or_Holiday'] = ((df['Is_Weekend'] == 1) | (df['Is_Holiday'] == 1)).astype(int)
    
    # Cyclical encodings
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)
    
    # Seasonality (Day of Year)
    df['DayOfYear'] = df['Datetime'].dt.dayofyear
    df['DayOfYear_Sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
    df['DayOfYear_Cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)
    
    # Ukrainian Price Zones (Price limits and consumption patterns differ here)
    df['Is_Night'] = df['Hour'].isin([0, 1, 2, 3, 4, 5, 6, 23]).astype(int)
    df['Is_Morning_Peak'] = df['Hour'].isin([7, 8, 9, 10]).astype(int)
    df['Is_Daytime'] = df['Hour'].isin([11, 12, 13, 14, 15, 16]).astype(int)
    df['Is_Evening_Peak'] = df['Hour'].isin([17, 18, 19, 20, 21, 22]).astype(int)
    
    # Lag Features (Now perfectly aligned to exact hours!)
    df['Price_Lag_24'] = df['Price'].shift(24)
    df['Price_Lag_48'] = df['Price'].shift(48)
    df['Price_Lag_168'] = df['Price'].shift(168)
    
    # Rolling features of lags
    df['Price_Mean_24h'] = df['Price'].shift(24).rolling(window=24).mean()
    
    # Weather lags
    for lag in [3, 6]:
        df[f'Temp_Lag_{lag}'] = df['Temperature'].shift(lag)
        df[f'Cloud_Lag_{lag}'] = df['Cloud_Cover'].shift(lag)
        df[f'Radiation_Lag_{lag}'] = df['Shortwave_Radiation'].shift(lag)
        
    # Drop rows with NaN due to lags (first 7 days)
    df = df.dropna().reset_index(drop=True)
    
    return df

def train_models():
    """
    Loads data, trains XGBoost and MLP Regressor, saves them and returns metrics.
    """
    # 1. Load data
    df_raw = dm.get_combined_historical_data()
    df = prepare_features(df_raw)
    
    X = df[FEATURES]
    y = df['Price']
    
    # 2. Train-test split (time-based split, last 15% as test)
    split_idx = int(len(df) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Scale features for evaluation
    scaler_eval = StandardScaler()
    X_train_scaled = scaler_eval.fit_transform(X_train)
    X_test_scaled = scaler_eval.transform(X_test)
    
    # 3. Train Tuned XGBoost on train split
    print("Evaluating XGBoost Regressor on test split...")
    xgb_eval = XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1
    )
    xgb_eval.fit(X_train, y_train)
    
    # Predict and evaluate XGBoost
    y_pred_xgb = xgb_eval.predict(X_test)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)
    
    # 4. Train MLP Regressor (Neural Network) on train split
    print("Evaluating MLP Regressor (Neural Network) on test split...")
    mlp_eval = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp_eval.fit(X_train_scaled, y_train)
    
    # Predict and evaluate MLP
    y_pred_mlp = mlp_eval.predict(X_test_scaled)
    mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
    rmse_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
    r2_mlp = r2_score(y_test, y_pred_mlp)
    
    metrics = {
        'xgboost': {'mae': float(mae_xgb), 'rmse': float(rmse_xgb), 'r2': float(r2_xgb)},
        'mlp': {'mae': float(mae_mlp), 'rmse': float(rmse_mlp), 'r2': float(r2_mlp)}
    }
    
    # Save metrics report
    metrics_path = os.path.join(DATA_DIR, "metrics_report.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"XGBoost Evaluation Metrics: MAE={mae_xgb:.2f}, RMSE={rmse_xgb:.2f}, R2={r2_xgb:.4f}")
    print(f"Neural Network Evaluation Metrics: MAE={mae_mlp:.2f}, RMSE={rmse_mlp:.2f}, R2={r2_mlp:.4f}")
    
    # 5. Retrain final models on the entire dataset for production predictions!
    print("Retraining final XGBoost model on full dataset...")
    xgb_final = XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1
    )
    xgb_final.fit(X, y)
    
    print("Retraining final MLP model on full dataset...")
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    
    mlp_final = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp_final.fit(X_scaled, y)
    
    # Save final models and scaler
    os.makedirs(DATA_DIR, exist_ok=True)
    
    with open(XGB_MODEL_PATH, 'wb') as f:
        pickle.dump(xgb_final, f)
        
    with open(MLP_MODEL_PATH, 'wb') as f:
        pickle.dump(mlp_final, f)
        
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler_final, f)
        
    # Also save primary model (XGBoost) as model.pkl in project root
    with open(DEFAULT_MODEL_PATH, 'wb') as f:
        pickle.dump(xgb_final, f)
        
    print("Final models successfully retrained and saved to disk.")
    return metrics

def predict_next_day(forecast_date, forecast_weather, last_prices, factors):
    """
    Predicts prices for a specific date (24 hours) using the trained models.
    """
    # Load models
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print("Models not trained yet. Training now...")
        train_models()
        
    with open(DEFAULT_MODEL_PATH, 'rb') as f:
        xgb_model = pickle.load(f)
    with open(MLP_MODEL_PATH, 'rb') as f:
        mlp_model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
        
    # Load historical weather to construct weather lags
    df_hist = pd.read_csv(os.path.join(DATA_DIR, "historical_data_merged.csv"))
    df_hist['Datetime'] = pd.to_datetime(df_hist['Datetime'])
    df_hist = df_hist.sort_values('Datetime')
    
    forecast_dt_start = pd.to_datetime(forecast_date)
    hist_before_target = df_hist[df_hist['Datetime'] < forecast_dt_start].sort_values('Datetime')
    
    if len(hist_before_target) >= 24:
        last_temps = hist_before_target['Temperature'].iloc[-24:].tolist()
        last_clouds = hist_before_target['Cloud_Cover'].iloc[-24:].tolist()
        last_rads = hist_before_target['Shortwave_Radiation'].iloc[-24:].tolist()
    else:
        last_temps = [15.0] * 24
        last_clouds = [40.0] * 24
        last_rads = [0.0] * 24
        
    # Prepare forecast inputs
    records = []
    if len(last_prices) < 168:
        print(f"Warning: insufficient last_prices ({len(last_prices)}). Padding with average.")
        mean_p = np.mean(last_prices) if len(last_prices) > 0 else 4000.0
        last_prices = [mean_p] * (168 - len(last_prices)) + list(last_prices)
        
    # We will build features for each hour of the forecast day (0-23)
    for h in range(24):
        dt = pd.to_datetime(forecast_date) + pd.to_timedelta(h, unit='h')
        
        # Weather parameters for this hour
        weather_row = forecast_weather.iloc[h] if h < len(forecast_weather) else forecast_weather.iloc[-1]
        
        # Calculate lag price indices
        lag_24 = last_prices[-24 + h]
        lag_48 = last_prices[-48 + h]
        lag_168 = last_prices[-168 + h]
        
        # Price mean 24h: average price of the rolling 24-hour window shifted by 24 hours
        mean_24h = np.mean(last_prices[121 + h : 145 + h])
        
        # Weather parameters
        rad = float(weather_row.get('Shortwave_Radiation', 0.0))
        clouds = float(weather_row.get('Cloud_Cover', 40.0))
        temp = float(weather_row.get('Temperature', 15.0))
        ws = float(weather_row.get('Wind_Speed', 12.0))
        
        # Weather lags
        temp_lag_3 = float(forecast_weather.iloc[h-3]['Temperature'] if h >= 3 else last_temps[-3 + h])
        temp_lag_6 = float(forecast_weather.iloc[h-6]['Temperature'] if h >= 6 else last_temps[-6 + h])
        
        cloud_lag_3 = float(forecast_weather.iloc[h-3]['Cloud_Cover'] if h >= 3 else last_clouds[-3 + h])
        cloud_lag_6 = float(forecast_weather.iloc[h-6]['Cloud_Cover'] if h >= 6 else last_clouds[-6 + h])
        
        rad_lag_3 = float(forecast_weather.iloc[h-3]['Shortwave_Radiation'] if h >= 3 else last_rads[-3 + h])
        rad_lag_6 = float(forecast_weather.iloc[h-6]['Shortwave_Radiation'] if h >= 6 else last_rads[-6 + h])
        
        solar_strike = float(factors.get('Solar_Strike', 0.0))
        nuke_outage = float(factors.get('Nuclear_Outage', 0.15))
        
        # Solar Gen MW
        solar_gen = np.clip(6500.0 * (rad / 1000.0) * (1.0 - 0.003 * (temp - 25.0)), 0.0, 5500.0)
        solar_gen = solar_gen * (1.0 - solar_strike)
        
        # Wind Gen MW (turbine curve)
        if ws < 8.0 or ws > 80.0:
            wind_gen = 0.0
        elif ws > 45.0:
            wind_gen = 1800.0
        else:
            wind_gen = 1800.0 * ((ws - 8.0) / (45.0 - 8.0)) ** 3
            
        # Nuclear Gen MW
        nuclear_gen = 9500.0 * (1.0 - nuke_outage)
        
        # Cyclical Encodings
        hour_sin = np.sin(2 * np.pi * h / 24.0)
        hour_cos = np.cos(2 * np.pi * h / 24.0)
        month_sin = np.sin(2 * np.pi * dt.month / 12.0)
        month_cos = np.cos(2 * np.pi * dt.month / 12.0)
        
        # Seasonality (Day of Year)
        day_of_year = dt.dayofyear
        day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365.25)
        day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365.25)
        
        # Ukrainian price zones
        is_night = int(h in [0, 1, 2, 3, 4, 5, 6, 23])
        is_morning_peak = int(h in [7, 8, 9, 10])
        is_daytime = int(h in [11, 12, 13, 14, 15, 16])
        is_evening_peak = int(h in [17, 18, 19, 20, 21, 22])
        
        is_we = int(dt.dayofweek in [5, 6])
        is_hol = is_ukrainian_holiday(dt)
        is_we_or_hol = int(is_we == 1 or is_hol == 1)
        
        records.append({
            'Hour': h,
            'Month': dt.month,
            'DayOfWeek': dt.dayofweek,
            'Is_Weekend': is_we,
            'Is_Holiday': is_hol,
            'Is_Weekend_Or_Holiday': is_we_or_hol,
            'Temperature': temp,
            'Cloud_Cover': clouds,
            'Wind_Speed': ws,
            'Shortwave_Radiation': rad,
            'Gas_Price': float(factors.get('Gas_Price', 35.0)),
            'Nuclear_Outage': nuke_outage,
            'Solar_Strike': solar_strike,
            'Market_Coeff': float(factors.get('Market_Coeff', 1.0)),
            'VDR_Volume': float(factors.get('VDR_Volume', 1.0)),
            'Grid_Import_Export': float(factors.get('Grid_Import_Export', 0.0)),
            'Solar_Gen': float(solar_gen),
            'Wind_Gen': float(wind_gen),
            'Nuclear_Gen': float(nuclear_gen),
            'Hour_Sin': hour_sin,
            'Hour_Cos': hour_cos,
            'Month_Sin': month_sin,
            'Month_Cos': month_cos,
            'DayOfYear_Sin': day_of_year_sin,
            'DayOfYear_Cos': day_of_year_cos,
            'Is_Night': is_night,
            'Is_Morning_Peak': is_morning_peak,
            'Is_Daytime': is_daytime,
            'Is_Evening_Peak': is_evening_peak,
            'Price_Lag_24': float(lag_24),
            'Price_Lag_48': float(lag_48),
            'Price_Lag_168': float(lag_168),
            'Price_Mean_24h': float(mean_24h),
            'Temp_Lag_3': temp_lag_3,
            'Temp_Lag_6': temp_lag_6,
            'Cloud_Lag_3': cloud_lag_3,
            'Cloud_Lag_6': cloud_lag_6,
            'Radiation_Lag_3': rad_lag_3,
            'Radiation_Lag_6': rad_lag_6
        })
        
    X_forecast = pd.DataFrame(records)[FEATURES]
    
    # Get predictions
    pred_xgb = xgb_model.predict(X_forecast)
    X_forecast_scaled = scaler.transform(X_forecast)
    pred_mlp = mlp_model.predict(X_forecast_scaled)
    
    # Post-processing range limits (updated to match 2026 peak price caps of 15000 UAH/MWh)
    pred_xgb = np.clip(pred_xgb, 10.0, 16000.0)
    pred_mlp = np.clip(pred_mlp, 10.0, 16000.0)
    
    final_xgb = []
    final_mlp = []
    for h in range(24):
        p_xgb = pred_xgb[h]
        p_mlp = pred_mlp[h]
        
        # Physical surplus check:
        # Midday hours (10:00 to 16:00)
        is_midday = 10 <= h <= 16
        rad = records[h]['Shortwave_Radiation']
        clouds = records[h]['Cloud_Cover']
        is_we = records[h]['Is_Weekend']
        strike = records[h]['Solar_Strike']
        wind = records[h]['Wind_Speed']
        nuke_outage = records[h]['Nuclear_Outage']
        market_coeff = records[h]['Market_Coeff']
        
        # Conditions for solar surplus
        solar_surplus = is_midday and (rad > 500) and (clouds < 25) and (strike < 0.3) and (is_we == 1 or market_coeff < 0.85)
        wind_surplus = (wind > 35) and (is_we == 1)
        
        # If surplus expected, force prices to 10 UAH
        if (solar_surplus or wind_surplus) and nuke_outage < 0.35:
            p_xgb = 10.0
            p_mlp = 10.0
            
        final_xgb.append(float(p_xgb))
        final_mlp.append(float(p_mlp))
        
    return {
        'hours': list(range(24)),
        'xgboost': final_xgb,
        'mlp': final_mlp,
        'features': records
    }
