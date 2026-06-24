import pulp
import numpy as np

def optimize_battery_schedule(
    prices,
    battery_capacity=1000.0,      # kWh
    max_charge_power=250.0,       # kW
    max_discharge_power=250.0,    # kW
    charge_efficiency=0.95,       # 0.0 to 1.0
    discharge_efficiency=0.95,    # 0.0 to 1.0
    initial_soc=0.20,             # 0.0 to 1.0 (fraction of capacity)
    min_soc=0.10,                 # 0.0 to 1.0 (fraction of capacity)
    max_soc=0.90,                 # 0.0 to 1.0 (fraction of capacity)
    max_cycles_per_day=1.5,       # equivalent full cycles
    degradation_cost=1.20,        # UAH per kWh discharged
    transmission_tariff=528.57,   # UAH/MWh
    distribution_tariff=1500.0,   # UAH/MWh
    dispatch_tariff=104.57,       # UAH/MWh
    supplier_margin=100.0,        # UAH/MWh
    mode='arbitrage'              # 'arbitrage' (FTM) or 'self_consumption' (BTM)
):
    """
    Optimizes battery charging and discharging schedule using PuLP Linear Programming.
    
    Parameters:
    - prices: list of 24 hourly prices (UAH/MWh)
    """
    # 1. Initialize LP problem
    prob = pulp.LpProblem("Battery_Schedule_Optimization", pulp.LpMaximize)
    
    # 2. Decision variables
    # x[t] - charge power (kW) during hour t
    # y[t] - discharge power (kW) during hour t
    # soc[t] - State of Charge (kWh) at the end of hour t
    x = pulp.LpVariable.dicts("Charge", range(24), lowBound=0, upBound=max_charge_power)
    y = pulp.LpVariable.dicts("Discharge", range(24), lowBound=0, upBound=max_discharge_power)
    soc = pulp.LpVariable.dicts("SoC", range(24), lowBound=min_soc * battery_capacity, upBound=max_soc * battery_capacity)
    
    # Total tariffs in UAH/MWh
    total_tariffs = transmission_tariff + distribution_tariff + dispatch_tariff + supplier_margin
    
    # Convert to UAH/kWh
    total_tariffs_kwh = total_tariffs / 1000.0
    
    # Formulate buy and sell prices for each hour in UAH/kWh
    p_buy = []
    p_sell = []
    
    for t in range(24):
        dam_kwh = prices[t] / 1000.0
        
        # Buy price is always DAM price + Tariffs
        p_buy.append(dam_kwh + total_tariffs_kwh)
        
        if mode == 'arbitrage':
            # Sell price is DAM price only
            p_sell.append(dam_kwh)
        else: # self_consumption / behind the meter
            # Discharging saves the full retail price (DAM + Tariffs)
            p_sell.append(dam_kwh + total_tariffs_kwh)
            
    # 3. Constraints
    # Initial SoC relationship at hour 0
    init_soc_kwh = initial_soc * battery_capacity
    prob += soc[0] == init_soc_kwh + x[0] * charge_efficiency - y[0] / discharge_efficiency, "SoC_Hour_0"
    
    # SoC relationship for hours 1 to 23
    for t in range(1, 24):
        prob += soc[t] == soc[t-1] + x[t] * charge_efficiency - y[t] / discharge_efficiency, f"SoC_Hour_{t}"
        
    # Cycle limit constraint: total energy discharged <= max_cycles * capacity
    # Total discharge throughput in kWh
    total_discharge = pulp.lpSum([y[t] for t in range(24)])
    prob += total_discharge <= max_cycles_per_day * battery_capacity, "Daily_Cycle_Limit"
    
    # Final SoC constraint: battery should end the day at or above initial SoC
    prob += soc[23] >= init_soc_kwh, "Final_SoC_Balance"
    
    # 4. Objective Function: Maximize net financial benefit
    # Revenue = selling (discharging) * sell_price - buying (charging) * buy_price - degradation_cost * discharge
    revenue = pulp.lpSum([p_sell[t] * y[t] for t in range(24)])
    cost_charging = pulp.lpSum([p_buy[t] * x[t] for t in range(24)])
    cost_degradation = pulp.lpSum([degradation_cost * y[t] for t in range(24)])
    
    prob += revenue - cost_charging - cost_degradation
    
    # 5. Solve the LP problem
    # Use default solver (CBC)
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Extract results
    charge_schedule = [x[t].varValue if x[t].varValue is not None else 0.0 for t in range(24)]
    discharge_schedule = [y[t].varValue if y[t].varValue is not None else 0.0 for t in range(24)]
    soc_schedule = [soc[t].varValue if soc[t].varValue is not None else init_soc_kwh for t in range(24)]
    
    # Calculate performance metrics
    total_cost_charging = sum(p_buy[t] * charge_schedule[t] for t in range(24))
    total_revenue_discharging = sum(p_sell[t] * discharge_schedule[t] for t in range(24))
    total_degradation = sum(degradation_cost * discharge_schedule[t] for t in range(24))
    net_profit = total_revenue_discharging - total_cost_charging - total_degradation
    
    actual_cycles = sum(discharge_schedule) / battery_capacity
    
    return {
        'status': pulp.LpStatus[status],
        'charge': charge_schedule,
        'discharge': discharge_schedule,
        'soc': [init_soc_kwh] + soc_schedule, # Include starting SoC
        'net_profit_uah': float(net_profit),
        'cost_charging_uah': float(total_cost_charging),
        'revenue_discharging_uah': float(total_revenue_discharging),
        'degradation_cost_uah': float(total_degradation),
        'cycles_used': float(actual_cycles),
        'hourly_buy_prices': [float(p) * 1000.0 for p in p_buy], # UAH/MWh
        'hourly_sell_prices': [float(p) * 1000.0 for p in p_sell] # UAH/MWh
    }
