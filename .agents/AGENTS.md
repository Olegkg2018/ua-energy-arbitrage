# UA Energy Arbitrage Project Rules & Session Summary

## 📌 Project Overview
This project is an EMS (Energy Management System) and price forecasting simulator for industrial Battery Energy Storage Systems (BESS) operating on the Ukrainian Day-Ahead Market (DAM / РДН).

---

## ⚙️ Core Technical Specifications
- **BESS Capacity**: 4000 kWh.
- **BESS Power**: 1000 kW (max charge/discharge limit).
- **Physical Boundaries**:
  - **Min SoC**: 10% (400 kWh) to prevent over-discharge.
  - **Max SoC**: 90% (3600 kWh) to prevent over-charge.
  - **Initial SoC / Вхідна ємність**: Configurable in sidebar (defaults to 20% = 800 kWh).
- **Roundtrip Efficiency**: 95% charge, 95% discharge (total roundtrip ~90.25%).
- **Cell Degradation Cost**: 1.20 UAH/kWh discharged (amortized wear cost).

---

## 📡 Operational Rules & Pricing Math
1. **Grid Arbitrage (FTM)**:
   - Buy price: $\text{DAM} + \text{Transmission} + \text{Distribution} + \text{Dispatch} + \text{Margin}$.
   - Sell price: $\text{DAM}$.
2. **Behind-the-Meter (BTM / Заміщення споживання)**:
   - Buy price: $\text{DAM} + \text{Tariffs}$.
   - Sell price (avoided cost): $\text{DAM} + \text{Tariffs}$.
   - Tariffs cancel out, making the required price spread much lower: $\Delta \text{DAM} > \text{degradation} / 0.9025 \approx 1.33$ UAH/kWh.

---

## 🎨 User Interface & Drag Mechanics
- **Tabbed Navigation**: 5 distinct sections (*Панель інструментів, Погодинний графік, Сценарний аналіз, Довідник тарифів та АКБ, Журнал імпортів та помилок*).
- **Languages**: 
  - User interaction/explanations: **Russian**.
  - User interface, tables, logs, and README: **Ukrainian**.
- **Interactive Drag-and-Drop Power Schedule**:
  - **Vertical drag**: Changes the charge/discharge power (kW) for the column.
  - **Horizontal drag (index-shifting)**: Shifts the whole column (power and mode) to another hour.
  - **Canvas-relative coordinate translation**: Converts client coordinates relative to bounding client rect of the canvas to support zoom/scrolling and mobile touches.
  - **Snapshot protection**: Keeps a full schedule snapshot at `onDragStart` and restores it at `onDragEnd` before moving, protecting adjacent columns from disappearing.
  - **Deferred recreation**: Destroys and reconstructs the chart inside `setTimeout(..., 50)` to prevent Chart.js event-loop lifecycle crashes.
- **BMS Auto-Balancing**:
  - **Forward Pass**: Mutually exclusive check, caps charge/discharge to prevent SoC violations ($10\% - 90\%$).
  - **Backward Pass**: Eliminates final SoC deviations, scaling down evening discharge (if deficit) or evening charge (if surplus) to return SoC to starting level.

---

## 🧠 Machine Learning Forecasting & Scenario Overrides
- **Autoregressive Features**: The ML models prioritize lags (`Price_Lag_24`, `Price_Lag_48`, `Price_Lag_168`).
- **Physical Overrides (Post-processing)**:
  - **Gas Price**: $\approx 15$ UAH/MWh price shift per 1 EUR/MWh gas price change from 35 EUR/MWh base.
  - **Nuclear Outage**: 4000 UAH/MWh price shift per 10% outage change from 15% base.
  - **Grid Import**: -300 UAH/MWh price shift per 1000 MW import.
  - **Market Coeff**: Direct scaling multiplier for the predicted curve.
  - **Solar Strike**: Increases daytime prices if solar generation is damaged.
