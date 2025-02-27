# Price Prediction Model Evaluation Metrics

## Overall Metrics
- RMSE: 64.10 öre/kWh
- MAE: 41.43 öre/kWh
- MAPE: 4014.52%
- R²: 0.604
- Correlation: 0.807
- Bias: 9.80 öre/kWh
- Standard Error: 63.35 öre/kWh
- Maximum Error: 668.13 öre/kWh

## Test vs. Validation Set Performance
### Test Set
- RMSE: 42.75 öre/kWh
- MAE: 34.02 öre/kWh
- MAPE: 4993.46%
- R²: 0.301

### Validation Set
- RMSE: 79.94 öre/kWh
- MAE: 48.84 öre/kWh
- MAPE: 3040.12%
- R²: 0.621

## Yearly Performance
### 2022
- RMSE: 130.25 öre/kWh
- MAE: 86.53 öre/kWh
- MAPE: 700.36%
- R²: 0.499

### 2023
- RMSE: 38.35 öre/kWh
- MAE: 30.87 öre/kWh
- MAPE: 3716.89%
- R²: 0.436

### 2024
- RMSE: 42.76 öre/kWh
- MAE: 34.54 öre/kWh
- MAPE: 6198.75%
- R²: 0.124

### 2025
- RMSE: 47.06 öre/kWh
- MAE: 35.99 öre/kWh
- MAPE: 718.64%
- R²: 0.312

## Monthly Performance (MAE in öre/kWh)
| Month | MAE | MAPE | R² |
|-------|-----|------|----|
| Jan | 36.34 | 889.49% | 0.333 |
| Feb | 29.37 | 1552.09% | 0.347 |
| Mar | 21.91 | 513.65% | 0.456 |
| Apr | 23.93 | 1299.22% | 0.342 |
| May | 38.63 | 9136.67% | -0.718 |
| Jun | 31.38 | 4217.69% | -0.315 |
| Jul | 32.54 | 5920.04% | -2.019 |
| Aug | 74.13 | 6527.95% | 0.489 |
| Sep | 70.41 | 8301.09% | 0.559 |
| Oct | 39.76 | 8297.40% | 0.377 |
| Nov | 34.87 | 875.98% | 0.655 |
| Dec | 44.91 | 1001.54% | 0.815 |

## Hourly Performance
| Hour | MAE | MAPE |
|------|-----|------|
| 00:00 | 29.89 | 5022.09% |
| 02:00 | 28.53 | 5023.57% |
| 04:00 | 39.82 | 5971.18% |
| 06:00 | 48.50 | 4286.04% |
| 08:00 | 48.17 | 2582.39% |
| 10:00 | 45.30 | 3374.93% |
| 12:00 | 43.57 | 6588.85% |
| 14:00 | 43.17 | 1864.89% |
| 16:00 | 51.06 | 3146.68% |
| 18:00 | 48.04 | 2473.29% |
| 20:00 | 39.98 | 3601.45% |
| 22:00 | 32.83 | 5779.38% |
