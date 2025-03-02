# Price Prediction Model Evaluation Metrics

## Overall Metrics
- RMSE: 66.59 öre/kWh
- MAE: 42.77 öre/kWh
- MAPE: 3801.52%
- R²: 0.573
- Correlation: 0.769
- Bias: 9.09 öre/kWh
- Standard Error: 65.96 öre/kWh
- Maximum Error: 693.31 öre/kWh

## Test vs. Validation Set Performance
### Test Set
- RMSE: 47.44 öre/kWh
- MAE: 35.64 öre/kWh
- MAPE: 4852.55%
- R²: 0.135

### Validation Set
- RMSE: 81.34 öre/kWh
- MAE: 49.90 öre/kWh
- MAPE: 2755.47%
- R²: 0.608

## Yearly Performance
### 2022
- RMSE: 130.19 öre/kWh
- MAE: 85.07 öre/kWh
- MAPE: 593.05%
- R²: 0.499

### 2023
- RMSE: 43.31 öre/kWh
- MAE: 33.78 öre/kWh
- MAPE: 3395.67%
- R²: 0.280

### 2024
- RMSE: 47.36 öre/kWh
- MAE: 35.76 öre/kWh
- MAPE: 6045.30%
- R²: -0.074

### 2025
- RMSE: 49.76 öre/kWh
- MAE: 35.82 öre/kWh
- MAPE: 544.17%
- R²: 0.225

## Monthly Performance (MAE in öre/kWh)
| Month | MAE | MAPE | R² |
|-------|-----|------|----|
| Jan | 35.84 | 827.95% | 0.311 |
| Feb | 30.68 | 1207.88% | 0.267 |
| Mar | 24.28 | 328.47% | 0.343 |
| Apr | 30.83 | 1401.00% | -0.028 |
| May | 45.55 | 9555.83% | -1.734 |
| Jun | 37.36 | 3506.31% | -1.105 |
| Jul | 36.83 | 5008.96% | -3.566 |
| Aug | 71.76 | 6290.92% | 0.474 |
| Sep | 67.65 | 8154.23% | 0.565 |
| Oct | 36.76 | 7973.08% | 0.396 |
| Nov | 33.67 | 783.79% | 0.678 |
| Dec | 49.68 | 882.96% | 0.781 |

## Hourly Performance
| Hour | MAE | MAPE |
|------|-----|------|
| 00:00 | 28.86 | 5061.34% |
| 02:00 | 25.46 | 4085.87% |
| 04:00 | 38.14 | 4333.27% |
| 06:00 | 48.70 | 4137.53% |
| 08:00 | 49.30 | 2548.55% |
| 10:00 | 48.01 | 3573.44% |
| 12:00 | 46.68 | 6028.34% |
| 14:00 | 45.57 | 1815.16% |
| 16:00 | 53.17 | 3536.22% |
| 18:00 | 51.30 | 2422.98% |
| 20:00 | 43.10 | 4650.21% |
| 22:00 | 33.65 | 6301.67% |
