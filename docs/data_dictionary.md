# Data dictionary

## Raw columns

| Column | Role | Type | Treatment |
|---|---|---|---|
| DateTime | timestamp | datetime | Parse, sort, validate 10-minute continuity |
| Temperature | weather feature | numeric | Hourly mean; only lagged values enter 1h-ahead model |
| Humidity | weather feature | numeric | Hourly mean; only lagged values enter model |
| Wind Speed | weather feature | numeric | Hourly mean; only lagged values enter model |
| general diffuse flows | weather feature | numeric | Hourly mean; only lagged values enter model |
| diffuse flows | weather feature | numeric | Hourly mean; only lagged values enter model |
| Zone 1 Power Consumption | load target component | numeric | Hourly mean |
| Zone 2 Power Consumption | load target component | numeric | Hourly mean |
| Zone 3 Power Consumption | load target component | numeric | Hourly mean |

## Derived target

`system_load = zone_1_hourly_mean + zone_2_hourly_mean + zone_3_hourly_mean`.

This is treated as a combined load index. Do not label it as hourly energy unless source units and integration are verified.

## Main engineered features

- Calendar: hour, day-of-week, month, day-of-year, weekend, cyclical encodings.
- Load lags: 1, 2, 3, 6, 12, 24, 48, 72, 168 hours.
- Load rolling statistics: mean/std/max over 3, 6, 12, 24, 168 hours, shifted by one hour.
- Weather history: 1h and 24h lags; rolling means over 6h and 24h, shifted by one hour.
- Seasonal-naive benchmark: system load at T-168h.
