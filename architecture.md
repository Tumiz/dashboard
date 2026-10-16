This is a development guide that code agents should follow.

1. The dashboard shows both historical data and predicted gold prices. It must display two forecasts and their ground truth:
   - a monthly forecast for the target month T, and
   - a quarterly forecast for month T + 2 (the quarterly horizon ends two months past T).

2. xau_predictor.py owns all forecast computation and writes the results to predictions.json. index.html only reads and renders that JSON; it must not compute forecasts or hardcode model coefficients. The model equations shown on the dashboard are stored in predictions.json (models.monthly.equation and models.quarterly.equation); the HTML contains no coefficients. The actual return series shown on the dashboard are also computed by xau_predictor.py and stored in predictions.json as actual_returns.monthly (trailing one-month log returns) and actual_returns.quarterly (trailing three-month log returns); the HTML does not compute them.

3. Significant events related to gold prices are stored in xau_events.js. Only breaking news, not documents or conclusion articles.

4. xau_predictor.py accepts one optional argument: the target month T to predict, in YYYY-MM format. The default target month is the current month. T must not be later than the current month: the month before a future T may still be month-to-date rather than a final close.

5. Forecasts must not use look-ahead data, and the regression for each forecast is fit on only the data it is allowed to use:
   - the monthly forecast for target month T uses only data from months before T; its predictors come from the latest month before T that has both TWEXB and real yield, plus the month before that, which supply the DXY log return and the real-yield change;
   - the quarterly forecast for month T + 2 uses only data from months before T. A quarterly "quarter" is the rolling window of the three months before the cutoff: (T - 3, T - 2, T - 1). Quarterly features are the window's summed DXY log returns and summed real-yield changes, plus the latest GPR observation inside the window. The quarterly regression is fit on complete three-month windows ending before T. If the latest window is incomplete, use the most recent complete window before it; if none exists, there is no quarterly forecast.

   Both the monthly anchor and the quarterly anchor are the last actual monthly XAU/USD close before T. Never use a forecast month's own XAU/USD value, which may be month-to-date rather than a final monthly close.

6. Every month carries two predictions: one from the monthly model and one from the quarterly model, each computed walk-forward. The monthly prediction for month m is produced exactly like the rule 5 monthly forecast with T = m: refit on data before m, with anchor and predictors before m. The quarterly prediction for month m applies rule 5 with T = m - 2. Both predicted-price series are stored in predictions.json as monthly_series and quarterly_series, each mapping a month to {predicted_price, predicted_return}, with one entry per month whose data is sufficient, and both are plotted by the dashboard as separate predicted-price lines. The headline forecasts from rules 1 and 4 are included at month T (monthly_series) and month T + 2 (quarterly_series).
