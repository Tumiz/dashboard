This is a development guide that code agents should follow.

1. The dashboard shows both historical data and predicted gold prices. It must display two forecasts:
   - a monthly forecast for the target month T, and
   - a quarterly forecast for month T + 2.

2. xau_predictor.py owns all forecast computation and writes the results to predictions.json. index.html only reads and renders that JSON; it must not compute forecasts or hardcode model coefficients.

3. Significant events related to gold prices are stored in xau_events.js. Only breaking news, not documents or conclusion articles.

4. xau_predictor.py accepts one optional argument: the target month T to predict, in YYYY-MM format. The default target month is the current month.

5. Forecasts must not use look-ahead data, and the regression for each forecast is fit on only the data it is allowed to use:
   - the monthly forecast for target month T uses only data from months before T;
   - the quarterly forecast for month T + 2 uses only data from months before T.

   The anchor price is the last actual monthly XAU/USD close before the forecast month. Never use a forecast month's own XAU/USD value, which may be month-to-date rather than a final monthly close.
