# Power-consumption-forecast
Forecasting power consumption to estimate bidding prices in the Norwegian Nord Pool bidding areas using different deep learning techniques.


# Notes
- Seasonal patters -> Daily, weekly and yearly

# Requirements

### General
- Must be able to forecast 24 hours into the future with a resolution of 1 hour.
- Must be able to predict on each individual bidding area separately (5 specific models or 1 generic model).
- Must be reproducable, and model should be pre-trained for demonstration.
- A model must be trained on a bidding area, and tested on another along with a reflection of the results. 
- At least three of four different forecasting models:
    LSTM (Long Short-Term Memory) at least one layer
    GRU (Gated Recurrent Unit) at least one layer
    CNN (Convolutional Neural Network)
    Transformer
    Feed-forward regression model
- Must add reqularizers (e.g. dropout, L1 or L2)

### Feature engineering
Features must be implemented, visualized and discussed. Not all must be used in the final model, but it should be easy to add/remove them.
The features to include is:
- time_of_day
- time_of_week
- time_of_year
- comsumption_lag_24
- temperature_lag_24
- comsumption_mean_24
- temperature_mean_24
- previous_y (for RNN)

### Visualization
- Training progress
- Observed/forecasted comsumptions (48 hour plots where 24 last is observed/forecasted)
- An error plot showing the mean and standard deviation of the error for each hour of the forecast horizon over the test set. This plot will have forecast horizon (1-24 hours) on the x axis and absolute error on the y axis.
- A summary bar plot that compares the different models you have trained using metrics of your choice (e.g. test set RMSE, MAE, MAPE or similar).
