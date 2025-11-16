\# Energy Consumption Forecasting



Forecast household energy consumption using historical data and time-series modeling.



\## Overview

This project analyzes electricity usage data from an individual household to build a predictive model for daily energy consumption. The workflow demonstrates a complete time-series forecasting pipeline.

Note: The dataset is not included in this repository due to size limits.
You can download it here: <https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption>
Key steps include:


Key steps include:



\- Data loading and cleaning  

\- Handling missing values and converting data types  

\- Exploratory data analysis (EDA) and visualization of trends  

\- Resampling and aggregation to daily consumption  

\- Time-series decomposition (trend, seasonality, residuals)  

\- SARIMA forecasting for daily energy consumption  

\- Model evaluation using MAE, RMSE, and MAPE  

\- Visualizing actual vs. predicted consumption







---



\## Key Steps Performed



&nbsp;\*\*Data Preprocessing\*\*

&nbsp;  - Converted Date and Time columns to datetime index  

&nbsp;  - Handled missing values using forward and backward fill  



\*\*Exploratory Data Analysis\*\*

&nbsp;  - Visualized global active power trends  

&nbsp;  - Decomposed time series to identify trend and seasonality  



\*\*Modeling\*\*

&nbsp;  - Applied SARIMA model for forecasting daily consumption  



\*\*Evaluation\*\*

&nbsp;  - Computed MAE, RMSE, and MAPE  

&nbsp;  - Visualized forecast vs actual consumption



---



\## Results Summary

\- The SARIMA model captured trends and seasonal patterns in daily energy usage  

\- Forecast accuracy was reasonable as indicated by MAE, RMSE, and MAPE metrics  

\- Time-series decomposition highlighted weekday/weekend differences and seasonal effects



---



\## Future Work

\- Hyperparameter tuning for SARIMA or trying Prophet/LSTM models  

\- Forecasting at higher granularity (hourly)  

\- Incorporating external variables like temperature or holidays to improve predictions



---





