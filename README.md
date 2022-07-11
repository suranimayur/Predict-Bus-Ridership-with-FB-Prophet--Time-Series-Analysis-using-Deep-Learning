# Predict-Bus-Ridership-with-FB-Prophet--Time-Series-Analysis-using-Deep-Learning
Predict Bus Ridership with FB Prophet- Time Series Analysis using Deep Learning

Hellow Friend , Today we will make time series analysis with python project for Bus Ridership with Facebook Prophet.

Step-01 Import necessary modules and library for time series analysis in Python.

import numpy as np
import pandas as pd
from itertools import product
from fbprophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from fbprophet.plot import plot_cross_validation_metric
from fbprophet.diagnostics import cross_validation, performance_metrics


plt.rcParams['figure.figsize']=(10,7.5)

Step 03 Read source file in python IDE. You can use jyputer notebook , google colab or visual code studio for projects( As these three are most userfriendly IDEs for python)

df= pd.read_csv("/content/portland-oregon-average-monthly-.csv")
df.head()


Step 04 rename columns

df.columns=['ds','y']
df.head()

df.tail()

df.describe().transpose()


Step 05- Dattype of Date is object. So, Need to change its data time to timeframe

df["ds"]= pd.to_datetime(df['ds'])

df.info()

Step 05- Make plot of Monthly Bus Ridership in Portlend Oregon

fig, ax= plt.subplots()

locator=mdates.AutoDateLocator()

ax.plot(df.ds,df.y)
ax.xaxis.set_major_locator(locator)

ax.set_xlabel("Date")
ax.set_ylabel("Bus Ridership")

ax.set_title("MOnthly Bus Ridership in Portland Oregon")

fig.autofmt_xdate()
fig.tight_layout()


Step-06 -- HyperParameter Tunning

param_grid={
    
    "changepoint_prior_scale": [0.001,0.01,0.1,0.5],
    "seasonality_prior_scale": [0.01,0.1,1.0,10.0]
}

params= [dict(zip(param_grid.keys(),v)) for v in product(*param_grid.values())]

rmses=[]

cutoffs= pd.date_range(start= "1961-01-01", end= "1969-06-01", freq= "6MS")

for param in params:
  m= Prophet(**param)
  m.add_country_holidays(country_name= "US")
  m.fit(df)

  df_cv= cross_validation(model=m, horizon= "365 days",cutoffs=cutoffs)
  df_p=performance_metrics(df_cv,rolling_window=1)
  rmses.append(df_p["rmse"].values[0])

tunning_results= pd.DataFrame(params)
tunning_results['rmse']= rmses

best_params= params[np.argmin(rmses)]

print(best_params)

It will take up to 30 minutes to give best_params

Step 06 - Fit Best Model

m= Prophet(
    changepoint_prior_scale=0.01,
    seasonality_prior_scale=0.01
)

m.add_country_holidays(country_name="US")

m.fit(df);


Step-07  Make Forecast

Now, make forecast for next 12 months

future= m.make_future_dataframe(periods=12, freq="M")

future.tail()

forecast= m.predict(future)

forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()

-- plot forecast

forecast_fig= m.plot(forecast)

-- check components of forecast

components_fig= m.plot_components(forecast)

-- Performance Metrics

-- Plot performance of matrics

  df_cv= cross_validation(model=m, horizon= "365 days",cutoffs=cutoffs)

  df_p= performance_metrics(df_cv)

  fig= plot_cross_validation_metric(df_cv,metric= "mape")
  
  

