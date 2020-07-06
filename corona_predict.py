import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

confirmed_df = pd.read_csv('datasets_494724_1296467_time_series_covid_19_confirmed.csv')
deaths_df = pd.read_csv('datasets_494724_1296467_time_series_covid_19_deaths.csv')
recoveries_df = pd.read_csv('datasets_494724_1296467_time_series_covid_19_recovered.csv')

print(confirmed_df.head())

columns = confirmed_df.columns

confirmed = confirmed_df.iloc[:,4:]
deaths = deaths_df.iloc[:,4:]
recoveries = recoveries_df.iloc[:,4:]

dates = confirmed.columns
world_cases = []
for i in dates:
    confirmed_sum = confirmed[i].sum()
    world_cases.append(confirmed_sum)


days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)

days_in_future = 15
future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)

x=days_since_1_22
y=np.ravel(world_cases)
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.15,shuffle=False)

# print(np.ravel(y_train))

#Support vector regressor
model=SVR(shrinking=False, kernel='poly',gamma=0.1,epsilon=1,C=10)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

plt.plot(y_pred)
plt.plot(y_test)

