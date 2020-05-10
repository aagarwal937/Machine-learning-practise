import numpy as np
from numpy.random import randn
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly as py
import sklearn

USAHousing = pd.read_csv('USA_Housing.csv')

# SIMPLY ANALYSING THE DATA

print(USAhousing.head())

print(USAhousing.info())

print(USAhousing.describe())

print(USAhousing.columns)

sns.pairplot(USAhousing)
sns.distplot(USAhousing['Price'])
sns.heatmap(USAhousing.corr())

# LINEAR REGRESSION MODEL


X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']

from sklearn.model_selection import train_test_split

print(X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.4, random_state=101))

# Creating atraining model

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)

# Model Evaluation

print(lm.intercept_)

coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

# Predictions from our Model

predictions = lm.predict(X_test)
print('predictions')

plt.scatter(y_test, predictions)

sns.distplot((y_test - predictions), bins=50)

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

plt.show()
