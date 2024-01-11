import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error 
from scipy import stats
import seaborn as sns
import warnings
%matplotlib inline
import types
import pandas as pd
from botocore.client import Config

df = pd.read_csv("Life Expectancy Data.csv")
print(df.head())
print(df.shape)
print(df.info())

df = df.drop(['Country'], axis=1)

new_df=df.fillna(df.mean())
new_df.isnull().sum()

new_df.replace(to_replace=['Developing', 'Developed'],
           value= [0, 1], 
           inplace=True)

sns.pairplot(new_df)
sns.distplot(new_df['Life expectancy '])
fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(ax=ax, data=new_df.corr())

columns = {1: 'Year', 2: 'Life expectancy ', 3: 'Adult Mortality', 4: 'infant deaths',
        5: 'Alcohol' , 6: 'percentage expenditure', 7: 'Hepatitis B',
       8: 'Measles ', 9: ' BMI ', 10: 'under-five deaths ', 11: 'Polio', 12: 'Total expenditure',
       13: 'Diphtheria ', 14: ' HIV/AIDS', 15: 'GDP', 16: 'Population',
       17: ' thinness  1-19 years', 18: ' thinness 5-9 years',
       19: 'Income composition of resources', 20: 'Schooling'}

plt.figure(figsize=(28, 30))

for i, column in columns.items():
                     plt.subplot(4,5,i)
                     sns.boxplot(new_df[column], orient='v')
                     plt.title(column)

plt.show()

X = new_df[['Year', 'Status', 'Adult Mortality',
       'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
       'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
       'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
       ' thinness  1-19 years', ' thinness 5-9 years',
       'Income composition of resources', 'Schooling']].values
y = new_df['Life expectancy '].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
new_df.shape
X_train.shape,X_test.shape,y_train.shape,y_test.shape
print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
lm = LinearRegression()
lm.fit(X_train,y_train)
lmpredictions = lm.predict(X_test)

mse = mean_squared_error(y_test,lmpredictions)
mae = mean_absolute_error(y_test,lmpredictions)
r2 = r2_score(y_test, lmpredictions)
print("Mean Squared Error:",mse)
print("Mean Absolute Error:",mae)
print("R2 Square:",r2)

plt.scatter(y_test,lmpredictions)

sns.distplot((y_test-lmpredictions),bins=50);
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 40, random_state = 50)
rf.fit(X_train, y_train)
rfpredictions= rf.predict(X_test)
mse = mean_squared_error(y_test,rfpredictions)
mae = mean_absolute_error(y_test,rfpredictions)
r2 = r2_score(y_test, rfpredictions)
print("Mean Squared Error:",mse)
print("Mean Absolute Error:",mae)
print("R2 Square:",r2)
plt.scatter(y_test, rfpredictions)
sns.distplot((y_test-rfpredictions),bins=50);

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
dt_predictions = dt.predict(X_test)
print(dt_predictions)
mse = mean_squared_error(y_test,dt_predictions)
mae = mean_absolute_error(y_test,dt_predictions)
r2 = r2_score(y_test, dt_predictions)
print("Mean Squared Error:",mse)
print("Mean Absolute Error:",mae)
print("R2 Square:",r2)

sns.distplot((y_test-dt_predictions),bins=50);
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(X_train, y_train)
ridge_predictions = ridge.predict(X_test)
mae = mean_absolute_error(y_test, ridge_predictions)
r2 = r2_score(y_test, ridge_predictions)
mse = mean_squared_error(y_test,ridge_predictions)
print("Mean absolute Error (Ridge Regression):", mae)
print("Mean squared Error (Ridge Regression):",mse)
print("R-squared Score (Ridge Regression):", r2)


features = {
    'Year': [2023],
    'Status': [1],
    'Adult Mortality': [200],
    'infant deaths': [100],
    'Alcohol': [10],
    'percentage expenditure': [100],
    'Hepatitis B': [90],
    'Measles ': [50],
    ' BMI ': [25],
    'under-five deaths ': [15],
    'Polio': [95.23],
    'Total expenditure': [7.5],
    'Diphtheria ': [94],
    ' HIV/AIDS': [2.4],
    'GDP': [1500],
    'Population': [1000],
    ' thinness  1-19 years': [7],
    ' thinness 5-9 years': [1.08],
    'Income composition of resources': [3],
    'Schooling': [11]
}
input_df = pd.DataFrame(features)
input_df['Status'].replace(to_replace=['Developing', 'Developed'], value=[0, 1], inplace=True)
predicted_life_expectancy = rf.predict(input_df)
predicted_life_expectancy1 = dt.predict(input_df)
predicted_life_expectancy2 = lm.predict(input_df)
predicted_life_expectancy3 = ridge.predict(input_df)
print("Predicted life expectancy using Random Forest:", int(predicted_life_expectancy))
print("Predicted life expectancy using Decision Tree:", int(predicted_life_expectancy1))
print("Predicted life expectancy using Linear Regression:", int(predicted_life_expectancy2))
print("Predicted life expectancy using Ridge Regression:", int(predicted_life_expectancy3))