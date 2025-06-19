import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

"""
print(dat.info())

print(dat.isnull().sum())

sns.pairplot(dat[['median_income', 'housing_median_age', 'median_house_value']])
plt.show()

print(dat['longitude'],dat['latitude'])

plt.plot(dat['longitude'],dat['latitude'],)
plt.title("Mapping data")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(dat['longitude'], dat['latitude'], 
            c=dat['median_house_value'], cmap='jet', alpha=0.4, s=10)
plt.colorbar(label='House Value ($)')
plt.title("California Housing Prices")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show() 
"""
# Reading data 
dat = pd.read_csv(r'C:\Users\bsbde\OneDrive\Desktop\Machine Learning\Practise Projects\First Step\housing.csv')
# print(dat.head())


# Preparing my target variable and input features to train model

dat['total_bedrooms'].fillna(dat['total_bedrooms'].median())

X = dat.drop(['median_house_value', 'ocean_proximity'],axis=1)

Y = dat['median_house_value']

# Z-Score Normalisation on the Dataset
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X.select_dtypes(include=['float64', 'int64'])),columns=X.select_dtypes(include=['float64', 'int64']).columns)


# Splitting Data 
xtr , xts , ytr , yts = train_test_split(X_scaled,Y,test_size=0.2,random_state=42)

X = X.fillna(X.mean())
xts = xts.fillna(xts.mean())

model = LinearRegression()

model.fit(xtr,ytr)

ypred = model.predict(xts)

for a in range(len(ypred[:10])):  # just printing 10 for brevity
    print(f"Predicted: {ypred[a]:.2f} | Actual: {yts.iloc[a]:.2f}")

accuracy = model.score(xts,yts) * 100

print(f"The model predicts with an accuracy : {accuracy:.3f} %")