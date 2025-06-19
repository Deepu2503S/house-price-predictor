"""# Preparing my target variable and input features to train model

dat['total_bedooms'].fillna(dat['total_bedooms'].median(),inplace=True)

X = dat.drop(dat['median_house_value'],axis=1)
Y = dat['median_house_value']

# Z-Score Normalisation on the Dataset
X_scaled = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)

# Splitting Data 
xtr , xts , ytr , yts = train_test_split(X_scaled,Y,test_size=0.2,random_state=42)

model = LinearRegression()

model.fit(xtr,ytr)

ypred = model.predict(xts)

accuracy = model.score(yts,ypred) * 100

print(f"The model predicts with an accuracy : {accuracy}%")"""