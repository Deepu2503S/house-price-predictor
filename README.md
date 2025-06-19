# 🏡 House Price Prediction

This is my first Machine Learning project — a simple **Linear Regression model** to predict California housing prices based on various features such as location, income, and demographics.

## 📂 Dataset

- 📁 `housing.csv`: California Housing Dataset
- Source: [California Housing Data](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)

## 🔍 Features Used

- `longitude`
- `latitude`
- `housing_median_age`
- `total_rooms`
- `total_bedrooms` (missing values filled)
- `population`
- `households`
- `median_income`

(Note: `ocean_proximity` was excluded for simplicity and I didn't had enough knowledge of dealing with String values at the moment.)

## ⚙️ Preprocessing

- Missing values handled
- Standardization using `StandardScaler` (Z-score normalization)
- Train-test split using `train_test_split` (80-20)

## 📈 Model

- Model: `LinearRegression()` from `sklearn`
- Performance Metric: R² Score (around **61.440%** on the test set)

## 💻 Output Example

```bash
Predicted: 75389.70 | Actual: 47700.00
Predicted: 165997.71 | Actual: 45800.00
Predicted: 264726.31 | Actual: 500001.00
Predicted: 275160.86 | Actual: 218600.00
Predicted: 278536.47 | Actual: 278000.00
Predicted: 167056.61 | Actual: 158700.00
Predicted: 298711.77 | Actual: 198200.00
Predicted: 235313.89 | Actual: 157500.00
Predicted: 268698.14 | Actual: 340000.00
Predicted: 415926.48 | Actual: 446600.00
The model predicts with an accuracy : 61.440 %
