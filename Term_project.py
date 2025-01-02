import pandas as pd
import numpy as np
from numpy import array, mean
from sklearn.linear_model import LinearRegression
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

pd.set_option('display.precision', 3)

print(GREEN + "\n ----------------------- Phase I: Feature Engineering & EDA : -----------------------\n" + GREEN)

car_prices_df = pd.read_csv("car_prices.csv")
print("info: \n", car_prices_df.info())
print("Describe: \n", car_prices_df.describe())
print("shape of the database ", car_prices_df.shape)
print("columns : ", car_prices_df.columns)

missing_values = car_prices_df.isna().sum()
print("missing values in each column:\n", missing_values.sort_values(ascending=False))
print("total number of missing values in the dataset:", car_prices_df.isna().sum().sum())

missing_values_per_column = car_prices_df.isna().sum()
print("percentage of missing values per column:\n", (missing_values_per_column / car_prices_df.shape[0]) * 100)
car_prices_df = car_prices_df.dropna(subset=['sellingprice'])

print("missing values:", car_prices_df.isna().sum())
for column in car_prices_df.select_dtypes(include=['number']).columns:
    car_prices_df[column].fillna(car_prices_df[column].mean(), inplace=True)

for column in car_prices_df.select_dtypes(exclude=['number']).columns:
    print("mode of the column", car_prices_df[column].mode()[0])
    car_prices_df[column].fillna(car_prices_df[column].mode()[0], inplace=True)

car_prices_df['saledate'] = pd.to_datetime(car_prices_df['saledate'], errors='coerce', utc=True)
car_prices_df['saleyear'] = car_prices_df['saledate'].dt.year
car_prices_df['saleyear'] = car_prices_df['saleyear'].fillna(-1).astype(int)

print(car_prices_df['saleyear'].head())

car_prices_df = car_prices_df.drop('saledate', axis=1)
print("total number of missing values after filling: ", car_prices_df.isna().sum())

duplicate_rows = car_prices_df[car_prices_df.duplicated()]
print(duplicate_rows.shape)
print("there is no duplicate data in the given dataset")

plt.figure(figsize=(10, 6))
plt.scatter(car_prices_df['odometer'], car_prices_df['sellingprice'], alpha=0.5)
plt.title('Scatter Plot of Odometer vs. Selling Price')
plt.xlabel('Odometer (Miles)')
plt.ylabel('Selling Price ($)')
plt.grid(True)
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

ax[0].hist(car_prices_df['odometer'], bins=30, color='skyblue', edgecolor='black')
ax[0].set_title('Histogram of Odometer')
ax[0].set_xlabel('Odometer (miles)')
ax[0].set_ylabel('Frequency')

# Histogram of Selling Price
ax[1].hist(car_prices_df['sellingprice'], bins=30, color='lightgreen', edgecolor='black')
ax[1].set_title('Histogram of Selling Price')
ax[1].set_xlabel('Selling Price ($)')
ax[1].set_ylabel('Frequency')

# Histogram of Year
ax[2].hist(car_prices_df['year'], bins=30, color='salmon', edgecolor='black')
ax[2].set_title('Histogram of Year')
ax[2].set_xlabel('Year')
ax[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

scatter_matrix(car_prices_df, alpha=0.2, figsize=(20, 20))
plt.legend()
plt.show()

# ---feature selection/dimensionality reduction-----
print("\n----------------- START OF FEATURE SELECTION & DIMENSIONALITY REDUCTION -------------------\n")

label_encoders = {}
data = car_prices_df.copy()
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le

X = data.drop('sellingprice', axis=1)
y = data['sellingprice']
rf = RandomForestRegressor(n_estimators=6, max_depth=10, random_state=5805)
print("X.columns", X.columns)

rf.fit(X, y)
importances = rf.feature_importances_
feature_names = X.columns
feature_importance_dict = dict(zip(feature_names, importances))
sorted_feature_importance = sorted([[key, feature_importance_dict[key]] for key in feature_importance_dict],
                                   key=lambda x: x[1])
print("feature_importance_dict: ", sorted_feature_importance)

features = list(feature_importance_dict.keys())
importances = list(feature_importance_dict.values())

plt.figure(figsize=(10, 8))
plt.barh(features, importances, color='skyblue')
plt.xlabel('Importance Score')
plt.title('Feature Importance from Random Forest')
plt.show()

X = data.drop(['sellingprice', 'mmr'], axis=1)
y = data['sellingprice']
rf = RandomForestRegressor(n_estimators=6, max_depth=10, random_state=5805)
print("X.columns", X.columns)

rf.fit(X, y)
importances = rf.feature_importances_
feature_names = X.columns
feature_importance_dict = dict(zip(feature_names, importances))
sorted_feature_importance = sorted([[key, feature_importance_dict[key]] for key in feature_importance_dict],
                                   key=lambda x: x[1])
print("feature_importance_dict: ", sorted_feature_importance)

features = list(feature_importance_dict.keys())
importances = list(feature_importance_dict.values())

plt.figure(figsize=(10, 8))
plt.barh(features, importances, color='skyblue')
plt.xlabel('Importance Score')
plt.title('Feature Importance from Random Forest')
plt.show()

"""'year', 'make', 'model', 'trim', 'body', 'transmission', 'vin', 'state',
       'condition', 'odometer', 'color', 'interior', 'seller', 'mmr',
       'saleyear'"""


# ----------------PCA------------------#

print("\n----------------- START OF PCA -------------------\n")

pca = PCA()
X_pca = pca.fit_transform(X)

print(X.columns)
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
n_features_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1

plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='-')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio vs Number of Features')
plt.grid(True)
plt.axvline(x=n_features_95, color='r', linestyle='--')
plt.axhline(y=0.95, color='g', linestyle='--')

plt.annotate(f'({n_features_95}, 0.95)', xy=(n_features_95, 0.95), xytext=(n_features_95 + 5, 0.90),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()

print("Number of features needed for explaining more than 95% of the dependent variance:", n_features_95)

# ---------------SINGLE VALUE DECOMPOSITION------------#
print("\n----------------- START OF SINGLE VALUE DECOMPOSITION -------------------\n")

numerical_cols = X.select_dtypes(include=[np.number])

U, s, Vt = svd(numerical_cols, full_matrices=False)

print(s, U.shape, Vt.shape)

print("\n----------------- END OF SINGLE VALUE DECOMPOSITION -------------------\n")

# -----------------VIF--------------------------------#

print("\n----------------- START OF VIF -------------------\n")

X_vif = X.select_dtypes(include='number').assign(const=1)

vif_data = pd.DataFrame({
    'Variable': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})

vif_sorted = vif_data.sort_values(by='VIF', ascending=False)
print(vif_sorted.head())
plt.figure(figsize=(10, 8))

plt.barh(vif_data['Variable'], vif_data['VIF'], color='skyblue')
plt.xlabel('VIF Score')
plt.title('VIF method')
plt.show()
print("\n----------------- END OF VIF -------------------\n")

print("\n -------------------START RANDOM FOREST AND PCA WITH STANDARDIZATION--------------- \n")

scalar = StandardScaler()
data_scaled = scalar.fit_transform(data)
data_scaled_df = pd.DataFrame(data=data_scaled, columns=data.columns)
data.update(data_scaled_df)

X = data.drop(['sellingprice', 'mmr'], axis=1)
y = data['sellingprice']

pca = PCA()
X_pca = pca.fit_transform(X)

print(X.columns)
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
n_features_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1

plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='-')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio vs Number of Features')
plt.grid(True)
plt.axvline(x=n_features_95, color='r', linestyle='--')
plt.axhline(y=0.95, color='g', linestyle='--')

plt.annotate(f'({n_features_95}, 0.95)', xy=(n_features_95, 0.95), xytext=(n_features_95 + 5, 0.90),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()

print("Number of features needed for explaining more than 95% of the dependent variance:", n_features_95)

rf = RandomForestRegressor(n_estimators=8, max_depth=10, random_state=5805)
print("X.columns", X.columns)

rf.fit(X, y)
importances = rf.feature_importances_
feature_names = X.columns
feature_importance_dict = dict(zip(feature_names, importances))
sorted_feature_importance = sorted([[key, feature_importance_dict[key]] for key in feature_importance_dict],
                                   key=lambda x: x[1], reverse=True)
print("feature_importance_dict: ", sorted_feature_importance)

features = list(feature_importance_dict.keys())
importances = list(feature_importance_dict.values())

plt.figure(figsize=(10, 8))
plt.barh(features, importances, color='skyblue')
plt.xlabel('Importance Score')
plt.title('Feature Importance from Random Forest')
plt.show()

selected_features = [key[0] for key in sorted_feature_importance[:8]]
selected_features.append('sellingprice')

print("\n -------------------END OF RANDOM FOREST AND PCA WITH STANDARDIZATION--------------- \n")


print("\n----------------- END OF FEATURE SELECTION & DIMENSIONALITY REDUCTION -------------------\n")


# -----------------------ANOMALY REMOVAL/OUTLIER DETECTION-----------------#
print("\n----------------- START OF ANOMALY REMOVAL/OUTLIER DETECTION (Z-SCORE) -------------------\n")

cleaned_carprices_df = data[selected_features]
z_scores = np.abs(
    (cleaned_carprices_df['sellingprice'] - cleaned_carprices_df['sellingprice'].mean()) / cleaned_carprices_df[
        'sellingprice'].std())
data_cleaned_z = cleaned_carprices_df[z_scores < 3]
print("df shapes after anomaly deletion: ", cleaned_carprices_df.shape, data_cleaned_z.shape)

print("\n----------------- END OF ANOMALY REMOVAL/OUTLIER DETECTION (Z-SCORE)-------------------\n")

# -------------Covariance Matrix-----------------------------------------#

print("\n----------------- START OF COVARIANCE MATRIX -------------------\n")

covariance_matrix = data_cleaned_z.cov()
plt.figure(figsize=(10, 10))
sns.heatmap(covariance_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Heatmap of the Sample Covariance Matrix')
plt.show()
print("\n----------------- END OF COVARIANCE MATRIX -------------------\n")

# ------------------------Correlation Matrix---------------------------#
print("\n----------------- START OF CORRELATION MATRIX -------------------\n")

correlation_matrix = data_cleaned_z.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Heatmap of the Sample Pearson Correlation Coefficients Matrix')
plt.show()

print("\n----------------- END OF CORRELATION MATRIX -------------------\n")

# ------------------------------ PHASE II -----------------------------#

print(BLUE + "Phase II: Regression Analysis" + BLUE)
print("\n----------------- START OF MULTIPLE LINEAR REGRESSION -------------------\n")

X = data_cleaned_z.drop('sellingprice', axis=1)
y = data_cleaned_z['sellingprice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

model = sm.OLS(y_train, X_train).fit()

model_summary = model.summary()
print(model_summary)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse}")

predicted_values = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test.index, y_test, label='Actual')
plt.scatter(y_test.index, predicted_values, label='Predicted')
plt.title(' Multiple linear Regression Actual vs Predicted')
plt.xlabel('Index')
plt.ylabel('sellingprice')
plt.legend()
plt.show()
print("\n----------------- END OF MULTIPLE LINEAR REGRESSION -------------------\n")

# -------------------------------------- PCA ------------------#
print("\n----------------- START OF MULTIPLE LINEAR REGRESSION USING PCA-------------------\n")

pca = PCA()
X_pca = pca.fit_transform(X_train)

cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
n_features_90 = np.argmax(cumulative_variance_ratio >= 0.9) + 1

plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='-')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio vs Number of Features')
plt.grid(True)

plt.axvline(x=n_features_90, color='r', linestyle='--')
plt.axhline(y=0.9, color='g', linestyle='--')
plt.annotate(f'({n_features_90}, 0.9)', xy=(n_features_90, 0.9), xytext=(n_features_90 + 5, 0.85),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()

print("\n----------------- END OF MULTIPLE LINEAR REGRESSION USING PCA-------------------\n")

print("\n----------------- START OF MULTIPLE LINEAR REGRESSION WITH RANDOM FOREST REGRESSOR  -------------------\n")

X_train_selected = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_selected).fit()
print(ols_model.summary())
X_test_selected = sm.add_constant(X_test)
predicted_sales = ols_model.predict(X_test_selected)

plt.figure(figsize=(10, 6))
plt.plot(y_test.sort_index(), label='Original Sales')
plt.plot(predicted_sales.sort_index(), label='Predicted Sales')
plt.xlabel('Index')
plt.ylabel('Sales')
plt.title('Original vs Predicted Sales [RANDOM FOREST]')
plt.legend()
plt.show()



mse = mean_squared_error(y_test, predicted_sales)
print("Mean Squared Error (MSE) (RANDOM FOREST):", mse)

confidence_intervals = model.conf_int()
print("Confidence Intervals: ", confidence_intervals)


print("\n----------------- END OF MULTIPLE LINEAR REGRESSION WITH RANDOM FOREST REGRESSOR  -------------------\n")

print("\n-----------------START OF BACKWARD STEPWISE REGRESSION-------------------\n")


def backward_stepwise_regression(X_train, X_test, y_train, y_test, threshold=0.05):
    selected_features = list(X_train.columns)
    all_results = []

    iteration = 1

    while len(selected_features) > 0:
        X_train_selected = X_train[selected_features]
        X_train_selected = sm.add_constant(X_train_selected)
        X_test_selected = X_test[selected_features]
        X_test_selected = sm.add_constant(X_test_selected)

        model = sm.OLS(y_train, X_train_selected).fit()
        p_values = model.pvalues[1:]

        eliminated_feature = None
        aic = model.aic
        bic = model.bic
        adj_r2 = model.rsquared_adj
        p_value = None

        if p_values.max() > threshold:
            eliminated_feature = p_values.idxmax()
            selected_features.remove(eliminated_feature)
            p_value = p_values.max()
        else:
            break

        results = {'Eliminated Feature': eliminated_feature,
                   'AIC': aic, 'BIC': bic, 'Adjusted R-squared': adj_r2,
                   'P-value': p_value,
                   'Selected Features': selected_features.copy()}
        all_results.append(results)

        print("Model summary at iteration {} ".format(iteration))
        print(model.summary())
        iteration += 1

    final_selected_features = selected_features
    return all_results, final_selected_features


X = data.drop(['sellingprice', 'mmr'], axis=1)
y = data['sellingprice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)

all_results, final_selected_features = backward_stepwise_regression(X_train, X_test, y_train, y_test)

results_dfs = []
for i, results in enumerate(all_results):
    results_df = pd.DataFrame([results])
    results_df['Iteration'] = i + 1
    results_dfs.append(results_df)

results_df = pd.concat(results_dfs, ignore_index=True)

print("Process and Justification for Backward Stepwise Regression:")
print(results_df)
print("\nFinal Selected Features:", final_selected_features)

final_model = sm.OLS(y_train[final_selected_features], X_train[final_selected_features]).fit()
predicted_sales = final_model.predict(X_test[final_selected_features])

plt.figure(figsize=(10, 6))
plt.plot(y_test[final_selected_features].sort_index(), label='Original Selling Price')
plt.plot(predicted_sales.sort_index(), label='Predicted Selling Price')
plt.xlabel('Index')
plt.ylabel('Selling Price')
plt.title('Original vs Predicted Selling Price (BACKWARD STEPWISE REGRESSION)')
plt.legend()
plt.show()

mse = mean_squared_error(y_test, predicted_sales)

print("Mean Squared Error (MSE) BACKWARD STEPWISE REGRESSION :", mse)

print("\n----------------- END OF MULTIPLE LINEAR REGRESSION (BACKWARD STEPWISE REGRESSION) -------------------\n")
