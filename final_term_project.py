import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.cluster import DBSCAN
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.ensemble import BaggingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from prettytable import PrettyTable

RED = "\033[91m"
GREEN = "\033[92m"
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

print("\n----------------- START OF FEATURE SELECTION & DIMENSIONALITY REDUCTION -------------------\n")
label_encoders = {}
data = car_prices_df.copy()
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le

encoded_data = data.copy()

print(data.head())

print("\n------------------------STANDARDIZE THE DATA---------------------\n")

scalar = StandardScaler()
data_scaled = scalar.fit_transform(data)
data_scaled_df = pd.DataFrame(data=data_scaled, columns=data.columns)
data.update(data_scaled_df)
print(data.head())

print("\n-----------Dimensionality reduction/feature selection---------\n")
X = data.drop(['sellingprice'], axis=1)
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

print("\n----------------- START OF SINGLE VALUE DECOMPOSITION -------------------\n")

numerical_cols = X.select_dtypes(include=[np.number])

U, s, Vt = svd(numerical_cols, full_matrices=False)

print(s, U.shape, Vt.shape)

print("\n----------------- END OF SINGLE VALUE DECOMPOSITION -------------------\n")

print("\n----------------- START OF VIF -------------------\n")

X_vif = X.select_dtypes(include='number').assign(const=1)

vif_data = pd.DataFrame({
    'Variable': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})

vif_sorted = vif_data.sort_values(by='VIF', ascending=False)
print(vif_sorted)
plt.figure(figsize=(10, 8))

plt.barh(vif_data['Variable'], vif_data['VIF'], color='skyblue')
plt.xlabel('VIF Score')
plt.title('VIF method')
plt.show()
print("\n----------------- END OF VIF -------------------\n")

final_selected_features = ['year', 'make', 'model', 'trim', 'condition', 'odometer', 'sellingprice', 'mmr']

print("\n----------------- START OF ANOMALY REMOVAL/OUTLIER DETECTION (Z-SCORE) -------------------\n")

cleaned_carprices_df = data[final_selected_features]
z_scores = np.abs(
    (cleaned_carprices_df['sellingprice'] - cleaned_carprices_df['sellingprice'].mean()) / cleaned_carprices_df[
        'sellingprice'].std())
data_cleaned_z = cleaned_carprices_df[z_scores < 3]
print("df shapes after anomaly deletion: ", cleaned_carprices_df.shape, data_cleaned_z.shape)
z_scores = np.abs((cleaned_carprices_df['sellingprice'] - cleaned_carprices_df['sellingprice'].mean()) / cleaned_carprices_df['sellingprice'].std())

outliers = z_scores >= 3

plt.figure(figsize=(10, 6))
plt.scatter(cleaned_carprices_df.index, cleaned_carprices_df['sellingprice'], c='blue', label='Non-outliers')
plt.scatter(cleaned_carprices_df[outliers].index, cleaned_carprices_df[outliers]['sellingprice'], c='red', label='Outliers')
plt.title('Outliers in Selling Price Using Z-score')
plt.xlabel('Index')
plt.ylabel('Selling Price')
plt.legend()
plt.show()

print("\n----------------- END OF ANOMALY REMOVAL/OUTLIER DETECTION (Z-SCORE)-------------------\n")

print("\n----------------- START OF COVARIANCE MATRIX -------------------\n")

covariance_matrix = encoded_data[final_selected_features].cov()
plt.figure(figsize=(10, 10))
sns.heatmap(covariance_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Heatmap of the Sample Covariance Matrix')
plt.show()
print("\n----------------- END OF COVARIANCE MATRIX -------------------\n")

print("\n----------------- START OF CORRELATION MATRIX -------------------\n")
correlation_matrix = data_cleaned_z.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Heatmap of the Sample Pearson Correlation Coefficients Matrix')
plt.show()

print("\n----------------- END OF CORRELATION MATRIX -------------------\n")

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

plt.figure(figsize=(10, 6))
plt.plot(X_train.index, y_train, color='blue', label='Train Actual', alpha=0.5, marker='o', linestyle='-')
plt.plot(X_test.index, y_test, color='green', label='Test Actual', alpha=0.5, marker='o', linestyle='-')
plt.plot(X_test.index, predicted_values, color='red', label='Test Predicted', alpha=0.5, marker='o', linestyle='-')
plt.xlabel('Index')
plt.ylabel('selling price')
plt.title('Original selling price vs Predicted selling price')
plt.legend()
plt.show()

print("\n----------------- END OF MULTIPLE LINEAR REGRESSION -------------------\n")


def backward_stepwise_regression(X_train, X_test, y_train, y_test, threshold=0.00001):
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)

final_model = sm.OLS(y_train, X_train).fit()
predicted_values = final_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test.index, y_test, label='Actual')
plt.scatter(y_test.index, predicted_values, label='Predicted')
plt.title(' Multiple linear Regression Actual vs Predicted')
plt.xlabel('Index')
plt.ylabel('sellingprice')
plt.legend()
plt.show()

mse = mean_squared_error(y_test, predicted_values)

print("Mean Squared Error (MSE) BACKWARD STEPWISE REGRESSION :", mse)

print("\n----------------- END OF MULTIPLE LINEAR REGRESSION (BACKWARD STEPWISE REGRESSION) -------------------\n")

print("\n----------------Phase III: Classification Analysis:-----------------\n")

print(car_prices_df.info())

downsampled_df = data.sample(frac=0.2, replace=False, random_state=1)

X = downsampled_df.drop(['condition'], axis=1)
y = downsampled_df['condition']

threshold_value = y.median()
y = pd.cut(y, bins=[-float('inf'), threshold_value, float('inf')], labels=[0, 1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=5805
)

print(y.value_counts())


def evaluate_model(preds, y_test, model_name, dt_best):
    print(f'{model_name} Results:')
    table = PrettyTable()

    # Adding headers for all metrics
    table.field_names = ["Confusion Matrix", "Precision", "Recall", "F1-Score", "ROC AUC"]

    # Compute metrics
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    # Compute ROC AUC
    fpr, tpr, _ = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    # Format confusion matrix
    cm = confusion_matrix(y_test, preds)
    cm_str = f"[[{cm[0, 0]}, {cm[0, 1]}],\n [{cm[1, 0]}, {cm[1, 1]}]]"

    # Add rows for metrics
    table.add_row([cm_str, f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}", f"{roc_auc:.3f}"])
    print(table)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve {}".format(model_name))
    plt.legend(loc="lower right")
    plt.show()

    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        model = dt_best.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_test_fold)
        scores.append(f1_score(y_test_fold, preds))

    print(f'Stratified K-fold Cross-Validation F1-Score: {np.mean(scores):.3f}')
    print('--------------------------------')


dt_model = DecisionTreeClassifier(random_state=5805)
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)
evaluate_model(y_pred, y_test, "Decision Tree Classifier", dt_model)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
importance = pd.Series(dt_model.feature_importances_, index=X.columns)
print("Feature Importance:", importance)
print("feature to be removed :", importance.idxmin())

plt.figure(figsize=(8, 6))
sns.barplot(x=importance, y=importance.index)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

tuned_parameters = [{"max_depth": range(2, 10, 1),
                     "min_samples_leaf": range(2, 10, 2),
                     "min_samples_split": range(2, 10, 2),
                     'max_features': [None, 'sqrt', 'log2'],
                     'splitter': ['best', 'random'],
                     'criterion': ['gini', 'entropy']}]

dt_model_pruned = DecisionTreeClassifier(random_state=5805, ccp_alpha=0.09)

grid_search = GridSearchCV(estimator=dt_model_pruned, param_grid=tuned_parameters, cv=5, n_jobs=-1)

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
dt_model_pre_pruned = grid_search.best_estimator_
dt_model_pre_pruned.fit(X_train, y_train)
y_pre_predicted = dt_model_pre_pruned.predict(X_test)
print(f'Accuracy of pre tuned tree {accuracy_score(y_test, y_pre_predicted):.3f}')

plt.figure(figsize=(20, 12))
plot_tree(dt_model_pre_pruned, rounded=True, filled=True)
plt.show()

path = dt_model.cost_complexity_pruning_path(X_train, y_train)
print(path)

ccp_alphas, impurities = path.ccp_alphas, path.impurities
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=5806, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

acc_scores = [accuracy_score(y_test, clf.predict(X_test)) for clf in clfs]
max_acc_score_index = acc_scores.index(max(acc_scores))
plt.figure(figsize=(10, 6))
plt.grid()
plt.plot(ccp_alphas[:-1], acc_scores[:-1])
plt.xlabel("effective alpha")
plt.ylabel("Accuracy scores")
plt.show()

post_pruning_clf = clfs[max_acc_score_index]
post_pruning_clf.fit(X_train, y_train)
y_post_predicted = post_pruning_clf.predict(X_test)
post_pruning_acc = accuracy_score(y_test, y_post_predicted)

print("accuracy for post pruning : ", round(post_pruning_acc, 3))
plot_tree(decision_tree=post_pruning_clf, rounded=True, filled=True)

pre_confusion_matrix = confusion_matrix(y_test, y_pre_predicted)
pre_accuracy = accuracy_score(y_test, y_pre_predicted)
pre_recall = recall_score(y_test, y_pre_predicted)
pre_roc_auc = roc_auc_score(y_test, y_pre_predicted)
print(" pre metrics ", pre_confusion_matrix, pre_recall, pre_accuracy, pre_roc_auc)

post_confusion_matrix = confusion_matrix(y_test, y_post_predicted)
post_accuracy = accuracy_score(y_test, y_post_predicted)
post_recall = recall_score(y_test, y_post_predicted)
post_roc_auc = roc_auc_score(y_test, y_post_predicted)
metrics_data = {
    'Pruning Method': ['Pre-Pruned', 'Post-Pruned'],
    'Accuracy': [pre_accuracy, post_accuracy],
    'Confusion Matrix': [pre_confusion_matrix, post_confusion_matrix],
    'Recall': [pre_recall, post_recall],
    'AUC': [pre_roc_auc, post_roc_auc]
}

metrics_df = pd.DataFrame(metrics_data)

print(metrics_df.head())

logreg_model = LogisticRegression(random_state=5805)
logreg_model.fit(X_train, y_train)
y_logistic_pred = logreg_model.predict(X_test)
accuracy = accuracy_score(y_test, y_logistic_pred)
print("Logistic Regression Accuracy:", round(accuracy, 3))

logistic_auc = roc_auc_score(y_test, y_logistic_pred)
dt_pre_auc = roc_auc_score(y_test, y_pre_predicted)
dt_post_auc = roc_auc_score(y_test, y_post_predicted)

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_logistic_pred)
fpr_dt_pre, tpr_dt_pre, _ = roc_curve(y_test, y_pre_predicted)
fpr_dt_post, tpr_dt_post, _ = roc_curve(y_test, y_post_predicted)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {logistic_auc:.3f})')
plt.plot(fpr_dt_pre, tpr_dt_pre, label=f'Decision Tree with pre pruning (AUC = {dt_pre_auc:.3f})')
plt.plot(fpr_dt_post, tpr_dt_post, label=f'Decision Tree with post pruning (AUC = {dt_post_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

logistic_confusion_matrix = confusion_matrix(y_test, y_logistic_pred)
logistic_accuracy = accuracy_score(y_test, y_logistic_pred)
logistic_recall = recall_score(y_test, y_logistic_pred)
logistic_auc = roc_auc_score(y_test, y_logistic_pred)
metrics_data = {
    'Pruning Method': ['Pre-Pruned', 'Post-Pruned', 'Logistic'],
    'Accuracy': [pre_accuracy, post_accuracy, logistic_accuracy],
    'Confusion Matrix': [pre_confusion_matrix, post_confusion_matrix, logistic_confusion_matrix],
    'Recall': [pre_recall, post_recall, logistic_recall],
    'AUC': [pre_roc_auc, post_roc_auc, logistic_auc]
}
metrics_df = pd.DataFrame(metrics_data)

evaluate_model(y_logistic_pred, y_test, "Logistic Regression", logreg_model)
evaluate_model(y_pre_predicted, y_test, "Pre Pruned Decision Tree", dt_model_pre_pruned)
evaluate_model(y_post_predicted, y_test, "Post Pruned Decision Tree", post_pruning_clf)

print(metrics_df)

print("\n--------------------------KNN-----------------------\n")

error = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)

plt.title('Error Rate for Different K Values')
plt.xlabel('K')
plt.ylabel('Error')
plt.show()

knn = KNeighborsClassifier(n_neighbors=error.index(min(error)) + 1)

knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print('Minimum error:', min(error))
print('Optimum K:', error.index(min(error)) + 1)
evaluate_model(pred, y_test, 'KNN', knn)

print("\n------------------------------SVM----------------------------\n")

parameters = {
    'kernel': ['linear', 'rbf', 'poly'],
}
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5805)

# Perform grid search with cross-validation
svm_classifier = SVC(probability=True)
grid_search = GridSearchCV(svm_classifier, parameters, cv=stratified_kfold, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best Hyperparameters:", grid_search.best_params_)
svm = grid_search.best_estimator_
svm.fit(X_train, y_train)
y_svm_pred = svm.predict(X_test)
evaluate_model(y_svm_pred, y_test, 'SVM', svm)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_nb_pred = nb_model.predict(X_test)
evaluate_model(y_nb_pred, y_test, 'Naive Bayes', nb_model)

# Bagging
bagging_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=5805)
bagging_clf.fit(X_train, y_train)
y_bagging_pred = bagging_clf.predict(X_test)
evaluate_model(y_bagging_pred, y_test, 'Bagging', bagging_clf)

# Stacking
estimators = [('dt', DecisionTreeClassifier()), ('lr', LogisticRegression())]
stacking_clf = StackingClassifier(estimators=estimators)
stacking_clf.fit(X_train, y_train)
y_stacking_pred = stacking_clf.predict(X_test)
evaluate_model(y_stacking_pred, y_test, 'Stacking', stacking_clf)

# Boosting
boosting_clf = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=5805)
boosting_clf.fit(X_train, y_train)
y_boosting_pred = boosting_clf.predict(X_test)
evaluate_model(y_boosting_pred, y_test, 'Boosting', boosting_clf)

nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=5805)
nn_model.fit(X_train, y_train)
y_nn_pred = nn_model.predict(X_test)
evaluate_model(y_nn_pred, y_test, 'Neural Network', nn_model)

sum_of_squared_distances = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=5805)
    kmeans.fit(X)
    sum_of_squared_distances.append(kmeans.inertia_)

plt.plot(range(1, 11), sum_of_squared_distances, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method for Optimal k Selection')
plt.show()

silhouette_coefficients = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=5805)
    kmeans.fit(X)

    silhouette_coefficients.append(silhouette_score(X, kmeans.labels_))

plt.plot(range(2, 11), silhouette_coefficients, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Analysis for Optimal k Selection')
plt.show()

print("\n---------------DBSCAN-------------\n")

data = car_prices_df.copy()
data_clean = data.dropna(subset=['condition', 'odometer', 'mmr', 'sellingprice', 'year'])
features = data_clean[['year', 'condition', 'odometer', 'mmr', 'sellingprice']]
features_scaled = StandardScaler().fit_transform(features)
dbscan = DBSCAN(eps=0.5, min_samples=10)
clusters = dbscan.fit_predict(features_scaled)
plt.figure(figsize=(10, 6))
plt.scatter(features_scaled[:, 2], features_scaled[:, 4], c=clusters, cmap='viridis', s=5)
plt.xlabel('Odometer (scaled)')
plt.ylabel('Selling Price (scaled)')
plt.title('DBSCAN Clustering Results')
plt.colorbar(label='Cluster Label')
plt.show()

print("\n---------------------ASSOCIATION RULE MINING------------------\n")

df_apriori_filtered = car_prices_df.copy()
df_apriori_filtered = pd.get_dummies(df_apriori_filtered, drop_first=True)
frequent_itemsets = apriori(df_apriori_filtered, min_support=0.1, use_colnames=True, verbose=1)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.2)
rules = rules.sort_values(['confidence'], ascending=False)
print(rules.head(10))
