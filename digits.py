from sklearn import grid_search, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

# Use spark_sklearn’s grid search instead:
from spark_sklearn import GridSearchCV

digits = datasets.load_digits()
X, y = digits.data, digits.target
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "n_estimators": [30, 70, 120]}

gs = grid_search.GridSearchCV(RandomForestClassifier(), param_grid=param_grid, verbose=1)
gs.fit(X, y)

#print(gs.cv_results_)
print(gs.best_params_)

# Save out the best model
joblib.dump(gs.best_estimator_, 'digits.pkl')
