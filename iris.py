from sklearn import svm, grid_search, datasets
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
# Use spark_sklearn’s grid search instead:
from spark_sklearn import GridSearchCV

iris = datasets.load_iris()

param_grid = {"max_depth": [3, None],
              "max_features": [1, 2, 4],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "n_estimators": [30, 70, 120]}

gs = grid_search.GridSearchCV(RandomForestClassifier(), param_grid=param_grid, verbose=1)
gs.fit(iris.data, iris.target)

print(gs.best_params_)

# Save out the best model
joblib.dump(gs.best_estimator_, 'iris.pkl')
