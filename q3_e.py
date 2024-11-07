import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, fbeta_score
from q1 import data_preprocessing
from q2 import data_splits, normalize_features

# Step 1: Create hyperparameter grids for each model
param_grid_decision_tree = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 3],
    'max_leaf_nodes': [25, 30, 35, 40],
}

param_grid_random_forest = {
    'n_estimators': [250, 300],
    'max_depth': [20, 30],
    'bootstrap': [True],
}

# param_grid_svm = {
#     'kernel': ['linear', 'rbf'], # 'poly', 'sigmoid'],
#     'shrinking': [True, False],
#     'C': [50, 100]
#     # 'tol': [1e-3, 1e-4],
#     # 'gamma': ['scale', 'auto']
# }

param_grid_svm = {
    'kernel': ['rbf', 'sigmoid'], # 'linear', 'poly'],
    # TODO: Add False for gradescope
    'shrinking': [True],
    'C': [1, 5]
    # 'tol': [1e-3, 1e-4],
    # 'gamma': ['scale', 'auto']
}

# Step 2: Initialize classifiers with random_state=0
decision_tree = DecisionTreeClassifier(random_state=0)
random_forest = RandomForestClassifier(random_state=0)
svm = SVC(random_state=0)

# Step 3: Create a scorer using F-beta score with beta=0.5
scorer = 'accuracy'

# Step 4: Perform grid search for each model using 10-fold StratifiedKFold cross-validation
def perform_grid_search(model, X_train, y_train, params):
    # Define the cross-validation strategy
    # TODO: See if random_state and shuffle are needed
    y_train = y_train.values.ravel()
    strat_kfold = StratifiedKFold(n_splits=10)

    # Grid search for the model
    grid_search = GridSearchCV(estimator=model, param_grid=params, scoring=scorer, cv=strat_kfold, n_jobs=10)
    grid_search.fit(X_train, y_train)

    best_param = grid_search.best_params_
    best_score = grid_search.score(X_train, y_train)
    print("Best parameters are:", best_param)
    print("Best score is:", best_score)

    # Return the fitted grid search objects
    return grid_search, best_param, best_score



# X, y = data_preprocessing()
# # TODO: Remove ravel for gradescope
# y = y.values.ravel()
# X_train, X_test, y_train, y_test = data_splits(X, y)
# X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)

# # Do Grid search for Decision Tree
# grid_decision_tree, best_params_decision_tree, best_score_decision_tree  = perform_grid_search(decision_tree, X_train_scaled, y_train, param_grid_decision_tree)

# # Do Grid search for Random Forest
# grid_random_forest, best_params_random_forest, best_score_random_forest  = perform_grid_search(random_forest, X_train_scaled, y_train, param_grid_random_forest)

# # Do Grid search for SVM
# grid_svm, best_params_svm, best_score_svm = perform_grid_search(svm, X_train_scaled, y_train, param_grid_svm)








