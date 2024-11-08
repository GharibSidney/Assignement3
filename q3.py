import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score
from q1 import data_preprocessing
from q2 import data_splits, normalize_features

# Step 1: Create hyperparameter grids for each model
# TODO fill out below dictionaries with reasonable values
param_grid_decision_tree = {
    'criterion': ['gini','entropy'],
    'max_depth': [10, 20, 50],
    'min_samples_leaf': [2, 3],
    'max_leaf_nodes': [5, 10]
}

param_grid_random_forest = {
    'n_estimators': [10, 50, 300],
    'max_depth': [20, 30],
    'bootstrap': [True, False],
}

param_grid_svm = {
    'kernel':['rbf', 'sigmoid'],
    'shrinking': [True, False],
    'C': [1, 10],
}

# Step 2: Initialize classifiers with random_state=0
decision_tree = DecisionTreeClassifier(random_state=0) # TODO
random_forest = RandomForestClassifier(random_state=0) # TODO
svm = SVC(random_state=0) # TODO

# Step 3: Create a scorer using accuracy_score
scorer =  "accuracy"  #make_scorer(accuracy_score) # TODO


# Step 4: Perform grid search for each model using 9-fold StratifiedKFold cross-validation
def perform_grid_search(model, X_train, y_train, params):
    print("Performing grid search for ", model)
    # Define the cross-validation strategy
    strat_kfold = StratifiedKFold(n_splits=10) # TODO

    # Grid search for the model 
    grid_search = GridSearchCV(model, params, scoring=scorer, cv=strat_kfold) # TODO scorer n_jobs=10

    # TODO fit to the data

    # if isinstance(model, RandomForestClassifier) or isinstance(model, SVC):
    #     grid_search.fit(X_train, y_train.values.ravel())
    # else:
    grid_search.fit(X_train, y_train)
    best_param = grid_search.best_params_ # TODO
    best_score = grid_search.best_score_ # TODO
    print("Best parameters are:", best_param)
    print("Best score is:", best_score)

    # Return the fitted grid search objects
    return grid_search, best_param, best_score



X, y = data_preprocessing()
X_train, X_test, y_train, y_test = data_splits(X, y)
X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)

# Do Grid search for Decision Tree
grid_decision_tree, best_params_decision_tree, best_score_decision_tree  = perform_grid_search(decision_tree, X_train_scaled, y_train, param_grid_decision_tree  ) # TODO

# Do Grid search for Random Forest
grid_random_forest, best_params_random_forest, best_score_random_forest  = perform_grid_search(random_forest, X_train_scaled, y_train, param_grid_random_forest) # TODO

# Do Grid search for SVM
grid_svm, best_params_svm, best_score_svm = perform_grid_search(svm, X_train_scaled, y_train, param_grid_svm) # TODO









