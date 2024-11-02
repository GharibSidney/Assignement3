from sklearn.model_selection import train_test_split
from q1 import data_preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


def data_splits(X, y):
    """
    Split the 'features' and 'labels' data into training and testing sets.
    Input(s): X: features (pd.DataFrame), y: labels (pd.DataFrame)
    Output(s): X_train, X_test, y_train, y_test
    """
    # Use random_state = 0 in the train_test_split
    # TODO write data split here
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, shuffle=True, random_state=0)

    return X_train, X_test, y_train, y_test


def normalize_features(X_train, X_test):
    """
    Take the input data and normalize the features.
    Input: X_train: features for train,  X_test: features for test (pd.DataFrame)
    Output: X_train_scaled, X_test_scaled (pd.DataFrame) the same shape of X_train and X_test
    """
    # TODO write normalization here
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def train_model(model_name, X_train_scaled, y_train):
    '''
    inputs:
       - model_name: the name of learning algorithm to be trained
       - X_train: features training set
       - y_train: label training set
    output: cls: the trained model
    '''
    if model_name == 'Decision Tree':
        # TODO call classifier here
        cls = DecisionTreeClassifier() 
        
    elif model_name == 'Random Forest':
        # TODO call classifier here
        cls = RandomForestClassifier() 
    elif model_name == 'SVM':
        # TODO call classifier here
        cls = SVC()

    # TODO train the model
    cls.fit(X_train_scaled, y_train)

    return cls


def eval_model(trained_models, X_train, X_test, y_train, y_test):
    '''
    inputs:
       - trained_models: a dictionary of the trained models,
       - X_train: features training set
       - X_test: features test set
       - y_train: label training set
       - y_test: label test set
    outputs:
        - y_train_pred_dict: a dictionary of label predicted for train set of each model
        - y_test_pred_dict: a dictionary of label predicted for test set of each model
        - a dict of accuracy and f1_score of train and test sets for each model
    '''
    evaluation_results = {}
    y_train_pred_dict = {
        'Decision Tree': None,
        'Random Forest': None,
        'SVM': None}
    y_test_pred_dict = {
        'Decision Tree': None,
        'Random Forest': None,
        'SVM': None}

    # Loop through each trained model
    for model_name, model in trained_models.items():
        # Predictions for training and testing sets
        y_train_pred = model.predict(X_train) # TODO predict y
        y_test_pred = model.predict(X_test) # TODO predict y

        # Calculate accuracy
        # train_accuracy = 0
        # train_true_positive = 0
        # train_true_negative = 0 
        # train_false_positive = 0
        # train_false_negative = 0

        # for i in range (len(y_train_pred)):
        #     if y_train_pred[i] == 1 and y_train[i] == 1:
        #         train_true_positive += 1
        #     elif y_train_pred[i] == 0 and y_train[i] == 0:
        #         train_true_negative += 1
        #     elif y_train_pred[i] == 1 and y_train[i] == 0:
        #         train_false_positive += 1
        #     elif y_train_pred[i] == 0 and y_train[i] == 1:
        #         train_false_negative += 1
        # precision = train_true_positive / (train_true_positive + train_false_positive)
        # recall = train_true_positive / (train_true_positive + train_false_negative)
        # train_accuracy =  train_accuracy / len(y_train_pred) 
        train_accuracy = accuracy_score(y_train, y_train_pred) # TODO find accuracy

        # test_accuracy = 0
        # test_true_positive = 0
        # test_true_negative = 0 
        # test_false_positive = 0
        # test_false_negative = 0
        # for i in range (len(y_train_pred)):
        #     if y_train_pred[i] == y_train[i]:
        #         test_accuracy += 1
        # test_accuracy = test_accuracy / len(test_accuracy) 
        test_accuracy = accuracy_score(y_test , y_test_pred) #TODO find accuracy

        # Calculate F1-score
        train_f1 = f1_score(y_train, y_train_pred) # TODO find f1_score
        test_f1 = f1_score(y_test, y_test_pred) # TODO find f1_score

        # Store predictions
        y_train_pred_dict[model_name] = y_train_pred # TODO
        y_test_pred_dict[model_name] = y_test_pred # TODO

        # Store the evaluation metrics
        evaluation_results[model_name] = {
            'Train Accuracy': train_accuracy, # TODO ,
            'Test Accuracy': test_accuracy,  # TODO ,
            'Train F1 Score': train_f1,  # TODO ,
            'Test F1 Score': test_f1, # TODO
        }

    # Return the evaluation results
    return y_train_pred_dict, y_test_pred_dict, evaluation_results


def report_model(y_train, y_test, y_train_pred_dict, y_test_pred_dict):
    '''
    inputs:
        - y_train: label training set
        - y_test: label test set
        - y_train_pred_dict: a dictionary of label predicted for train set of each model, len(y_train_pred_dict.keys)=3
        - y_test_pred_dict: a dictionary of label predicted for test set of each model, len(y_train_pred_dict.keys)=3
    '''

    # Loop through each trained model
    for model_name, model in trained_models.items():
        print(f"\nModel: {model_name}")

        # Predictions for training and testing sets
        y_train_pred = y_train_pred_dict[model_name] # TODO compelete it
        y_test_pred = y_test_pred_dict[model_name] # TODO compelete it

        # Print classification report for training set
        print("\nTraining Set Classification Report:")
        # TODO write Classification Report train
        print(classification_report(y_train, y_train_pred))

        # Print confusion matrix for training set
        print("Training Set Confusion Matrix:")
        # TODO write Confusion Matrix train
        print(confusion_matrix(y_train, y_train_pred))

        # Print classification report for testing set
        print("\nTesting Set Classification Report:")
        # TODO write Classification Report test
        print(classification_report(y_test, y_test_pred))

        # Print confusion matrix for testing set
        print("Testing Set Confusion Matrix:")
        # TODO write Confusion Matrix test
        print(confusion_matrix(y_test, y_test_pred))



X, y = data_preprocessing() # TODO call data preprocessing from q1
X_train, X_test, y_train, y_test =  data_splits(X, y) # TODO split data
X_train_scaled, X_test_scaled = normalize_features(X_train, X_test) # TODO normalize data

cls_decision_tree =  train_model("Decision Tree", X_train_scaled, y_train )# TODO train the model
cls_randomforest = train_model("Random Forest", X_train_scaled, y_train ) # TODO train the model
cls_svm = train_model("SVM", X_train_scaled, y_train ) # TODO train the model

# Define a dictionary of model name and their trained model
trained_models = {
        'Decision Tree': cls_decision_tree,
        'Random Forest': cls_randomforest,
        'SVM': cls_svm }

# predict labels and calculate accuracy and F1score
y_train_pred_dict, y_test_pred_dict, evaluation_results = eval_model(trained_models, X_train_scaled, X_test_scaled, y_train, y_test)

# classification report and calculate confusion matrix
report_model(y_train, y_test, y_train_pred_dict, y_test_pred_dict)



















