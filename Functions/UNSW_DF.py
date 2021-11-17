# --------------------------------------------------------------------------- #
# ---------------------------- LIBRARY IMPORTS ------------------------------ #
# --------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# ----------------------------- DATASET IMPORT ------------------------------ #
# --------------------------------------------------------------------------- #
# 1. When importing libraries, call for the file with: from UNSW_DF import *
# 2. Initiate with: x_train, x_test, y_train, y_test = XY_import()
def DF_XY():
    try:
        print("( 1 ) Reading Preprocessed CSV files..")
        train = pd.read_csv("../Dataset/train_pp3.csv")
        print("\t Training dataset loaded..")
        test = pd.read_csv("../Dataset/test_pp3.csv")
        print("\t Testing dataset loaded..\n")

        print("( 2 ) Loading done, splitting into X and Y..")
        x_train, y_train = train.drop(["label"], axis=1), train["label"]
        x_test, y_test = test.drop(["label"], axis=1), test["label"]
        print('\t ( 2.1 ) x_train Shape: ', '\t', x_train.shape)
        print('\t ( 2.2 ) y_train Shape: ', '\t', y_train.shape)
        print('\t ( 2.3 ) x_test Shape: ', '\t', x_test.shape)
        print('\t ( 2.4 ) y_test Shape: ', '\t', y_test.shape)

        print("( 3 ) Done!")
        print("PS! Import with: x_train, x_test, y_train, y_test = XY_import()")
    except:
        print("Could not load dataset, try again..")
    return x_train, x_test, y_train, y_test

# For importiong the preprocessed dataset by train and test
def DF_preprocessed_traintest():
    print("Reading Preprocessed CSV Files..")
    train = pd.read_csv("../Dataset/train_pp3.csv")
    test = pd.read_csv("../Dataset/test_pp3.csv")
    print('\t Train Shape: ', '\t', train.shape)
    print('\t Test Shape: ', '\t', test.shape)
    print("Dataset Loaded!")
    return train, test

# For importiong the orignal dataset by train and test
def DF_original_traintest():
    print("Reading Original CSV Files..")
    # importing original dataset
    UNSW_train = pd.read_csv("../Dataset/UNSW_NB15_training-set.csv", delimiter=",")
    UNSW_test = pd.read_csv("../Dataset/UNSW_NB15_testing-set.csv", delimiter=",")
    if UNSW_train.shape < UNSW_test.shape:
        UNSW_train, UNSW_test = UNSW_test, UNSW_train
    print('\t Train Shape: ', '\t', UNSW_train.shape)
    print('\t Test Shape: ', '\t', UNSW_test.shape)
    print("Dataset Loaded!")
    return UNSW_train, UNSW_test
    

# --------------------------------------------------------------------------- #
# ------------------------ ARTIFICIAL NEURAL NETWORK ------------------------ #
# --------------------------------------------------------------------------- #
# For prediciton of ANN models
# The function takes in the dataset, model build and model name as input.
def UNSW_predict_ANN_model(x_train, y_train, x_test, y_test, model, model_name):
    # Printing the name of the model
    print("( 1 ) Running: ", model_name)
    
    # Predicting the training dataset
    print("( 2 ) Predicting on train..")
    train = model.predict(x_train)
    train_scores = model.evaluate(x_train, y_train, verbose=0)
    print('\tAccuracy on training data: \t{}'.format(train_scores[1]))
    print('\tError on training data: \t{}'.format(1 - train_scores[1]), "\n")

    # Predicting the testing dataset
    print("( 3 ) Predicting on test..")
    test = model.predict(x_test)
    test_scores = model.evaluate(x_test, y_test, verbose=0)
    print('\tAccuracy on testing data: \t{}'.format(test_scores[1]))
    print('\tError on testing data: \t\t{}'.format(1 - test_scores[1]), "\n")
    print("( 4 ) Done!")
    
    
# --------------------------------------------------------------------------- #
# ------------------------------ KNN(Fixed K) ------------------------------- #
# --------------------------------------------------------------------------- #    
def UNSW_predict_KNN_model(K, x_train, y_train):
    print("Predicting KNN model with K = %s" %K)
    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=K)
    
    # Train the model using the training sets
    knn.fit(x_train, y_train)
    print("Done: Train the model using the training set..")
    
    # Predict the response for test dataset
    y_pred = knn.predict(x_test)
    print("Done: Predict the response for test dataset..\n")
    
    print('Accuracy of the model: ', metrics.accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# --------------------------------------------------------------------------- #
# ------------------------------ PLOTTING  ---------------------------------- #
# --------------------------------------------------------------------------- #
def UNSW_plot_corr_matrix(dataset, fig_x, fig_y):
    f = plt.figure(figsize=(fig_x, fig_y))
    plt.matshow(dataset.corr(), fignum=f.number)
    
    plt.xticks(range(dataset.select_dtypes(['number']).shape[1]), 
               dataset.select_dtypes(['number']).columns, 
               fontsize=14, 
               rotation=45)
    
    plt.yticks(range(dataset.select_dtypes(['number']).shape[1]), 
               dataset.select_dtypes(['number']).columns, 
               fontsize=14)
    
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    
    
# --------------------------------------------------------------------------- #
# --------------------------- DATA ANALYSIS  -------------------------------- #
# --------------------------------------------------------------------------- #    
def UNSW_data_analysis_preprocess(train, test):
    # Defining an empty list
    categorical = []
    # Iterating through the columns and checking for columns with datatyp "Object"
    for col in train.columns:
        if train[col].dtype == 'object':
            categorical.append(col) # appending "object" columns to categorical
            
    non_categorical_columns = [x for x in train.columns if x not in categorical]

    # Label encoding the categorical columns
    le = preprocessing.LabelEncoder()
    print("(1) \tLabel encoding the columns for training and testing set..")  
    # Label encoding the columns for the test and training set
    test[categorical] = test[categorical].apply(le.fit_transform)
    train[categorical] = train[categorical].apply(le.fit_transform)

    print("(2) \tApplying Standardscaler on training dataset..")
    # Applying StandardScaler on train to normalize the values.
    ss = StandardScaler()
    train = pd.DataFrame(ss.fit_transform(train),columns = train.columns)
    print("(3) \tDone!")
    return train
    
    