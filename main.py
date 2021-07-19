import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

st.title("Welcome to machine learning Classifier")

df_test = pd.read_csv(
    'data_testing.csv')
df_train = pd.read_csv(
    'data_training.csv')

st.write('Training set')
st.dataframe(df_train)

df_train_clean = df_train.drop(['index', 'feature_10'], axis=1)

df_test_clean = df_test.drop(['index', 'feature_10'], axis=1)

st.write('Cleaned Test set')
st.dataframe(df_test_clean)

st.write('Cleaned Training set')
st.dataframe(df_train_clean)

st.write(f"shape of test and train dataset are ",
         df_test.shape, df_train.shape)
st.write(f'the number of classes are ', len(
    np.unique(df_train['type'])), np.unique(df_train['type']))

classifier = st.sidebar.selectbox("Select Model Classifier",
                                  ("SGD", "SVM", "Decision tree[Random Forest]"))

st.subheader("Investigate dataset")


def add_parameter(classifier):
    params = dict()
    if classifier == "SGD":
        n = st.sidebar.selectbox("What kind of loss?", options=[
            "hinge", "log", "modified_huber"])
        params["What kind of loss?"] = n
    elif classifier == "SVM":
        c = st.sidebar.slider(
            "What is the significance of C?", 0.01, 10.0)
        params["What is the significance of C"] = c
    else:
        max_depth = st.sidebar.slider("What is max depth?", 2, 15)
        n_estimators = st.sidebar.slider("What is n_estimators?", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params


params = add_parameter(classifier)


def get_model(classifier, params):
    if classifier == "SGD":
        classifier = SGDClassifier(loss=params["What kind of loss?"])
    elif classifier == "SVM":
        classifier = SVC(C=params["What is the significance of C"])
    else:
        classifier = RandomForestClassifier(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=1234)
    return classifier


classifier = get_model(classifier, params)

# Classification
X_train = df_train_clean.iloc[:, :-1].values
y_train = df_train_clean.iloc[:, -1].values

X_test = df_test_clean.iloc[:, :].values
y_test = df_test_clean.iloc[:, :].values

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


st.write(f'classifier = {classifier}')
st.write(f'the y_test {y_test} predicted to class {y_pred}')

df_result1 = pd.DataFrame(y_test)
df_result2 = pd.DataFrame(y_pred)
df_result = pd.concat([df_result1, df_result2], axis=1, join="outer")
st.dataframe(df_result)
