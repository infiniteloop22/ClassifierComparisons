import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import ConfusionMatrixDisplay

def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop("Label", axis=1)
    y = df["Label"]

    return X, y

def preprocessing(X):
    # preparing a list of categorical columns and numerical columns for preprocessing
    categorical_columns = ['worktype', 'EducationLevel', 'marital_status', 'CurrentOccupation', 'RelationshipStatus', 'race', 'Gender']
    numerical_columns = ['age', 'fnlwgt', 'educationnum', 'capitalgain', 'capitalloss', 'hoursperweek']

    # one-hot encoding turns categorical columns into binary numbers
    # standard scaler removes units from numerical columns by using the z-score formula
    preprocessor = ColumnTransformer(transformers=[('categorical', OneHotEncoder(), categorical_columns), ('numerical', StandardScaler(), numerical_columns)])
    
    # applying one-hot encoding and feature scaling to the input data then converting the result into a dense array for models
    X_processed = preprocessor.fit_transform(X).toarray()

    return X_processed

def split_test_train(test_percent, X, y):
    return train_test_split(X, y, test_size=test_percent, random_state=22) # setting a fixed state to split into training and testing sets identically, every time

def decision_tree(X_train, Y_train, X_test, Y_test):
    model = DecisionTreeClassifier(random_state=22) # setting a fixed state in model's class constructor
    model.fit(X_train, Y_train)
    score = model.score(X_test, Y_test)
    return model, score

def naive_bayesian(X_train, Y_train, X_test, Y_test):
    model = GaussianNB()
    model.fit(X_train, Y_train)
    score = model.score(X_test, Y_test)
    return model, score

def confusion_matrix(y_true, y_pred):
    return ConfusionMatrixDisplay.from_predictions(y_true, y_pred)

def main():
    X, y = load_data("ClassificationLabData.csv")

    X_processed = preprocessing(X)

    X_train_scaled, X_test_scaled, y_train, y_test = split_test_train(.2, X_processed, y) # 80/20 split

    models = [decision_tree(X_train_scaled, y_train, X_test_scaled, y_test), 
              naive_bayesian(X_train_scaled, y_train, X_test_scaled, y_test)]
    
    for model, score in models:
        print(f"Model: {model} | Testing score result: {score}")
        y_pred = model.predict(X_test_scaled)
        confusion_matrix(y_test, y_pred)
        plt.show()

if __name__ == "__main__":
    main()