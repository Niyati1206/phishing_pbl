import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Read the CSV files and create pandas dataframes
legitimate_df = pd.read_csv("structured_data_legitimate.csv")
phishing_df = pd.read_csv("structured_data_phishing.csv")

# Combine legitimate and phishing dataframes, and shuffle
df = pd.concat([legitimate_df, phishing_df], axis=0).sample(frac=1)

# Remove 'URL' column and duplicates
df = df.drop('URL', axis=1).drop_duplicates()

# Create X and Y for the models
X = df.drop('label', axis=1)
Y = df['label']

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# Create ML models
svm_model = svm.LinearSVC()
rf_model = RandomForestClassifier(n_estimators=60)
dt_model = tree.DecisionTreeClassifier()
ab_model = AdaBoostClassifier()
nb_model = GaussianNB()
nn_model = MLPClassifier(alpha=1)
kn_model = KNeighborsClassifier()

# Train the models
svm_model.fit(x_train, y_train)
rf_model.fit(x_train, y_train)
dt_model.fit(x_train, y_train)
ab_model.fit(x_train, y_train)
nb_model.fit(x_train, y_train)
nn_model.fit(x_train, y_train)
kn_model.fit(x_train, y_train)

# Make predictions
svm_predictions = svm_model.predict(x_test)
rf_predictions = rf_model.predict(x_test)
dt_predictions = dt_model.predict(x_test)
ab_predictions = ab_model.predict(x_test)
nb_predictions = nb_model.predict(x_test)
nn_predictions = nn_model.predict(x_test)
kn_predictions = kn_model.predict(x_test)

# Create confusion matrices
confusion_matrices = {
    "SVM": confusion_matrix(y_test, svm_predictions),
    "Random Forest": confusion_matrix(y_test, rf_predictions),
    "Decision Tree": confusion_matrix(y_test, dt_predictions),
    "AdaBoost": confusion_matrix(y_test, ab_predictions),
    "Gaussian Naive Bayes": confusion_matrix(y_test, nb_predictions),
    "Neural Network": confusion_matrix(y_test, nn_predictions),
    "K-Nearest Neighbors": confusion_matrix(y_test, kn_predictions)
}

# Calculate accuracy, precision, and recall scores
results = {}
for model, matrix in confusion_matrices.items():
    tn, fp, fn, tp = matrix.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    results[model] = {"Accuracy": accuracy, "Precision": precision, "Recall": recall}

# Print results
for model, scores in results.items():
    print(f"--- {model} ---")
    print(f"Accuracy: {scores['Accuracy']}")
    print(f"Precision: {scores['Precision']}")
    print(f"Recall: {scores['Recall']}")
    print("-------------------")

# Return results for access in other modules if needed
df_results = pd.DataFrame(results)
