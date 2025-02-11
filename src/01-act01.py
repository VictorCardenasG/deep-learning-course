from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine_data = load_wine()

# Convert data to pandas dataframe
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

# Split data into features and labels
X = wine_df[wine_data.feature_names].copy()
y = wine_data["target"].copy()

# Instantiate scaler and fit on features
scaler = StandardScaler()
scaler.fit(X)

# Transform features
X_scaled = scaler.transform(X.values)

# View first instance
print(X_scaled[0])

# Split data into train and test
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, train_size = .7, random_state=25)

# Check the splits are correct
print(f"""Train size: {round(len(X_train_scaled)/len(X)*100)}% 
Test Size: {round(len(X_test_scaled)/len(X)*100)}%""")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Instantiating the models
logistic_regression = LogisticRegression()
svm = SVC()
tree = DecisionTreeClassifier()

# Training the models
logistic_regression.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)
tree.fit(X_train_scaled, y_train)

# Making predictions with each model
log_reg_preds = logistic_regression.predict(X_test_scaled)
svm_preds = svm.predict(X_test_scaled)
tree_preds = tree.predict(X_test_scaled)

from sklearn.metrics import classification_report

# Store the model predictions in a dictionary

model_preds = {
        "Logistic Regression": log_reg_preds,
        "Support Vector Machine": svm_preds,
        "Decission Tree": tree_preds
}

for model, preds in model_preds.items():
    print(f"""{model} Results:{classification_report(y_test, preds)}""", sep="\n\n")