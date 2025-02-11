import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Load the data
df = pd.read_csv(r"C:\Users\Victor Cardenas\Documents\msc\semestre-4\deep-learning\data\raw\diabetes.csv")

# Split features and labels
X = df.drop(columns="Outcome", axis=1)
y = df['Outcome']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X_scaled, y, train_size=0.7, random_state=25
)

# Define the parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],                
    "learning_rate": [0.01, 0.1, 1.0],            
    "estimator": [
        DecisionTreeClassifier(max_depth=1),      
        DecisionTreeClassifier(max_depth=3),      
    ],
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=AdaBoostClassifier(),
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,   # 5-fold cross-validation
    verbose=1,  # Show progress
    n_jobs=-1,  # Use all CPU cores
)

# Fit GridSearchCV to training data
grid_search.fit(X_train_scaled, y_train)

# Display the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Use the best model to make predictions
best_adaboost = grid_search.best_estimator_
adaboost_preds = best_adaboost.predict(X_test_scaled)

# Evaluate the model
print(f"""Optimized AdaBoost Results:\n{classification_report(y_test, adaboost_preds)}""")
