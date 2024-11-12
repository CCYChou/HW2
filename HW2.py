import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import optuna

# Load datasets
train_data = pd.read_csv(r"C:\Users\administrator\Desktop\HW2\train.csv")
test_data = pd.read_csv(r"C:\Users\administrator\Desktop\HW2\test.csv")

# Drop unnecessary columns
train_data = train_data.drop(columns=["Name", "Ticket", "Cabin"])
test_data = test_data.drop(columns=["Name", "Ticket", "Cabin"])

# Encode categorical columns
train_data["Sex"] = train_data["Sex"].map({"male": 0, "female": 1})
test_data["Sex"] = test_data["Sex"].map({"male": 0, "female": 1})
train_data = pd.get_dummies(train_data, columns=["Embarked"], drop_first=True)
test_data = pd.get_dummies(test_data, columns=["Embarked"], drop_first=True)

# Make sure both train and test datasets have the same columns after encoding
test_data = test_data.reindex(columns=train_data.columns.drop("Survived"), fill_value=0)

# Split features and target variable
X = train_data.drop(columns=["Survived"])
y = train_data["Survived"]

# Impute missing values for numeric columns
numeric_imputer = SimpleImputer(strategy="median")
X[X.select_dtypes(include=[np.number]).columns] = numeric_imputer.fit_transform(X.select_dtypes(include=[np.number]))

# Apply imputer to test data numeric columns
test_data[test_data.select_dtypes(include=[np.number]).columns] = numeric_imputer.transform(
    test_data.select_dtypes(include=[np.number])
)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define RFE function
def rfe_feature_selection(X, y, n_features=5):
    model = RandomForestClassifier(random_state=42)
    selector = RFE(model, n_features_to_select=n_features, step=1)
    selector = selector.fit(X, y)
    return selector.transform(X), selector

# Apply RFE
X_train_rfe, rfe_selector = rfe_feature_selection(X_train, y_train)

# Define SelectKBest function
def select_k_best_features(X, y, k=5):
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    return selector.transform(X), selector

# Apply SelectKBest
X_train_kbest, kbest_selector = select_k_best_features(X_train, y_train)

# Define objective function for Optuna
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth = trial.suggest_int("max_depth", 2, 32)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    
    score = cross_val_score(model, X_train_kbest, y_train, cv=5, scoring="accuracy")
    return score.mean()

# Optimize hyperparameters with Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Get the best parameters and train the model
best_params = study.best_params
model = RandomForestClassifier(**best_params, random_state=42)
model.fit(X_train_kbest, y_train)

# Evaluate on validation set
X_val_kbest = kbest_selector.transform(X_val)
y_pred = model.predict(X_val_kbest)
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)

print("Best Parameters from Optuna:", best_params)
print("Validation Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# Transform and predict on test set
X_test_kbest = kbest_selector.transform(test_data)
test_predictions = model.predict(X_test_kbest)

# Ensure PassengerId is int and create submission file
test_data["PassengerId"] = test_data["PassengerId"].astype(int)
submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": test_predictions})
submission.to_csv(r"C:\Users\administrator\Desktop\HW2\submission.csv", index=False)
