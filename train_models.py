import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

def train_and_evaluate_models():
    raw_path = os.path.join("data", "raw", "historical_launches.csv")
    proc_dir = os.path.join("data", "processed")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    df = pd.read_csv(raw_path)
    
    categorical_features = ["rocket_type", "target_orbit", "launch_site", "weather_condition"]
    numeric_features = ["payload_mass_kg", "wind_speed_kmh", "temperature_c", "humidity_pct", "launch_month", "launch_year"]
    
    X = df[categorical_features + numeric_features]
    y_reg = df["flight_time_minutes"]
    y_clf = df["launch_risk"]
    
    preprocessor = joblib.load(os.path.join(proc_dir, "preprocessor.pkl"))
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.20, random_state=42, stratify=y_clf
    )
    
    print("\n==================================================")
    print("TASK 1: REGRESSION (Flight Time Prediction)")
    print("==================================================")
    
    regressors = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    best_reg_name = None
    best_reg_model = None
    best_r2 = -float("inf")
    reg_results = {}
    
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    for name, model in regressors.items():
        model.fit(X_train_trans, y_reg_train)
        preds = model.predict(X_test_trans)
        
        mae = mean_absolute_error(y_reg_test, preds)
        mse = mean_squared_error(y_reg_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_reg_test, preds)
        
        reg_results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
        print(f"[{name}] MAE: {mae:.3f} min | RMSE: {rmse:.3f} min | R2: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_reg_name = name
            best_reg_model = model
            
    print(f"\n[BEST] Best Regressor: {best_reg_name} (R2 = {best_r2:.4f})")
    
    reg_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", best_reg_model)
    ])
    
    reg_model_path = os.path.join(models_dir, "flight_time_model.pkl")
    joblib.dump(reg_pipeline, reg_model_path)
    print(f"[SAVE] Saved Regressor Pipeline to {reg_model_path}")
    
    print("\n==================================================")
    print("TASK 2: CLASSIFICATION (Launch Delay Risk)")
    print("==================================================")
    
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    best_clf_name = None
    best_clf_model = None
    best_f1 = -float("inf")
    clf_results = {}
    
    for name, model in classifiers.items():
        model.fit(X_train_trans, y_clf_train)
        preds = model.predict(X_test_trans)
        
        acc = accuracy_score(y_clf_test, preds)
        prec = precision_score(y_clf_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_clf_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_clf_test, preds, average="weighted", zero_division=0)
        
        clf_results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
        print(f"[{name}] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_clf_name = name
            best_clf_model = model
            
    print(f"\n[BEST] Best Classifier: {best_clf_name} (Weighted F1 = {best_f1:.4f})")
    
    clf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", best_clf_model)
    ])
    
    clf_model_path = os.path.join(models_dir, "launch_risk_model.pkl")
    joblib.dump(clf_pipeline, clf_model_path)
    print(f"[SAVE] Saved Classifier Pipeline to {clf_model_path}")
    
    return reg_results, clf_results

if __name__ == "__main__":
    train_and_evaluate_models()
