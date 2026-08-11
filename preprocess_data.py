import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocess_and_save_data():
    raw_path = os.path.join("data", "raw", "historical_launches.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}. Run generate_dataset.py first.")
        
    df = pd.read_csv(raw_path)
    print(f"[LOAD] Loaded raw dataset shape: {df.shape}")
    
    categorical_features = ["rocket_type", "target_orbit", "launch_site", "weather_condition"]
    numeric_features = ["payload_mass_kg", "wind_speed_kmh", "temperature_c", "humidity_pct", "launch_month", "launch_year"]
    
    X = df[categorical_features + numeric_features]
    y_reg = df["flight_time_minutes"]
    y_clf = df["launch_risk"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features)
        ]
    )
    
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.20, random_state=42, stratify=y_clf
    )
    
    preprocessor.fit(X_train)
    
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    proc_dir = os.path.join("data", "processed")
    os.makedirs(proc_dir, exist_ok=True)
    
    joblib.dump(preprocessor, os.path.join(proc_dir, "preprocessor.pkl"))
    
    pd.DataFrame(X_train_trans).to_csv(os.path.join(proc_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_test_trans).to_csv(os.path.join(proc_dir, "X_test.csv"), index=False)
    y_reg_train.to_csv(os.path.join(proc_dir, "y_reg_train.csv"), index=False)
    y_reg_test.to_csv(os.path.join(proc_dir, "y_reg_test.csv"), index=False)
    y_clf_train.to_csv(os.path.join(proc_dir, "y_clf_train.csv"), index=False)
    y_clf_test.to_csv(os.path.join(proc_dir, "y_clf_test.csv"), index=False)
    
    print("[OK] Preprocessing complete. Pipelines and train/test splits saved to data/processed/")

if __name__ == "__main__":
    preprocess_and_save_data()
