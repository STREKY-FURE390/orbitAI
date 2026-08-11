# OrbitAI 🚀

**Intelligent Rocket Flight Time Prediction and Launch Analysis using Machine Learning**

> *"Predicting rocket flight time and analyzing launch conditions using AI."*

---

## 📌 Project Overview & Problem Statement

**OrbitAI** is an educational Machine Learning and Aerospace Analytics project designed to analyze historical rocket launch, mission, and atmospheric environmental data. 

In aerospace engineering, determining flight durations and evaluating environmental launch conditions (such as wind speed, temperature, and atmospheric pressure) are crucial operational concerns. OrbitAI demonstrates how data science, statistical analysis, supervised Machine Learning, and Explainable AI (XAI) can model these historical flight parameters and launch risk factors.

---

## 🎯 Project Objectives

1. **Flight Time Regression**: Estimate rocket flight/orbit-reach duration based on payload mass, vehicle parameters, and orbital trajectory requirements when reliable historical target data is available.
2. **Launch Risk Classification**: Classify launch delay / scrub risk (Low, Medium, High) from environmental and mission attributes.
3. **Explainable AI (XAI)**: Provide quantitative feature importance and human-interpretable reasoning for why a prediction was made.
4. **Interactive Aerospace Dashboard**: Provide an intuitive, professional Streamlit interface for scenario testing and parameter exploration.
5. **Aerospace Data Education**: Demonstrate fundamental concepts in orbital mechanics, trajectory dynamics, and atmospheric influences on launch vehicles.

---

## ⚠️ Important Educational Disclaimer

> **OrbitAI is strictly an educational Machine Learning project intended for learning, experimentation, and demonstration.**
>
> - It is **NOT** a real rocket flight controller.
> - It is **NOT** a flight-critical safety system.
> - It is **NOT** an operational Go/No-Go launch decision software.
> - It does **NOT** control physical rockets.
> - It does **NOT** replace aerospace systems engineers.
>
> Real rocket launches require complex multiphysics simulations, telemetry, real-time guidance navigation and control (GNC), and stringent NASA/ESA/ISRO safety protocols.

---

## 🚀 Fundamental Aerospace Concepts

- **Rocket (Launch Vehicle)**: Generates high thrust via chemical reaction mass ejection to carry payloads into space.
- **Payload**: The scientific, commercial, or military equipment (e.g., satellite, space station module) transported by the rocket.
- **Thrust & Newton's Third Law**: *"For every action, there is an equal and opposite reaction."* Exhaust gas accelerated downwards pushes the rocket upward.
- **Gravity & Orbital Velocity**: Space (including Low Earth Orbit) is **not zero gravity**. Gravity at 400 km altitude is still ~90% of sea-level gravity. A spacecraft stays in orbit because it moves at very high **horizontal velocity** (~7.8 km/s for LEO), continuously falling around Earth's curved horizon.
- **Orbital Regimes**:
  - **LEO (Low Earth Orbit)**: ~160 km to 2,000 km altitude (e.g., ISS).
  - **MEO (Medium Earth Orbit)**: ~2,000 km to 35,786 km altitude (e.g., GPS satellites).
  - **GEO (Geostationary Orbit)**: Exactly ~35,786 km altitude. The orbital period equals Earth's rotation period (24 hours), keeping the satellite fixed over one longitude.

---

## 🏗️ Application Architecture

```
                 User Input (Sidebar)
                          │
                          ▼
               Input Validation & Pipeline
                          │
                          ▼
            Feature Engineering & Scaling
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
Flight Time Regressor           Launch Risk Classifier
 (Scikit-Learn / XGB)              (Scikit-Learn / XGB)
         │                                 │
         └────────────────┬────────────────┘
                          ▼
             Explainable AI (XAI) Engine
            (Feature Importance / SHAP)
                          │
                          ▼
          Streamlit Interactive Dashboard
```

---

## 📁 Project Structure

```
OrbitAI/
├── data/
│   ├── raw/                # Historical raw launch and weather logs
│   └── processed/          # Cleaned, encoded, and scaled datasets
├── notebooks/
│   ├── 01_EDA.ipynb        # Exploratory Data Analysis & statistical visualization
│   ├── 02_Preprocessing.ipynb # Feature encoding, missing handling & scaling
│   └── 03_Model_Training.ipynb # Model comparison, hyperparameter tuning & export
├── models/
│   ├── flight_time_model.pkl   # Serialized trained regressor
│   └── launch_risk_model.pkl   # Serialized trained classifier
├── app/
│   ├── app.py              # Main Streamlit dashboard application
│   ├── components/         # Modular UI UI layout elements
│   └── utils/              # Model loaders, helper functions & XAI formatters
├── report/                 # Research summary and statistical reports
├── presentation/           # Slides and educational material
├── requirements.txt        # Python package dependencies
├── LICENSE                 # Open-source MIT License
└── README.md               # Project documentation
```

---

## 📊 Machine Learning Problem Formulation

### Task 1: Regression (Flight / Orbit-Reach Time)
- **Goal**: Predict `flight_time_minutes` (continuous numeric target).
- **Candidate Models**: Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor.
- **Evaluation Metrics**:
  - **MAE (Mean Absolute Error)**: Average magnitude of prediction errors in minutes.
  - **MSE (Mean Squared Error)**: Penalizes larger prediction errors exponentially.
  - **RMSE (Root Mean Squared Error)**: Standard deviation of residual errors.
  - **R² (Coefficient of Determination)**: Proportion of variance explained by the model.

### Task 2: Classification (Launch Risk Assessment)
- **Goal**: Predict `launch_risk` (`Low`, `Medium`, `High`).
- **Candidate Models**: Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Gradient Boosting Classifier.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.

---

## 🔍 Explainable AI (XAI) Principles

OrbitAI prioritizes scientific clarity and non-causal wording:
- ✅ *"Higher wind speed contributed to the model's unfavorable launch-condition prediction."*
- ✅ *"Payload mass was influential in the flight time model prediction."*
- ❌ *"Wind speed directly caused launch failure."* (Not claimed without direct causal modeling).

---

## ⚙️ Installation & Usage

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Environment Setup (VS Code / Terminal)
```bash
# Navigate to the workspace folder
cd OrbitAI

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app/app.py
```

---

## 🔮 Future Enhancements

1. **Bayesian Networks**: Probabilistic graph models representing conditional launch risks.
2. **Particle Swarm Optimization (PSO)**: Metaheuristic hyperparameter optimization.
3. **DOLILU Inspiration**: Day-of-Launch I-Loads Update adaptive weather profile integration.
4. **Weakly-Hard Real-Time Bounds**: Educational simulation of real-time telemetry constraint monitoring.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
