import os
import numpy as np
import pandas as pd

def generate_historical_launch_dataset(num_samples=1200, random_seed=42):
    np.random.seed(random_seed)
    
    rocket_config = {
        "Falcon 9": {
            "capacity_leo": 22800,
            "capacity_geo": 8300,
            "sites": ["CCAFS SLC-40", "KSC LC-39A", "VAFB SLC-4E"]
        },
        "PSLV": {
            "capacity_leo": 3800,
            "capacity_geo": 1200,
            "sites": ["SDSC SHAR First Launch Pad", "SDSC SHAR Second Launch Pad"]
        },
        "GSLV": {
            "capacity_leo": 10000,
            "capacity_geo": 4000,
            "sites": ["SDSC SHAR Second Launch Pad"]
        },
        "Electron": {
            "capacity_leo": 300,
            "capacity_geo": 0,
            "sites": ["Mahia Launch Complex 1", "MARS Pad 0C"]
        }
    }
    
    rockets = list(rocket_config.keys())
    data = []
    
    for i in range(num_samples):
        r_type = np.random.choice(rockets, p=[0.45, 0.25, 0.15, 0.15])
        cfg = rocket_config[r_type]
        
        if r_type == "Electron":
            orbit = np.random.choice(["LEO", "MEO"], p=[0.85, 0.15])
        else:
            orbit = np.random.choice(["LEO", "MEO", "GEO"], p=[0.55, 0.25, 0.20])
            
        max_cap = cfg["capacity_leo"] if orbit == "LEO" else (cfg["capacity_geo"] if cfg["capacity_geo"] > 0 else 300)
        payload = int(np.random.uniform(min(100, max_cap), max_cap * 0.95))
        
        site = np.random.choice(cfg["sites"])
        
        wind_speed = round(float(np.random.gamma(shape=3.0, scale=8.0)), 1)
        wind_speed = min(wind_speed, 90.0)
        
        temperature = round(float(np.random.normal(loc=24.0, scale=8.0)), 1)
        temperature = np.clip(temperature, -5.0, 42.0)
        
        humidity = round(float(np.random.uniform(25.0, 98.0)), 1)
        
        if wind_speed > 48.0 or np.random.rand() < 0.08:
            weather = "Thunderstorm / High Wind"
        elif humidity > 82.0 or np.random.rand() < 0.15:
            weather = "Rainy"
        elif humidity > 60.0 or np.random.rand() < 0.25:
            weather = "Cloudy"
        else:
            weather = "Clear"
            
        launch_month = int(np.random.randint(1, 13))
        launch_year = int(np.random.randint(2013, 2026))
        
        if orbit == "LEO":
            base_time = np.random.uniform(8.5, 11.5)
        elif orbit == "MEO":
            base_time = np.random.uniform(45.0, 90.0)
        else:
            base_time = np.random.uniform(190.0, 340.0)
            
        payload_ratio = payload / max_cap if max_cap > 0 else 0.5
        payload_time_delta = payload_ratio * (3.5 if orbit == "LEO" else 25.0)
        
        wind_drag_delta = (wind_speed / 100.0) * 0.8
        
        flight_time_minutes = round(base_time + payload_time_delta + wind_drag_delta, 2)
        
        if wind_speed > 45.0 or weather == "Thunderstorm / High Wind" or temperature < 0.0 or temperature > 38.0:
            launch_risk = "High"
        elif wind_speed > 30.0 or weather == "Rainy" or humidity > 85.0:
            launch_risk = "Medium"
        else:
            launch_risk = "Low"
            
        data.append({
            "rocket_type": r_type,
            "payload_mass_kg": payload,
            "target_orbit": orbit,
            "launch_site": site,
            "wind_speed_kmh": wind_speed,
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "weather_condition": weather,
            "launch_month": launch_month,
            "launch_year": launch_year,
            "flight_time_minutes": flight_time_minutes,
            "launch_risk": launch_risk
        })
        
    df = pd.DataFrame(data)
    
    out_dir = os.path.join("data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "historical_launches.csv")
    df.to_csv(out_path, index=False)
    print(f"[OK] Generated {len(df)} historical launch records saved to {out_path}")
    return df

if __name__ == "__main__":
    generate_historical_launch_dataset()
