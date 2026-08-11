import sys
import subprocess

def main():
    print("🚀 --- OrbitAI Complete Execution Manager ---")
    
    python_exe = sys.executable
    print(f"Using Python executable: {python_exe}\n")
    
    print("1️⃣ Generating historical launch & weather dataset...")
    res = subprocess.run([python_exe, "generate_dataset.py"])
    if res.returncode != 0:
        print("❌ Error generating dataset.")
        return
        
    print("\n2️⃣ Running data preprocessing & scaling...")
    res = subprocess.run([python_exe, "preprocess_data.py"])
    if res.returncode != 0:
        print("❌ Error preprocessing data.")
        return
        
    print("\n3️⃣ Training ML models (Regression & Classification)...")
    res = subprocess.run([python_exe, "train_models.py"])
    if res.returncode != 0:
        print("❌ Error training models.")
        return
        
    print("\n🎉 All ML models trained and saved to models/")
    print("Launching Streamlit dashboard...\n")
    subprocess.run([python_exe, "-m", "streamlit", "run", "app/app.py"])

if __name__ == "__main__":
    main()
