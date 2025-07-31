import subprocess
import sys
import os

scripts = [
    "01_filter_data.py",
    "02_extract_features.py",
    "03_build_dataset.py",
    "04_data_quality.py",
    "05_feature_target_correlation.py"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_pipeline():
    for script in scripts:
        script_path = os.path.join(BASE_DIR, script)
        print(f"\nRunning {script}...")
        result = subprocess.run([sys.executable, script_path])
        if result.returncode != 0:
            print(f"{script} failed with exit code {result.returncode}. Stopping.")
            sys.exit(result.returncode)
        print(f"{script} completed successfully.")


if __name__ == "__main__":
    print("Starting full PRIMA pipeline...")
    run_pipeline()
    print("Pipeline completed successfully.")
