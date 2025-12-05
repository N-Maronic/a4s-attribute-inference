#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
import kagglehub

# ------------------------------
# Helpers
# ------------------------------

def info(msg):
    print(f"[INFO] {msg}")

def error(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)

def run(cmd, cwd=None):
    """Run a shell command and stream output."""
    result = subprocess.run(cmd, cwd=cwd, shell=True)
    if result.returncode != 0:
        error(f"Command failed: {cmd}")


# ------------------------------
# 1. Ask user for a4s-eval path
# ------------------------------

print("=========================================================")
print(" Attribute Inference Metric — Setup Script (Python)")
print("=========================================================\n")

a4s_path = input("Enter FULL PATH to your a4s-eval root directory:\n> ").strip()

if not os.path.isdir(a4s_path):
    error(f"Directory does not exist: {a4s_path}")

expected_subdirs = ["a4s_eval", "tests"]
missing = [d for d in expected_subdirs if not os.path.isdir(os.path.join(a4s_path, d))]

if missing:
    error(
        "The provided directory does NOT appear to be the root of a4s-eval.\n"
        f"Missing required subdirectories: {missing}\n"
        "Please point to the folder that contains both /a4s_eval and /tests."
    )

info(f"Using a4s-eval root: {a4s_path}")

# ------------------------------
# 2. Download Kaggle dataset
# ------------------------------

info("Downloading Adult Census dataset from Kaggle...")

data_dir = os.path.join(a4s_path, "data")
os.makedirs(data_dir, exist_ok=True)

# kagglehub handles ZIP download + extraction automatically
dataset_path = kagglehub.dataset_download("uciml/adult-census-income")

info(f"Dataset downloaded to: {dataset_path}")

# Expected files include: adult.csv, adult.test, adult.names
src_csv = os.path.join(dataset_path, "adult.csv")

if not os.path.isfile(src_csv):
    error("adult.csv not found in downloaded dataset directory. Check dataset structure.")

adult_csv = os.path.join(data_dir, "adult.csv")
shutil.copy(src_csv, adult_csv)

info(f"adult.csv copied to: {adult_csv}")

# ------------------------------
# 3. Install metric + test + notebook + pyproject.toml
# ------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))

info("Copying metric file...")
shutil.copy(
    os.path.join(script_dir, "attribute_inference.py"),
    os.path.join(a4s_path, "a4s_eval/metrics/model_metrics/")
)

info("Copying test file...")
shutil.copy(
    os.path.join(script_dir, "test_attribute_inference.py"),
    os.path.join(a4s_path, "tests/metrics/model_metrics/")
)

info("Copying notebook...")
shutil.copy(
    os.path.join(script_dir, "attribute_inference_demo.ipynb"),
    os.path.join(a4s_path, "")
)

info("Updating pyproject.toml...")
shutil.copy(
    os.path.join(script_dir, "pyproject.toml"),
    os.path.join(a4s_path, "pyproject.toml")
)

# ------------------------------
# 4. Run uv sync
# ------------------------------

info("Running uv sync...")
run("uv sync", cwd=a4s_path)

# ------------------------------
# 5. Done!
# ------------------------------

print("\n=========================================================")
print("     Setup Completed Successfully!")
print("=========================================================\n")

print("The dataset is available at:")
print(f"   {adult_csv}\n")

print("The metric, test, and notebook have been installed into your a4s-eval environment.")
print("\nYou can now run:")
print("   uv run pytest -s tests/metrics/model_metrics/")
print("\nOr open the notebook:")
print(f"   {os.path.join(a4s_path, 'attribute_inference_demo.ipynb')}")
