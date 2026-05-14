import os
import glob
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PARAM_DIR = os.path.join(BASE_DIR, "outputs", "carla_params")
TRAJ_DIR = os.path.join(BASE_DIR, "outputs", "trajectories")
OUTCOME_DIR = os.path.join(BASE_DIR, "outputs", "outcomes")

os.makedirs(TRAJ_DIR, exist_ok=True)
os.makedirs(OUTCOME_DIR, exist_ok=True)

param_files = sorted(glob.glob(os.path.join(PARAM_DIR, "*_carla.json")))

print(f"Found {len(param_files)} CARLA parameter files")

for param_path in param_files:
    case_id = os.path.basename(param_path).replace("_carla.json", "")

    traj_out = os.path.join(TRAJ_DIR, f"{case_id}.csv")
    outcome_out = os.path.join(OUTCOME_DIR, f"{case_id}_outcome.json")

    print(f"\nRunning case {case_id}")

    cmd = [
        "python3",
        "scripts/04_run_carla_single.py",
        "--params", param_path,
        "--traj_out", traj_out,
        "--outcome_out", outcome_out,
        "--duration", "12"
    ]

    subprocess.run(cmd)