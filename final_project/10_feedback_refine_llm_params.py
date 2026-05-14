import os
import json
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GT_OUTCOME_DIR = os.path.join(BASE_DIR, "outputs", "pseudo_gt", "outcomes")
LLM_PARAM_DIR = os.path.join(BASE_DIR, "outputs", "llm_predictions_fewshot")

REFINED_PARAM_DIR = os.path.join(BASE_DIR, "outputs", "llm_predictions_feedback")
REFINED_TRAJ_DIR = os.path.join(BASE_DIR, "outputs", "llm_trajectories_feedback")
REFINED_OUTCOME_DIR = os.path.join(BASE_DIR, "outputs", "llm_outcomes_feedback")

os.makedirs(REFINED_PARAM_DIR, exist_ok=True)
os.makedirs(REFINED_TRAJ_DIR, exist_ok=True)
os.makedirs(REFINED_OUTCOME_DIR, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def clamp_params(p):
    p["steer_magnitude"] = round(max(0.05, min(0.85, float(p["steer_magnitude"]))), 3)
    p["steer_duration"] = round(max(0.5, min(3.0, float(p["steer_duration"]))), 2)
    p["brake_factor"] = round(max(0.0, min(1.0, float(p["brake_factor"]))), 3)
    p["target_yaw_change"] = round(max(30, min(240, float(p["target_yaw_change"]))), 2)
    p["counter_steer_magnitude"] = round(max(-0.85, min(0.0, float(p["counter_steer_magnitude"]))), 3)
    return p


def refine_params(pred_params, gt_outcome, pred_outcome):
    refined = pred_params.copy()

    gt_yaw = float(gt_outcome["max_yaw_change"])
    pred_yaw = float(pred_outcome["max_yaw_change"])

    gt_lat = float(gt_outcome["max_lateral_offset"])
    pred_lat = float(pred_outcome["max_lateral_offset"])

    yaw_error = gt_yaw - pred_yaw
    lat_error = gt_lat - pred_lat

    if yaw_error > 25:
        refined["spin_mode"] = "aggressive"
    elif yaw_error < -25:
        refined["spin_mode"] = "mild"
    else:
        refined["spin_mode"] = "medium"

        
    # Need more spin
    if yaw_error > 25:
        refined["steer_magnitude"] += 0.08
        refined["steer_duration"] += 0.20
        refined["brake_factor"] += 0.08
        refined["counter_steer_magnitude"] -= 0.04

    # Too much spin
    elif yaw_error < -25:
        refined["steer_magnitude"] -= 0.07
        refined["brake_factor"] -= 0.06
        refined["counter_steer_magnitude"] += 0.04

    # Need more lateral departure
    if lat_error > 4:
        refined["steer_magnitude"] += 0.05
        refined["steer_duration"] += 0.15

    # Too much lateral departure
    elif lat_error < -4:
        refined["steer_magnitude"] -= 0.05
        refined["steer_duration"] -= 0.10

    refined["source"] = "hf_fewshot_plus_carla_feedback"

    return clamp_params(refined)


def run_carla(case_id, param_path):
    traj_out = os.path.join(REFINED_TRAJ_DIR, f"{case_id}.csv")
    outcome_out = os.path.join(REFINED_OUTCOME_DIR, f"{case_id}_outcome.json")

    cmd = [
        "python3",
        "scripts/04_run_carla_single.py",
        "--params", param_path,
        "--traj_out", traj_out,
        "--outcome_out", outcome_out,
        "--duration", "12"
    ]

    subprocess.run(cmd, check=True)


def main():
    pred_files = [
        f for f in os.listdir(LLM_PARAM_DIR)
        if f.endswith("_fewshot.json")
    ]

    for fname in pred_files:
        case_id = fname.replace("_fewshot.json", "")

        gt_outcome_path = os.path.join(GT_OUTCOME_DIR, f"{case_id}_outcome.json")
        pred_outcome_path = os.path.join(BASE_DIR, "outputs", "llm_outcomes_fewshot", f"{case_id}_outcome.json")
        pred_param_path = os.path.join(LLM_PARAM_DIR, fname)

        if not os.path.exists(gt_outcome_path) or not os.path.exists(pred_outcome_path):
            print("Skipping missing outcome:", case_id)
            continue

        gt_outcome = load_json(gt_outcome_path)
        pred_outcome = load_json(pred_outcome_path)
        pred_params = load_json(pred_param_path)

        refined = refine_params(pred_params, gt_outcome, pred_outcome)

        refined_path = os.path.join(REFINED_PARAM_DIR, f"{case_id}_feedback.json")
        save_json(refined, refined_path)

        print(f"\nRunning feedback refined case: {case_id}")
        run_carla(case_id, refined_path)

    print("\nFeedback refinement complete.")


if __name__ == "__main__":
    main()