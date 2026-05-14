import os
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PSEUDO_PARAM_DIR = os.path.join(BASE_DIR, "outputs", "pseudo_gt", "params")
PSEUDO_TRAJ_DIR = os.path.join(BASE_DIR, "outputs", "pseudo_gt", "trajectories")
PSEUDO_OUTCOME_DIR = os.path.join(BASE_DIR, "outputs", "pseudo_gt", "outcomes")

LLM_PARAM_DIR = os.path.join(BASE_DIR, "outputs", "llm_predictions_fewshot")
LLM_TRAJ_DIR = os.path.join(BASE_DIR, "outputs", "llm_trajectories_fewshot")
LLM_OUTCOME_DIR = os.path.join(BASE_DIR, "outputs", "llm_outcomes_fewshot")

OUT_DIR = os.path.join(BASE_DIR, "outputs", "evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

PARAM_KEYS = [
    "ego_speed_kmh",
    "steer_trigger_time",
    "steer_magnitude",
    "steer_duration",
    "counter_steer_magnitude",
    "brake_factor",
    "friction_scale",
    "target_yaw_change",
]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def param_error(gt, pred):
    errs = {}
    vals = []

    for k in PARAM_KEYS:
        e = abs(float(gt[k]) - float(pred[k]))
        errs[f"{k}_abs_error"] = e
        vals.append(e)

    errs["mean_param_abs_error"] = float(np.mean(vals))
    return errs


def trajectory_error(gt_csv, pred_csv):
    gt = pd.read_csv(gt_csv)
    pred = pd.read_csv(pred_csv)

    n = min(len(gt), len(pred))
    gt = gt.iloc[:n]
    pred = pred.iloc[:n]

    pos_err = np.sqrt(
        (gt["x"].values - pred["x"].values) ** 2 +
        (gt["y"].values - pred["y"].values) ** 2
    )

    yaw_err = np.abs(gt["yaw_change"].values - pred["yaw_change"].values)

    return {
        "mean_xy_error": float(np.mean(pos_err)),
        "final_xy_error": float(pos_err[-1]),
        "mean_yaw_error": float(np.mean(yaw_err)),
        "final_yaw_error": float(yaw_err[-1]),
    }


def outcome_error(gt, pred):
    return {
        "gt_lane_departure": gt.get("lane_departure_success"),
        "pred_lane_departure": pred.get("lane_departure_success"),
        "gt_roadside_contact": gt.get("roadside_contact_success"),
        "pred_roadside_contact": pred.get("roadside_contact_success"),
        "lane_departure_match": gt.get("lane_departure_success") == pred.get("lane_departure_success"),
        "roadside_contact_match": gt.get("roadside_contact_success") == pred.get("roadside_contact_success"),
        "yaw_change_error": abs(float(gt.get("max_yaw_change", 0)) - float(pred.get("max_yaw_change", 0))),
        "lateral_offset_error": abs(float(gt.get("max_lateral_offset", 0)) - float(pred.get("max_lateral_offset", 0))),
    }


def main():
    pred_files = [
        f for f in os.listdir(LLM_PARAM_DIR)
        if f.endswith("_fewshot.json")
    ]

    rows = []

    for fname in pred_files:
        case_id = fname.replace("_fewshot.json", "")

        gt_param_path = os.path.join(PSEUDO_PARAM_DIR, f"{case_id}_carla.json")
        pred_param_path = os.path.join(LLM_PARAM_DIR, f"{case_id}_fewshot.json")

        gt_traj_path = os.path.join(PSEUDO_TRAJ_DIR, f"{case_id}.csv")
        pred_traj_path = os.path.join(LLM_TRAJ_DIR, f"{case_id}.csv")

        gt_outcome_path = os.path.join(PSEUDO_OUTCOME_DIR, f"{case_id}_outcome.json")
        pred_outcome_path = os.path.join(LLM_OUTCOME_DIR, f"{case_id}_outcome.json")

        if not all(os.path.exists(p) for p in [
            gt_param_path, pred_param_path,
            gt_traj_path, pred_traj_path,
            gt_outcome_path, pred_outcome_path
        ]):
            print("Skipping missing:", case_id)
            continue

        gt_param = load_json(gt_param_path)
        pred_param = load_json(pred_param_path)

        gt_outcome = load_json(gt_outcome_path)
        pred_outcome = load_json(pred_outcome_path)

        row = {
            "case_id": case_id,
            "method": "HF TinyLlama Few-shot"
        }

        row.update(param_error(gt_param, pred_param))
        row.update(trajectory_error(gt_traj_path, pred_traj_path))
        row.update(outcome_error(gt_outcome, pred_outcome))

        rows.append(row)

    df = pd.DataFrame(rows)

    out_csv = os.path.join(OUT_DIR, "fewshot_vs_pseudogt_eval.csv")
    df.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)

    if len(df) > 0:
        print("\nPer-case:")
        print(df[[
            "case_id",
            "mean_param_abs_error",
            "mean_xy_error",
            "mean_yaw_error",
            "lane_departure_match",
            "roadside_contact_match",
            "yaw_change_error",
            "lateral_offset_error"
        ]])

        print("\nSummary:")
        print(df[[
            "mean_param_abs_error",
            "mean_xy_error",
            "mean_yaw_error",
            "lane_departure_match",
            "roadside_contact_match",
            "yaw_change_error",
            "lateral_offset_error"
        ]].mean(numeric_only=True))


if __name__ == "__main__":
    main()