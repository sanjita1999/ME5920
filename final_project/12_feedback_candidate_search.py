import os
import json
import subprocess
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GT_TRAJ_DIR = os.path.join(BASE_DIR, "outputs", "pseudo_gt", "trajectories")
LLM_PARAM_DIR = os.path.join(BASE_DIR, "outputs", "llm_predictions_fewshot")

SEARCH_PARAM_DIR = os.path.join(BASE_DIR, "outputs", "llm_predictions_search")
SEARCH_TRAJ_DIR = os.path.join(BASE_DIR, "outputs", "llm_trajectories_search")
SEARCH_OUTCOME_DIR = os.path.join(BASE_DIR, "outputs", "llm_outcomes_search")
SEARCH_LOG_DIR = os.path.join(BASE_DIR, "outputs", "search_logs")

for d in [SEARCH_PARAM_DIR, SEARCH_TRAJ_DIR, SEARCH_OUTCOME_DIR, SEARCH_LOG_DIR]:
    os.makedirs(d, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def clamp(p):
    p["steer_magnitude"] = round(max(0.05, min(0.85, float(p["steer_magnitude"]))), 3)
    p["steer_duration"] = round(max(0.5, min(3.0, float(p["steer_duration"]))), 2)
    p["brake_factor"] = round(max(0.0, min(1.0, float(p["brake_factor"]))), 3)
    p["counter_steer_magnitude"] = round(max(-0.85, min(0.0, float(p["counter_steer_magnitude"]))), 3)
    return p


def make_candidates(base):
    candidates = []

    configs = [
        ("base", 0.00, 0.00, 0.00, "medium"),
        ("more_spin", 0.08, 0.20, 0.10, "aggressive"),
        ("strong_spin", 0.14, 0.35, 0.18, "aggressive"),
        ("less_spin", -0.07, -0.10, -0.06, "mild"),
        ("less_departure", -0.10, -0.20, 0.08, "mild"),
        ("more_departure", 0.10, 0.25, 0.05, "medium"),
    ]

    for name, ds, dd, db, mode in configs:
        p = base.copy()
        p["candidate_name"] = name
        p["steer_magnitude"] += ds
        p["steer_duration"] += dd
        p["brake_factor"] += db
        p["counter_steer_magnitude"] -= ds * 0.5
        p["spin_mode"] = mode
        p["source"] = "candidate_search"
        candidates.append(clamp(p))

    return candidates


def run_carla(param_path, traj_out, outcome_out):
    cmd = [
        "python3",
        "scripts/04_run_carla_single.py",
        "--params", param_path,
        "--traj_out", traj_out,
        "--outcome_out", outcome_out,
        "--duration", "12"
    ]
    subprocess.run(cmd, check=True)


def score_trajectory(gt_csv, pred_csv):
    gt = pd.read_csv(gt_csv)
    pred = pd.read_csv(pred_csv)

    n = min(len(gt), len(pred))
    gt = gt.iloc[:n]
    pred = pred.iloc[:n]

    xy = np.sqrt(
        (gt["x"].values - pred["x"].values) ** 2 +
        (gt["y"].values - pred["y"].values) ** 2
    )

    yaw = np.abs(gt["yaw_change"].values - pred["yaw_change"].values)

    mean_xy = float(np.mean(xy))
    mean_yaw = float(np.mean(yaw))

    score = mean_xy + 0.05 * mean_yaw

    return score, mean_xy, mean_yaw


def main():
    pred_files = sorted([
        f for f in os.listdir(LLM_PARAM_DIR)
        if f.endswith("_fewshot.json")
    ])

    final_rows = []

    for fname in pred_files:
        case_id = fname.replace("_fewshot.json", "")
        print(f"\n=== Candidate search for {case_id} ===")

        base_param_path = os.path.join(LLM_PARAM_DIR, fname)
        gt_traj = os.path.join(GT_TRAJ_DIR, f"{case_id}.csv")

        if not os.path.exists(gt_traj):
            print("Missing GT trajectory:", case_id)
            continue

        base = load_json(base_param_path)
        candidates = make_candidates(base)

        case_logs = []
        best = None

        for i, cand in enumerate(candidates):
            cand_id = f"{case_id}_cand{i}_{cand['candidate_name']}"

            param_path = os.path.join(SEARCH_PARAM_DIR, f"{cand_id}.json")
            traj_path = os.path.join(SEARCH_TRAJ_DIR, f"{cand_id}.csv")
            outcome_path = os.path.join(SEARCH_OUTCOME_DIR, f"{cand_id}_outcome.json")

            save_json(cand, param_path)
            run_carla(param_path, traj_path, outcome_path)

            score, mean_xy, mean_yaw = score_trajectory(gt_traj, traj_path)

            row = {
                "case_id": case_id,
                "candidate_id": cand_id,
                "candidate_name": cand["candidate_name"],
                "score": score,
                "mean_xy_error": mean_xy,
                "mean_yaw_error": mean_yaw,
                "steer_magnitude": cand["steer_magnitude"],
                "steer_duration": cand["steer_duration"],
                "brake_factor": cand["brake_factor"],
                "spin_mode": cand["spin_mode"],
                "param_path": param_path,
                "traj_path": traj_path,
                "outcome_path": outcome_path
            }

            case_logs.append(row)

            if best is None or score < best["score"]:
                best = row

        log_df = pd.DataFrame(case_logs)
        log_df.to_csv(os.path.join(SEARCH_LOG_DIR, f"{case_id}_candidate_log.csv"), index=False)

        best_param = load_json(best["param_path"])
        final_param_path = os.path.join(SEARCH_PARAM_DIR, f"{case_id}_search.json")
        final_traj_path = os.path.join(SEARCH_TRAJ_DIR, f"{case_id}.csv")
        final_outcome_path = os.path.join(SEARCH_OUTCOME_DIR, f"{case_id}_outcome.json")

        save_json(best_param, final_param_path)

        # copy best trajectory/outcome into final standardized names
        pd.read_csv(best["traj_path"]).to_csv(final_traj_path, index=False)
        save_json(load_json(best["outcome_path"]), final_outcome_path)

        final_rows.append(best)

        print("Best:", best["candidate_name"], "score:", round(best["score"], 3))

    pd.DataFrame(final_rows).to_csv(
        os.path.join(SEARCH_LOG_DIR, "best_candidate_summary.csv"),
        index=False
    )

    print("\nCandidate search complete.")


if __name__ == "__main__":
    main()