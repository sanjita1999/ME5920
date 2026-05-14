import os
import json
import subprocess
import random
import shutil
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GT_TRAJ_DIR = os.path.join(BASE_DIR, "outputs", "pseudo_gt", "trajectories")
GT_OUTCOME_DIR = os.path.join(BASE_DIR, "outputs", "pseudo_gt", "outcomes")
LLM_PARAM_DIR = os.path.join(BASE_DIR, "outputs", "llm_predictions_fewshot")

RL_PARAM_DIR = os.path.join(BASE_DIR, "outputs", "llm_predictions_rl")
RL_TRAJ_DIR = os.path.join(BASE_DIR, "outputs", "llm_trajectories_rl")
RL_OUTCOME_DIR = os.path.join(BASE_DIR, "outputs", "llm_outcomes_rl")
RL_LOG_DIR = os.path.join(BASE_DIR, "outputs", "rl_logs")

for d in [RL_PARAM_DIR, RL_TRAJ_DIR, RL_OUTCOME_DIR, RL_LOG_DIR]:
    os.makedirs(d, exist_ok=True)


TUNE_KEYS = [
    "steer_magnitude",
    "steer_duration",
    "brake_factor",
    "counter_steer_magnitude",
]


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


def mutate(base, scale, rng):
    p = base.copy()

    p["steer_magnitude"] += rng.uniform(-0.12, 0.16) * scale
    p["steer_duration"] += rng.uniform(-0.25, 0.35) * scale
    p["brake_factor"] += rng.uniform(-0.15, 0.18) * scale
    p["counter_steer_magnitude"] += rng.uniform(-0.10, 0.10) * scale

    mode = rng.choice(["mild", "medium", "aggressive"])
    p["spin_mode"] = mode

    p["source"] = "llm_carla_rl_search"
    return clamp(p)


def run_carla(param_path, traj_path, outcome_path):
    cmd = [
        "python3",
        "scripts/04_run_carla_single.py",
        "--params",
        param_path,
        "--traj_out",
        traj_path,
        "--outcome_out",
        outcome_path,
        "--duration",
        "12",
    ]
    subprocess.run(cmd, check=True)


def trajectory_score(gt_csv, pred_csv, gt_outcome_path, pred_outcome_path):
    gt = pd.read_csv(gt_csv)
    pred = pd.read_csv(pred_csv)

    n = min(len(gt), len(pred))
    gt = gt.iloc[:n]
    pred = pred.iloc[:n]

    xy_err = np.sqrt(
        (gt["x"].values - pred["x"].values) ** 2
        + (gt["y"].values - pred["y"].values) ** 2
    )

    yaw_err = np.abs(gt["yaw_change"].values - pred["yaw_change"].values)

    mean_xy = float(np.mean(xy_err))
    mean_yaw = float(np.mean(yaw_err))

    gt_outcome = load_json(gt_outcome_path)
    pred_outcome = load_json(pred_outcome_path)

    lane_penalty = 0 if gt_outcome.get("lane_departure_success") == pred_outcome.get("lane_departure_success") else 20
    roadside_penalty = 0 if gt_outcome.get("roadside_contact_success") == pred_outcome.get("roadside_contact_success") else 20

    score = mean_xy + 0.05 * mean_yaw + lane_penalty + roadside_penalty
    reward = -score

    return {
        "score": score,
        "reward": reward,
        "mean_xy_error": mean_xy,
        "mean_yaw_error": mean_yaw,
        "lane_match": gt_outcome.get("lane_departure_success") == pred_outcome.get("lane_departure_success"),
        "roadside_match": gt_outcome.get("roadside_contact_success") == pred_outcome.get("roadside_contact_success"),
        "yaw_change_error": abs(float(gt_outcome.get("max_yaw_change", 0)) - float(pred_outcome.get("max_yaw_change", 0))),
        "lateral_offset_error": abs(float(gt_outcome.get("max_lateral_offset", 0)) - float(pred_outcome.get("max_lateral_offset", 0))),
    }


def main():
    pred_files = sorted([
        f for f in os.listdir(LLM_PARAM_DIR)
        if f.endswith("_fewshot.json")
    ])

    all_best = []

    # Keep small for class-project runtime
    episodes = 8
    candidates_per_episode = 3

    for fname in pred_files:
        case_id = fname.replace("_fewshot.json", "")
        print(f"\n===== RL search for {case_id} =====")

        gt_traj = os.path.join(GT_TRAJ_DIR, f"{case_id}.csv")
        gt_outcome = os.path.join(GT_OUTCOME_DIR, f"{case_id}_outcome.json")
        base_param_path = os.path.join(LLM_PARAM_DIR, fname)

        if not os.path.exists(gt_traj) or not os.path.exists(gt_outcome):
            print("Missing GT files for", case_id)
            continue

        base = load_json(base_param_path)
        base["spin_mode"] = base.get("spin_mode", "medium")

        rng = random.Random(int(case_id[-5:]))

        best_param = clamp(base.copy())
        best_record = None
        logs = []

        scale = 1.0

        for ep in range(episodes):
            print(f"Episode {ep + 1}/{episodes}")

            candidates = []

            if ep == 0:
                candidates.append(best_param.copy())

            for _ in range(candidates_per_episode):
                candidates.append(mutate(best_param, scale, rng))

            for ci, cand in enumerate(candidates):
                run_id = f"{case_id}_ep{ep:02d}_cand{ci:02d}"

                param_path = os.path.join(RL_PARAM_DIR, f"{run_id}.json")
                traj_path = os.path.join(RL_TRAJ_DIR, f"{run_id}.csv")
                outcome_path = os.path.join(RL_OUTCOME_DIR, f"{run_id}_outcome.json")

                cand["case_id"] = case_id
                save_json(cand, param_path)

                run_carla(param_path, traj_path, outcome_path)

                result = trajectory_score(gt_traj, traj_path, gt_outcome, outcome_path)

                record = {
                    "case_id": case_id,
                    "run_id": run_id,
                    "episode": ep,
                    "candidate": ci,
                    "param_path": param_path,
                    "traj_path": traj_path,
                    "outcome_path": outcome_path,
                    "spin_mode": cand.get("spin_mode", "medium"),
                    "steer_magnitude": cand["steer_magnitude"],
                    "steer_duration": cand["steer_duration"],
                    "brake_factor": cand["brake_factor"],
                    "counter_steer_magnitude": cand["counter_steer_magnitude"],
                    **result,
                }

                logs.append(record)

                if best_record is None or record["reward"] > best_record["reward"]:
                    best_record = record
                    best_param = cand.copy()
                    print(
                        "  New best:",
                        run_id,
                        "score=",
                        round(record["score"], 3),
                        "xy=",
                        round(record["mean_xy_error"], 3),
                        "yaw=",
                        round(record["mean_yaw_error"], 3),
                    )

            scale *= 0.75

        log_df = pd.DataFrame(logs)
        log_df.to_csv(os.path.join(RL_LOG_DIR, f"{case_id}_rl_log.csv"), index=False)

        final_param_path = os.path.join(RL_PARAM_DIR, f"{case_id}_rl.json")
        final_traj_path = os.path.join(RL_TRAJ_DIR, f"{case_id}.csv")
        final_outcome_path = os.path.join(RL_OUTCOME_DIR, f"{case_id}_outcome.json")

        save_json(best_param, final_param_path)
        shutil.copyfile(best_record["traj_path"], final_traj_path)
        shutil.copyfile(best_record["outcome_path"], final_outcome_path)

        all_best.append(best_record)

        print("BEST FINAL:", case_id, "score=", round(best_record["score"], 3))

    pd.DataFrame(all_best).to_csv(
        os.path.join(RL_LOG_DIR, "rl_best_summary.csv"),
        index=False,
    )

    print("\nRL search complete.")
    print("Saved:", os.path.join(RL_LOG_DIR, "rl_best_summary.csv"))


if __name__ == "__main__":
    main()