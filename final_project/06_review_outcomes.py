import os
import glob
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTCOME_DIR = os.path.join(BASE_DIR, "outputs", "outcomes")
PARAM_DIR = os.path.join(BASE_DIR, "outputs", "carla_params")
REVIEW_DIR = os.path.join(BASE_DIR, "outputs", "review")

os.makedirs(REVIEW_DIR, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def classify_case(outcome):
    lane_ok = outcome.get("lane_departure_success", False)
    roadside_ok = outcome.get("roadside_contact_success", False)

    yaw = float(outcome.get("max_yaw_change", 0))
    target_yaw = float(outcome.get("target_yaw_change", 90))
    lateral = float(outcome.get("max_lateral_offset", 0))

    yaw_ratio = yaw / max(target_yaw, 1)

    if not lane_ok:
        return "FAILED_DEPARTURE", "Vehicle did not leave lane enough."

    if lane_ok and not roadside_ok:
        return "WEAK_DEPARTURE", "Vehicle left lane but did not reach roadside/contact region."

    if lateral > 35:
        return "TOO_MUCH_DEPARTURE", "Vehicle departed too far from roadway."

    if yaw_ratio < 0.60:
        return "NEEDS_MORE_SPIN", "Yaw/spin is much lower than target."

    if yaw_ratio > 1.60:
        return "TOO_MUCH_SPIN", "Yaw/spin is much higher than target."

    return "GOOD", "Trajectory matches expected departure/spin behavior."


def suggest_tweak(label, params):
    new_params = params.copy()

    if label == "FAILED_DEPARTURE":
        new_params["steer_magnitude"] += 0.10
        new_params["steer_duration"] += 0.30
        new_params["brake_factor"] += 0.05

    elif label == "WEAK_DEPARTURE":
        new_params["steer_magnitude"] += 0.06
        new_params["steer_duration"] += 0.20

    elif label == "NEEDS_MORE_SPIN":
        new_params["steer_magnitude"] += 0.08
        new_params["steer_duration"] += 0.20
        new_params["brake_factor"] += 0.10

    elif label == "TOO_MUCH_DEPARTURE":
        new_params["steer_magnitude"] -= 0.08
        new_params["steer_duration"] -= 0.20
        new_params["brake_factor"] += 0.10

    elif label == "TOO_MUCH_SPIN":
        new_params["steer_magnitude"] -= 0.06
        new_params["brake_factor"] -= 0.05

    new_params["steer_magnitude"] = round(max(0.05, min(0.85, new_params["steer_magnitude"])), 3)
    new_params["steer_duration"] = round(max(0.5, min(3.0, new_params["steer_duration"])), 2)
    new_params["brake_factor"] = round(max(0.0, min(1.0, new_params["brake_factor"])), 3)

    return new_params


def main():
    outcome_files = sorted(glob.glob(os.path.join(OUTCOME_DIR, "*_outcome.json")))

    rows = []

    for outcome_path in outcome_files:
        outcome = load_json(outcome_path)
        case_id = outcome["case_id"]

        param_path = os.path.join(PARAM_DIR, f"{case_id}_carla.json")

        if not os.path.exists(param_path):
            continue

        params = load_json(param_path)

        label, reason = classify_case(outcome)
        suggested = suggest_tweak(label, params)

        rows.append({
            "case_id": case_id,
            "label": label,
            "reason": reason,
            "lane_departure_success": outcome.get("lane_departure_success"),
            "roadside_contact_success": outcome.get("roadside_contact_success"),
            "max_yaw_change": outcome.get("max_yaw_change"),
            "target_yaw_change": outcome.get("target_yaw_change"),
            "yaw_ratio": round(
                float(outcome.get("max_yaw_change", 0)) /
                max(float(outcome.get("target_yaw_change", 1)), 1),
                3
            ),
            "max_lateral_offset": outcome.get("max_lateral_offset"),
            "old_steer_magnitude": params["steer_magnitude"],
            "new_steer_magnitude": suggested["steer_magnitude"],
            "old_steer_duration": params["steer_duration"],
            "new_steer_duration": suggested["steer_duration"],
            "old_brake_factor": params["brake_factor"],
            "new_brake_factor": suggested["brake_factor"],
        })

        suggested_path = os.path.join(REVIEW_DIR, f"{case_id}_suggested.json")

        with open(suggested_path, "w") as f:
            json.dump(suggested, f, indent=4)

    df = pd.DataFrame(rows)

    review_csv = os.path.join(REVIEW_DIR, "review_summary.csv")
    df.to_csv(review_csv, index=False)

    print("\nReview saved to:", review_csv)

    if len(df) > 0:
        print("\nLabel counts:")
        print(df["label"].value_counts())

        print("\nPreview:")
        print(df[[
            "case_id",
            "label",
            "max_yaw_change",
            "target_yaw_change",
            "yaw_ratio",
            "max_lateral_offset",
            "old_steer_magnitude",
            "new_steer_magnitude",
            "old_brake_factor",
            "new_brake_factor"
        ]].head(20))


if __name__ == "__main__":
    main()