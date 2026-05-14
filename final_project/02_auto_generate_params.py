import os
import json
import random
import pandas as pd

FAMILY_ID = "A"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTED_PATH = os.path.join(BASE_DIR, "outputs", "selected_cases", f"family_{FAMILY_ID}_25_cases.csv")
PARAM_DIR = os.path.join(BASE_DIR, "outputs", "params")

os.makedirs(PARAM_DIR, exist_ok=True)

random.seed(42)


def clamp(x, low, high):
    return max(low, min(high, x))


def contains(text, keywords):
    text = str(text).lower()
    return any(k.lower() in text for k in keywords)


def is_good_case(row):
    narrative = str(row.get("NARRATIVE", "")).lower()
    normalized = str(row.get("normalized_json", "")).lower()

    if "snowmobile" in narrative:
        return False

    if '"pedestrian_involved": true' in normalized:
        return False

    if "walker" in narrative or "passenger's legs" in narrative:
        return False

    try:
        if int(row.get("VEHICLES_family",1)) != 1:
            return False
    except:
        pass

    if "roll" not in narrative and "ditch" not in narrative and "embankment" not in narrative and "off ramp" not in narrative and "overcorrect" not in narrative:
        return False

    return True

def generate_params(row):
    narrative = str(row.get("NARRATIVE", ""))
    detected_rollover = str(row.get("detected_rollover", "False")).lower() == "true"
    detected_lane_departure = str(row.get("detected_lane_departure", "False")).lower() == "true"
    detected_fixed_object = str(row.get("detected_fixed_object", "False")).lower() == "true"

    injuries = 0 if pd.isna(row.get("INJURIES", 0)) else int(row.get("INJURIES", 0))
    major = 0 if pd.isna(row.get("MAJINJURY", 0)) else int(row.get("MAJINJURY", 0))

    ego_speed = random.uniform(58, 78)
    collision_angle = random.uniform(10, 18)
    lane_offset = random.uniform(0.55, 0.95)
    steer_noise = random.uniform(0.22, 0.38)
    brake_factor = random.uniform(0.10, 0.40)
    friction_scale = random.uniform(0.65, 0.90)

    if detected_rollover or contains(narrative, ["roll", "rolled", "rollover"]):
        ego_speed += 8
        collision_angle += 7
        steer_noise += 0.08
        friction_scale -= 0.08

    if detected_lane_departure or contains(narrative, ["ditch", "embankment", "median", "gore", "off road"]):
        lane_offset += 0.25
        steer_noise += 0.05

    if detected_fixed_object or contains(narrative, ["curb", "guardrail", "sign", "field drive"]):
        collision_angle += 4
        lane_offset += 0.10

    if contains(narrative, ["over-corrected", "overcorrected", "over corrected"]):
        steer_noise += 0.12
        collision_angle += 5

    if contains(narrative, ["ice", "snow"]):
        friction_scale -= 0.12
        steer_noise += 0.05

    if contains(narrative, ["airborne"]):
        collision_angle += 5
        ego_speed += 4

    if major > 0 or injuries > 0:
        ego_speed += 4

    params = {
        "case_id": str(row["CRASH_KEY"]),
        "split": str(row["split"]),
        "ego_speed": round(clamp(ego_speed, 45, 95), 2),
        "collision_angle": round(clamp(collision_angle, 5, 35), 2),
        "lane_offset": round(clamp(lane_offset, 0.3, 1.6), 2),
        "steer_noise": round(clamp(steer_noise, 0.05, 0.65), 2),
        "brake_factor": round(clamp(brake_factor, 0.0, 1.0), 2),
        "friction_scale": round(clamp(friction_scale, 0.40, 1.0), 2),
        "source_narrative": narrative
    }

    return params


def main():
    df = pd.read_csv(SELECTED_PATH)

    df = df[df.apply(is_good_case, axis=1)].copy()
    df = df.head(25)

    print(f"Physics-consistent usable cases: {len(df)}")

    summary_rows = []

    for _, row in df.iterrows():
        params = generate_params(row)
        case_id = params["case_id"]

        out_json = os.path.join(PARAM_DIR, f"{case_id}_auto.json")

        with open(out_json, "w") as f:
            json.dump(params, f, indent=4)

        summary_rows.append(params)

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(PARAM_DIR, f"family_{FAMILY_ID}_auto_params_summary.csv"),
        index=False
    )

    print("Auto params generated successfully.")


if __name__ == "__main__":
    main()