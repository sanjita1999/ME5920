import os
import json
import random
import pandas as pd

FAMILY_ID = "A"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTED_PATH = os.path.join(BASE_DIR, "outputs", "selected_cases", f"family_{FAMILY_ID}_25_cases.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "carla_params")

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(42)


def contains(text, words):
    text = str(text).lower()
    return any(w in text for w in words)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def is_good_case(row):
    narrative = str(row.get("NARRATIVE", "")).lower()
    normalized = str(row.get("normalized_json", "")).lower()

    if "snowmobile" in narrative:
        return False
    if "walker" in narrative or "passenger's legs" in narrative:
        return False
    if '"pedestrian_involved": true' in normalized:
        return False

    return (
        "roll" in narrative
        or "ditch" in narrative
        or "embankment" in narrative
        or "median" in narrative
        or "over correct" in narrative
        or "overcorrect" in narrative
        or "off ramp" in narrative
        or "gore" in narrative
    )


def generate_carla_params(row):
    narrative = str(row["NARRATIVE"])
    normalized = str(row.get("normalized_json", "")).lower()

    speed = random.uniform(55, 75)
    steer_mag = random.uniform(0.25, 0.45)
    steer_duration = random.uniform(0.8, 1.5)
    brake = random.uniform(0.05, 0.35)
    friction = random.uniform(0.65, 0.95)
    yaw_target = random.uniform(60, 130)

    if contains(narrative, ["high rate of speed", "excessive speed", "fled"]):
        speed += 15

    if contains(narrative, ["ice", "snow", "poor road"]):
        friction -= 0.20
        steer_mag += 0.08
        brake += 0.10

    if contains(narrative, ["over-corrected", "over corrected", "overcorrected"]):
        steer_mag += 0.12
        yaw_target += 40

    if contains(narrative, ["rolled multiple", "rolled several", "six times"]):
        yaw_target += 70
        steer_mag += 0.08

    if contains(narrative, ["ditch", "embankment", "median", "gore"]):
        steer_duration += 0.3

    if contains(narrative, ["right ditch", "right shoulder", "passenger side"]):
        side = "right"
    elif contains(narrative, ["left", "north ditch", "west ditch"]):
        side = "left"
    else:
        side = random.choice(["left", "right"])

    params = {
        "case_id": str(row["CRASH_KEY"]),
        "split": str(row["split"]),
        "town": "Town04",
        "vehicle_model": "vehicle.tesla.model3",
        "spin_mode": "medium",
        "ego_speed_kmh": round(clamp(speed, 35, 105), 2),
        "steer_trigger_time": round(random.uniform(2.0, 4.0), 2),
        "steer_magnitude": round(clamp(steer_mag, 0.05, 0.75), 3),
        "steer_duration": round(clamp(steer_duration, 0.5, 2.5), 2),
        "counter_steer_magnitude": round(-0.65 * steer_mag, 3),
        "brake_factor": round(clamp(brake, 0.0, 1.0), 3),
        "friction_scale": round(clamp(friction, 0.35, 1.0), 3),

        "lane_departure_side": side,
        "target_yaw_change": round(clamp(yaw_target, 30, 240), 2),
        "rollover_label": True,

        "weather": "WetCloudyNoon" if contains(narrative, ["ice", "snow", "poor road"]) else "ClearNoon",
        "source_narrative": narrative
    }

    return params


def main():
    df = pd.read_csv(SELECTED_PATH)
    df = df[df.apply(is_good_case, axis=1)].copy()

    print(f"CARLA usable cases: {len(df)}")

    rows = []

    for _, row in df.iterrows():
        params = generate_carla_params(row)
        case_id = params["case_id"]

        out_path = os.path.join(OUT_DIR, f"{case_id}_carla.json")

        with open(out_path, "w") as f:
            json.dump(params, f, indent=4)

        rows.append(params)

    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_DIR, "carla_params_summary.csv"),
        index=False
    )

    print(f"Saved CARLA params to {OUT_DIR}")


if __name__ == "__main__":
    main()