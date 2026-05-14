import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SELECTED_CSV = os.path.join(BASE_DIR, "outputs", "selected_cases", "family_A_25_cases.csv")
PARAM_DIR = os.path.join(BASE_DIR, "outputs", "pseudo_gt", "params")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "llm_dataset")

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(SELECTED_CSV)

rows = []

for _, row in df.iterrows():
    case_id = str(row["CRASH_KEY"])
    param_path = os.path.join(PARAM_DIR, f"{case_id}_carla.json")

    if not os.path.exists(param_path):
        continue

    with open(param_path, "r") as f:
        params = json.load(f)

    target = {
        "ego_speed_kmh": params["ego_speed_kmh"],
        "steer_trigger_time": params["steer_trigger_time"],
        "steer_magnitude": params["steer_magnitude"],
        "steer_duration": params["steer_duration"],
        "counter_steer_magnitude": params["counter_steer_magnitude"],
        "brake_factor": params["brake_factor"],
        "friction_scale": params["friction_scale"],
        "lane_departure_side": params["lane_departure_side"],
        "target_yaw_change": params["target_yaw_change"],
        "rollover_label": params["rollover_label"]
    }

    item = {
        "case_id": case_id,
        "split": params["split"],
        "messages": [
            {
                "role": "system",
                "content": "You convert real crash narratives into CARLA simulation parameter JSON. Return only valid JSON."
            },
            {
                "role": "user",
                "content": str(row["NARRATIVE"])
            },
            {
                "role": "assistant",
                "content": json.dumps(target)
            }
        ]
    }

    rows.append(item)

train_path = os.path.join(OUT_DIR, "train.jsonl")
test_path = os.path.join(OUT_DIR, "test.jsonl")

with open(train_path, "w") as train_f, open(test_path, "w") as test_f:
    for item in rows:
        line = json.dumps(item) + "\n"
        if item["split"] == "test":
            test_f.write(line)
        else:
            train_f.write(line)

print(f"Total examples: {len(rows)}")
print(f"Saved train: {train_path}")
print(f"Saved test: {test_path}")