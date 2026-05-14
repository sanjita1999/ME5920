import os
import json
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_JSONL = os.path.join(BASE_DIR, "outputs", "llm_dataset", "train.jsonl")
TEST_JSONL = os.path.join(BASE_DIR, "outputs", "llm_dataset", "test.jsonl")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "llm_predictions_fewshot")

os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def read_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_msg(item, role):
    for m in item["messages"]:
        if m["role"] == role:
            return m["content"]
    return ""


def extract_json(text):
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found.")
    return json.loads(match.group(0))


def clamp(pred):
    pred["ego_speed_kmh"] = max(35, min(105, float(pred.get("ego_speed_kmh", 65))))
    pred["steer_trigger_time"] = max(1.0, min(5.0, float(pred.get("steer_trigger_time", 3))))
    pred["steer_magnitude"] = max(0.05, min(0.85, float(pred.get("steer_magnitude", 0.35))))
    pred["steer_duration"] = max(0.5, min(3.0, float(pred.get("steer_duration", 1.3))))
    pred["counter_steer_magnitude"] = max(-0.85, min(0.0, float(pred.get("counter_steer_magnitude", -0.25))))
    pred["brake_factor"] = max(0.0, min(1.0, float(pred.get("brake_factor", 0.25))))
    pred["friction_scale"] = max(0.35, min(1.0, float(pred.get("friction_scale", 0.8))))
    pred["target_yaw_change"] = max(30, min(240, float(pred.get("target_yaw_change", 120))))

    if pred.get("lane_departure_side") not in ["left", "right"]:
        pred["lane_departure_side"] = "right"

    pred["rollover_label"] = bool(pred.get("rollover_label", True))
    return pred


def build_prompt(train_examples, test_narrative):
    prompt = """You convert crash narratives into CARLA simulation parameter JSON.

Return ONLY JSON with keys:
ego_speed_kmh, steer_trigger_time, steer_magnitude, steer_duration,
counter_steer_magnitude, brake_factor, friction_scale,
lane_departure_side, target_yaw_change, rollover_label.

Examples:
"""

    for ex in train_examples:
        prompt += "\nNarrative:\n"
        prompt += get_msg(ex, "user")[:900] + "\n"
        prompt += "JSON:\n"
        prompt += get_msg(ex, "assistant") + "\n"

    prompt += "\nNarrative:\n"
    prompt += test_narrative[:1200] + "\n"
    prompt += "JSON:\n"

    return prompt


def main():
    train = read_jsonl(TRAIN_JSONL)
    test = read_jsonl(TEST_JSONL)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    fewshot = train[:2]

    rows = []

    for item in test:
        case_id = item["case_id"]
        narrative = get_msg(item, "user")

        prompt = build_prompt(fewshot, narrative)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        gen = text[len(prompt):]

        with open(os.path.join(OUT_DIR, f"{case_id}_raw.txt"), "w") as f:
            f.write(gen)

        try:
            pred = extract_json(gen)
            pred = clamp(pred)

        except Exception:
            pred = {
                "ego_speed_kmh": 65,
                "steer_trigger_time": 3.0,
                "steer_magnitude": 0.35,
                "steer_duration": 1.3,
                "counter_steer_magnitude": -0.23,
                "brake_factor": 0.25,
                "friction_scale": 0.8,
                "lane_departure_side": "right",
                "target_yaw_change": 120,
                "rollover_label": True
            }

        pred["case_id"] = case_id
        pred["split"] = "test"
        pred["town"] = "Town04"
        pred["vehicle_model"] = "vehicle.tesla.model3"
        pred["weather"] = "ClearNoon"
        pred["source"] = "hf_fewshot_tinyllama"
        pred["spin_mode"] = "medium"

        out_path = os.path.join(OUT_DIR, f"{case_id}_fewshot.json")
        with open(out_path, "w") as f:
            json.dump(pred, f, indent=4)

        rows.append(pred)
        print("Saved:", case_id)

    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_DIR, "fewshot_predictions_summary.csv"),
        index=False
    )


if __name__ == "__main__":
    main()