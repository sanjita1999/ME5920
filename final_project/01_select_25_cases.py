import pandas as pd
import os

FAMILY_ID = "A"   # change if needed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "selected_cases")

os.makedirs(OUT_DIR, exist_ok=True)

inventory_path = os.path.join(DATA_DIR, "crash_family_tagged_inventory.xlsx")
normalized_path = os.path.join(DATA_DIR, "oto_normalized.xlsx")

inventory = pd.read_excel(inventory_path)
normalized = pd.read_excel(normalized_path)

inventory["CRASH_KEY"] = inventory["CRASH_KEY"].astype(str)
normalized["CRASH_KEY"] = normalized["CRASH_KEY"].astype(str)

family_df = inventory[inventory["assigned_family_id"] == FAMILY_ID].copy()

merged = family_df.merge(
    normalized,
    on="CRASH_KEY",
    how="inner",
    suffixes=("_family", "_norm")
)

merged = merged.dropna(subset=["NARRATIVE"])
merged = merged[merged["NARRATIVE"].astype(str).str.len() > 20]

# if less than 25, take all available
if len(merged) < 25:
    selected = merged.copy()
else:
    selected = merged.head(25).copy()

# benchmark/test split
split_labels = ["benchmark"] * min(20, len(selected))
remaining = len(selected) - len(split_labels)
split_labels += ["test"] * remaining

selected["split"] = split_labels

out_path = os.path.join(OUT_DIR, f"family_{FAMILY_ID}_25_cases.csv")
selected.to_csv(out_path, index=False)

print(f"\nSelected {len(selected)} cases from family {FAMILY_ID}")
print(f"Saved to: {out_path}\n")

print(selected[[
    "CRASH_KEY",
    "assigned_family_id",
    "assigned_subfamily",
    "split",
    "NARRATIVE"
]].head())