# Install dependencies if needed:
# pip install data huggingface_hub

import os
import json
from data import Dataset
from huggingface_hub import snapshot_download

# 1) Download the dataset repository locally
#    This will clone all files (e.g., raw.json) into the specified cache_dir
repo_local_dir = snapshot_download(
    repo_id="searle-j/kote",
    repo_type="dataset",
    cache_dir="data/kote"
)
print(f"Dataset files downloaded to: {repo_local_dir}")

# 2) Construct path to raw.json and load the data
json_path = os.path.join(repo_local_dir, "raw.json")
with open(json_path, encoding="utf-8") as f:
    raw_map = json.load(f)  # raw_map: dict of {id: {"text":..., "labels":{...}}}

# 3) Convert dict-of-dicts into a list of examples
records = []
for _id, obj in raw_map.items():
    text = obj.get("text", "")
    # Merge labels from all raters into a single deduplicated list
    label_set = set()
    for rater_labels in obj.get("labels", {}).values():
        label_set.update(rater_labels)
    records.append({
        "id": _id,
        "text": text,
        "labels": list(label_set)
    })

# 4) Create a Hugging Face Dataset and split into train/validation
dataset = Dataset.from_list(records)
splits = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = splits["train"]
val_ds   = splits["test"]

# 5) Inspect sizes
print(f"Train examples: {len(train_ds)}")
print(f"Validation examples: {len(val_ds)}")

# 6) Show a few example records from training set
print("\nSample training examples:")
for i in range(min(5, len(train_ds))):
    sample = train_ds[i]
    print(f"\nExample {i + 1} (ID: {sample['id']}):")
    print(f"Text: {sample['text']}")
    print(f"Labels: {sample['labels']}")
