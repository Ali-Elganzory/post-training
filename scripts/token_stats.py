# merge dataset subsets into a single mix
import yaml

with open("data-selection-datasets.yaml", "r") as f:
    datasets = yaml.load(f, Loader=yaml.FullLoader)

merged = {}
for dataset in datasets:
    path = dataset["datasets"][0]["path"]
    if path not in merged:
        merged[path] = dataset
    else:
        merged[path]["datasets"].extend(dataset["datasets"])

with open("data-selection-datasets-merged.yaml", "w") as f:
    yaml.dump(merged, f)

print(f"Merged {len(datasets)} datasets into {len(merged)} datasets")

import sys

sys.path.append("../src")
from post_training.config import DataConfig, DatasetEntry, PostTrainingConfig
from post_training.data.loader import load_and_mix_datasets
from post_training.data.utils import count_tokens
from post_training.methods.common import build_tokenizer
from post_training.methods.sft import MESSAGES_FEATURES, _sft_row_filter

overall_stats = {}

tokenizer = build_tokenizer(PostTrainingConfig.load("../configs/sft.yaml"))

with open("token-stats.yaml", "r") as f:
    overall_stats = yaml.load(f, Loader=yaml.FullLoader) or {}

for path, dataset_list in merged.items():
    print(f"Processing (dataset) {path}")
    dataset_list = dataset_list["datasets"]
    if path not in overall_stats:
        overall_stats[path] = {}
    for subset in dataset_list:
        subset_name = subset.get("subset", "default")
        actual_subset = subset.get("subset", None)
        print(f"Processing (subset) {subset_name}")
        if path not in overall_stats or subset_name not in overall_stats[path]:
            overall_stats[path][subset_name] = {}
        for split in subset["split"].split("+"):
            print(f"Processing (split) {split}")
            if (
                path in overall_stats
                and subset_name in overall_stats[path]
                and split in overall_stats[path][subset_name]
            ):
                print(f"Skipping (split) {split} because it already exists")
                continue
            config = DataConfig(
                chat_template="olmo3",
                datasets=[
                    DatasetEntry(
                        name=subset["name"],
                        path=subset["path"],
                        data_dir=subset.get("data_dir", None),
                        subset=actual_subset,
                        split=split,
                        weight=subset["weight"],
                        transform=subset.get("transform", None),
                    )
                ],
                num_proc=64,
                seed=42,
            )
            dataset = load_and_mix_datasets(
                config,
                row_filter=_sft_row_filter,
                columns_to_keep=["messages"],
                features=MESSAGES_FEATURES,
            )
            print(f"Loaded {len(dataset)} rows")
            if len(dataset) == 0:
                print(f"No rows found for {path} {subset_name} {split}")
                stats = {}
            else:
                stats = count_tokens(
                    processing_class=tokenizer,
                    dataset=dataset,
                )
            overall_stats[path][subset.get("subset", "default")][split] = stats
            print("Token stats:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
            print()

            with open("token-stats.yaml", "w") as f:
                yaml.dump(overall_stats, f)
