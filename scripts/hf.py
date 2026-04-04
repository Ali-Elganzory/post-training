import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        type=str,
        default="list-configs-and-splits",
        choices=["list-configs-and-splits"],
        help="Which utility to run",
    )
    parser.add_argument("--dataset-name", type=str, required=True)
    return parser.parse_args()


def list_configs_and_splits(dataset_name):
    from post_training.data.utils import get_configs_and_splits

    print(f"--- Dataset: {dataset_name} ---")
    configs_and_splits = get_configs_and_splits(dataset_name)
    for config, splits in configs_and_splits.items():
        print(f"Subset: {config}")
        for split, num_samples in splits:
            print(f"  └─ Split: {split} (samples: {num_samples})")


if __name__ == "__main__":
    args = parse_args()
    if args.action == "list-configs-and-splits":
        list_configs_and_splits(args.dataset_name)
    else:
        raise ValueError(f"Unknown action: {args.action}")
