import logging
import os
import statistics
from typing import Any

from datasets import Dataset, concatenate_datasets
from datasets.inspect import get_dataset_config_names, get_dataset_split_names
from datasets.load import load_dataset

logger = logging.getLogger(__name__)


_MAX_NUM_PROC_IF_NOT_CONFIGURED = 64


def resolve_num_proc(configured: int | None) -> int:
    """Return the number of worker processes for ``.map()`` / ``.filter()``.

    When *configured* is ``None`` (the default), auto-detect from
    ``os.cpu_count()`` but cap at ``_MAX_NUM_PROC_IF_NOT_CONFIGURED`` to avoid process
    explosion on large HPC nodes.  An explicit value is still clamped to the
    available CPU count so we never request more workers than cores.
    """
    available = os.cpu_count() or 1
    if configured is not None:
        return min(configured, available)
    return min(available, _MAX_NUM_PROC_IF_NOT_CONFIGURED)


def resample_to_size(ds: Dataset, target_n: int, seed: int) -> Dataset:
    """Return a dataset of exactly ``target_n`` rows.

    When ``target_n`` is less than the dataset length, this performs a
    downsampling without replacement. When ``target_n`` is greater, it
    oversamples by concatenating full shuffled copies plus a final
    remainder slice.
    """
    n = len(ds)
    if target_n <= 0:
        raise ValueError("target_n must be positive in _resample_to_size()")

    ds_shuffled = ds.shuffle(seed=seed)

    if target_n <= n:
        return ds_shuffled.select(range(target_n))

    full_copies = target_n // n
    remainder = target_n % n

    copies: list[Dataset] = [ds_shuffled] * full_copies
    if remainder > 0:
        copies.append(ds_shuffled.select(range(remainder)))

    if len(copies) == 1:
        return copies[0]
    return concatenate_datasets(copies)


def count_tokens(
    processing_class: Any,
    dataset: Dataset,
) -> dict[str, int | float]:
    """Count the total number of tokens in a dataset.

    Returns
    -------
    dict[str, int | float]
        A dictionary with the following keys:
        - "total_tokens"
        - "avg_tokens"
        - "min_tokens"
        - "max_tokens"
        - "std_tokens"
    """
    # Tokenize dataset
    tokenized_ds = dataset.map(
        lambda x: processing_class.apply_chat_template(
            x["messages"],
            tokenize=True,
            add_generation_prompt=False,
            desc="Tokenizing dataset",
        ),
        num_proc=resolve_num_proc(None),
    )

    lengths = [len(x) for x in tokenized_ds["input_ids"]]
    total_tokens = sum(lengths)
    avg_tokens = total_tokens / len(lengths)
    min_tokens = min(lengths)
    max_tokens = max(lengths)
    std_tokens = statistics.stdev(lengths)

    stats = {
        "total_tokens": total_tokens,
        "avg_tokens": avg_tokens,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "std_tokens": std_tokens,
    }

    logger.info("Token statistics:")
    for k, v in stats.items():
        logger.info("  %s: %s", k, v)

    return stats


def get_configs_and_splits(dataset_name: str) -> dict[str, list[tuple[str, int]]]:
    """Get the available configs and splits for a dataset.

    Parameters
    ----------
    dataset_name: str
        The name of the dataset to get the configs and splits for.

    Returns
    -------
    dict[str, list[tuple[str, int]]]
        A dictionary with the available configs and splits for the dataset.
        The keys are the config names and the values are lists of tuples
        (split name, number of samples).
    """
    configs = get_dataset_config_names(dataset_name)
    return {
        config: [
            (split, len(load_dataset(dataset_name, config, split=split)))
            for split in get_dataset_split_names(dataset_name, config_name=config)
        ]
        for config in configs
    }
