"""Dataset loading, transformation, and mixing.

The main entry point is :func:`load_and_mix_datasets` which reads the
``data`` section of the config, loads each dataset, applies per-dataset
transforms and filters, rescales each dataset according to its weight,
then concatenates and shuffles the result.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from collections.abc import Callable

from datasets import Dataset, Features, concatenate_datasets, load_dataset

from post_training.data.transforms import get_transform
from post_training.data.utils import resample_to_size, resolve_num_proc

if TYPE_CHECKING:
    from post_training.config import DataConfig

logger = logging.getLogger(__name__)


def load_and_mix_datasets(
    config: DataConfig,
    row_filter: Callable[[dict], bool] | None = None,
    columns_to_keep: list[str] | None = None,
    features: Features | None = None,
) -> Dataset:
    """Load, transform, and optionally filter/mix datasets.

    Parameters
    ----------
    config:
        The ``data`` section of :class:`PostTrainingConfig`.
    row_filter:
        Optional predicate applied after transforms to exclude invalid
        rows.  Each training method passes its own filter (e.g. SFT
        checks ``messages``, DPO checks ``chosen`` / ``rejected``).
        When ``None``, no filtering is applied.
    columns_to_keep:
        Optional list of column names to retain in the final dataset(s).
        When a transform is applied, all original columns are dropped and only the
        transform outputs are kept. When no transform is applied, only the
        columns listed here (that are present) are retained.
    features:
        Optional features schema to enforce on the dataset.
        When a transform is applied, the features schema is enforced on the output.
        When no transform is applied, the features schema is enforced on the input.

    Returns
    -------
    datasets.Dataset
        A single concatenated and shuffled dataset ready for the trainer.
    """
    entries = config.datasets
    if not entries:
        raise ValueError("No datasets specified in data.datasets.")

    num_proc = resolve_num_proc(config.num_proc)
    logger.info("Dataset processing will use num_proc=%d", num_proc)

    seed = getattr(config, "seed", 42)

    loaded_datasets: list[Dataset] = []
    weights: list[float] = []

    for entry in entries:
        logger.info(
            "Loading dataset '%s' from '%s' (data_dir=%s, subset=%s, split=%s, "
            "weight=%s, transform=%s)",
            entry.name,
            entry.path,
            entry.data_dir,
            entry.subset,
            entry.split,
            entry.weight,
            entry.transform,
        )

        ds: Dataset = load_dataset(
            entry.path,
            data_dir=entry.data_dir,
            name=entry.subset,
            split=entry.split,
        )

        # Apply optional per-dataset transform.
        if entry.transform is not None:
            try:
                transform_fn = get_transform(entry.transform)
            except KeyError as exc:
                raise KeyError(
                    f"Unknown transform '{entry.transform}' for dataset '{entry.name}'. "
                    "Define it in 'post_training.data.transforms' using "
                    "@register_transform, or update your 'data.datasets[].transform' "
                    "value. Original error: "
                    f"{exc}"
                ) from exc

            logger.info(
                "Applying transform '%s' to dataset '%s'.",
                entry.transform,
                entry.name,
            )

            map_kwargs: dict = {"num_proc": num_proc}
            if columns_to_keep is not None:
                # Remove all columns except for the ones returned by the transform.
                map_kwargs["remove_columns"] = ds.column_names
            if features is not None:
                # Enforce features schema
                map_kwargs["features"] = features

            ds = ds.map(transform_fn, **map_kwargs)
        elif columns_to_keep is not None:
            # No transform → keep only the requested columns that are present.
            present = [c for c in columns_to_keep if c in ds.column_names]
            if not present:
                raise KeyError(
                    f"None of the expected columns {columns_to_keep} were found in "
                    f"dataset '{entry.name}'. Available columns: {ds.column_names}"
                )
            ds = ds.select_columns(present)

        if row_filter is not None:
            ds = ds.filter(row_filter, num_proc=num_proc)

        loaded_datasets.append(ds)
        weights.append(entry.weight)

    # Resample each dataset according to its weight and concatenate.
    resampled_datasets: list[Dataset] = []
    for idx, (ds, weight) in enumerate(zip(loaded_datasets, weights)):
        n = len(ds)
        target_n = int(round(weight * n))
        if target_n <= 0:
            continue

        ds_seed = seed + idx
        resampled = resample_to_size(ds, target_n, ds_seed)
        resampled_datasets.append(resampled)

    if not resampled_datasets:
        raise ValueError(
            "No rows left after applying data.datasets[].weight. Check your weights and filters."
        )

    if len(resampled_datasets) == 1:
        mixed = resampled_datasets[0]
    else:
        mixed = concatenate_datasets(resampled_datasets)

    mixed = mixed.shuffle(seed=seed)
    return mixed
