"""Datamodule factory aligned with ``examples/train_gaia_real_data.py``."""

from __future__ import annotations

import logging
from typing import Any, Optional

from torch_geometric.transforms import BaseTransform, Compose

from astro_lab.data.dataset.astrolab import AstroLabInMemoryDataset
from astro_lab.data.dataset.lightning import AstroLabDataModule
from astro_lab.data.samplers.neighbor import KNNSampler
from astro_lab.data.transforms import AstronomicalFeatures, CosmicWebClassification

logger = logging.getLogger(__name__)


class _IdentityTransform(BaseTransform):
    def forward(self, data):
        return data


_GRAPH_METHOD_TO_STRATEGY = {
    "knn": "knn",
    "radius": "radius",
    "neighbor": "neighbor",
    "adaptive": "adaptive",
}


def create_datamodule(
    survey: str = "gaia",
    task: str = "node_classification",
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    k_neighbors: int = 20,
    graph_method: str = "knn",
    astronomical_features: bool = True,
    cosmic_web_features: bool = False,
    multi_scale: bool = False,
    enable_dynamic_batching: bool = False,
    **kwargs: Any,
) -> AstroLabDataModule:
    """Build an :class:`AstroLabDataModule` for the given survey and task."""
    if multi_scale:
        logger.warning(
            "create_datamodule: multi_scale=True is not implemented; ignoring."
        )
    if kwargs:
        logger.debug("create_datamodule: unused kwargs dropped: %s", sorted(kwargs))

    sampling_strategy = _GRAPH_METHOD_TO_STRATEGY.get(
        graph_method.lower(), "knn"
    )
    if graph_method.lower() not in _GRAPH_METHOD_TO_STRATEGY:
        logger.warning(
            "Unknown graph_method %r; using sampling_strategy=%r",
            graph_method,
            sampling_strategy,
        )

    sampler_kwargs: dict[str, Any] = {"k": k_neighbors}

    if cosmic_web_features and astronomical_features:
        transform: BaseTransform | None = Compose(
            [AstronomicalFeatures(), CosmicWebClassification()]
        )
    elif cosmic_web_features:
        transform = CosmicWebClassification()
    elif astronomical_features:
        transform = AstronomicalFeatures()
    else:
        transform = _IdentityTransform()

    dataset = AstroLabInMemoryDataset(
        survey_name=survey,
        task=task,
        max_samples=max_samples,
        sampling_strategy=sampling_strategy,
        sampler_kwargs=sampler_kwargs,
        transform=transform,
        pre_transform=None,
    )

    if sampling_strategy == "knn":
        sampler = KNNSampler(k=k_neighbors)
    elif sampling_strategy == "neighbor":
        sampler = None
    else:
        sampler = dataset.sampler

    return AstroLabDataModule(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        enable_dynamic_batching=enable_dynamic_batching,
    )
