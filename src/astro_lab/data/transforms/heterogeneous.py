"""
Heterogeneous Graph Transforms
=============================

Transforms for building heterogeneous graphs from multi-survey data.
"""

import logging
from typing import Dict, List

from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import radius
from torch_geometric.transforms import BaseTransform

logger = logging.getLogger(__name__)


class MultiSurveyMerger(BaseTransform):
    """Merge multiple survey datasets into a heterogeneous graph."""

    def __init__(
        self,
        surveys: List[str],
        cross_match_radius: float = 1.0,
        coord_unit: str = "arcsec",
    ):
        """
        Initialize multi-survey merger.

        Args:
            surveys: List of survey names to merge
            cross_match_radius: Radius for cross-matching (in coord_unit)
            coord_unit: Unit for coordinates ('arcsec', 'deg', 'rad')
        """
        self.surveys = surveys
        self.cross_match_radius = cross_match_radius
        self.coord_unit = coord_unit

    def __call__(self, data_dict: Dict[str, Data]) -> HeteroData:
        """Merge multiple surveys into heterogeneous graph."""
        hetero_data = HeteroData()

        # Add nodes from each survey
        for survey in self.surveys:
            if survey not in data_dict:
                logger.warning(f"Survey {survey} not found in data")
                continue

            survey_data = data_dict[survey]

            # Copy node features
            hetero_data[survey].x = survey_data.x
            if hasattr(survey_data, "pos"):
                hetero_data[survey].pos = survey_data.pos
            if hasattr(survey_data, "y"):
                hetero_data[survey].y = survey_data.y

            # Copy intra-survey edges
            if hasattr(survey_data, "edge_index"):
                hetero_data[
                    survey, "within", survey
                ].edge_index = survey_data.edge_index

        # Add cross-survey matches based on positions
        self._add_cross_matches(hetero_data)

        logger.info(
            f"Created heterogeneous graph with {len(hetero_data.node_types)} surveys"
        )
        return hetero_data

    def _add_cross_matches(self, hetero_data: HeteroData):
        """Add cross-survey edges based on spatial proximity."""
        node_types = hetero_data.node_types

        for i, survey1 in enumerate(node_types):
            for survey2 in node_types[i + 1 :]:
                # Check if both have positions
                if not (
                    hasattr(hetero_data[survey1], "pos")
                    and hasattr(hetero_data[survey2], "pos")
                ):
                    continue

                pos1 = hetero_data[survey1].pos
                pos2 = hetero_data[survey2].pos

                # Find matches within radius using PyG's radius function
                # Note: radius returns edges from pos2 to pos1
                edge_index = radius(
                    pos1, pos2, r=self.cross_match_radius, max_num_neighbors=1
                )

                if edge_index.size(1) > 0:
                    # Add bidirectional edges
                    hetero_data[survey1, "matches", survey2].edge_index = edge_index
                    hetero_data[survey2, "matches", survey1].edge_index = edge_index[
                        [1, 0]
                    ]

                    logger.debug(
                        f"Found {edge_index.size(1)} matches between "
                        f"{survey1} and {survey2}"
                    )


class CrossMatchObjects(BaseTransform):
    """Cross-match objects between different catalogs based on position."""

    def __init__(
        self,
        primary_key: str,
        match_keys: List[str],
        radius: float = 1.0,
        max_matches: int = 1,
    ):
        """
        Initialize cross-matcher.

        Args:
            primary_key: Primary catalog key
            match_keys: Keys of catalogs to match against
            radius: Matching radius
            max_matches: Maximum matches per object
        """
        self.primary_key = primary_key
        self.match_keys = match_keys
        self.radius = radius
        self.max_matches = max_matches

    def __call__(self, data: HeteroData) -> HeteroData:
        """Apply cross-matching."""
        if self.primary_key not in data.node_types:
            logger.warning(f"Primary key {self.primary_key} not found")
            return data

        primary_pos = data[self.primary_key].pos

        for match_key in self.match_keys:
            if match_key not in data.node_types:
                continue

            match_pos = data[match_key].pos

            # Use PyG's radius function for matching
            edge_index = radius(
                match_pos,
                primary_pos,
                r=self.radius,
                max_num_neighbors=self.max_matches,
            )

            if edge_index.size(1) > 0:
                # edge_index is [target, source] so swap for intuitive direction
                data[self.primary_key, "matched_to", match_key].edge_index = edge_index[
                    [1, 0]
                ]

                logger.info(
                    f"Matched {edge_index.size(1)} objects from "
                    f"{self.primary_key} to {match_key}"
                )

        return data
