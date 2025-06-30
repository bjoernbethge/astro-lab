"""
Graph Partitioning for Distributed Training
===========================================

Provides graph partitioning utilities for distributed GNN training.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import degree

logger = logging.getLogger(__name__)

# Optional imports for advanced partitioning
try:
    import torch_geometric.transforms as T
    from torch_geometric.partition import ClusterData
    HAS_CLUSTER = True
except ImportError:
    HAS_CLUSTER = False
    logger.warning("ClusterData not available. Install with: pip install torch-cluster")

try:
    import metis
    HAS_METIS = True
except ImportError:
    HAS_METIS = False
    logger.info("METIS not available. Using random partitioning as fallback.")


class GraphPartitioner:
    """
    Graph partitioning for distributed GNN training.
    
    Supports multiple partitioning methods:
    - Random partitioning
    - METIS partitioning (if available)
    - Cluster-based partitioning
    - Degree-based partitioning
    """
    
    def __init__(
        self,
        method: str = "random",
        num_partitions: int = 4,
        save_dir: Optional[Path] = None,
        balance_edges: bool = True,
        balance_nodes: bool = True,
    ):
        """
        Initialize graph partitioner.
        
        Args:
            method: Partitioning method ("random", "metis", "cluster", "degree")
            num_partitions: Number of partitions to create
            save_dir: Directory to save partition data
            balance_edges: Whether to balance edge distribution
            balance_nodes: Whether to balance node distribution
        """
        self.method = method
        self.num_partitions = num_partitions
        self.save_dir = save_dir
        self.balance_edges = balance_edges
        self.balance_nodes = balance_nodes
        
        # Validate method availability
        if method == "metis" and not HAS_METIS:
            logger.warning("METIS not available, falling back to random partitioning")
            self.method = "random"
        elif method == "cluster" and not HAS_CLUSTER:
            logger.warning("ClusterData not available, falling back to random partitioning")
            self.method = "random"
    
    def partition(self, data: Data) -> Dict[str, Union[Tensor, List[Data]]]:
        """
        Partition graph data.
        
        Args:
            data: PyG Data object to partition
            
        Returns:
            Dictionary containing:
            - partition_ids: Node partition assignments
            - partitions: List of Data objects for each partition
            - partition_stats: Statistics about partitions
        """
        if self.method == "random":
            return self._random_partition(data)
        elif self.method == "metis":
            return self._metis_partition(data)
        elif self.method == "cluster":
            return self._cluster_partition(data)
        elif self.method == "degree":
            return self._degree_partition(data)
        else:
            raise ValueError(f"Unknown partitioning method: {self.method}")
    
    def _random_partition(self, data: Data) -> Dict[str, Union[Tensor, List[Data]]]:
        """Random partitioning with optional balancing."""
        num_nodes = data.num_nodes
        
        if self.balance_nodes:
            # Balanced random assignment
            nodes_per_partition = num_nodes // self.num_partitions
            remainder = num_nodes % self.num_partitions
            
            partition_ids = torch.zeros(num_nodes, dtype=torch.long)
            perm = torch.randperm(num_nodes)
            
            start = 0
            for i in range(self.num_partitions):
                size = nodes_per_partition + (1 if i < remainder else 0)
                partition_ids[perm[start:start + size]] = i
                start += size
        else:
            # Purely random assignment
            partition_ids = torch.randint(0, self.num_partitions, (num_nodes,))
        
        return self._create_partitions(data, partition_ids)
    
    def _metis_partition(self, data: Data) -> Dict[str, Union[Tensor, List[Data]]]:
        """METIS-based graph partitioning."""
        if not HAS_METIS:
            return self._random_partition(data)
        
        # Convert to adjacency list format for METIS
        edge_index = data.edge_index.cpu()
        num_nodes = data.num_nodes
        
        # Create adjacency list
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src != dst:  # Skip self-loops
                adj_list[src].append(dst)
                adj_list[dst].append(src)
        
        # Remove duplicates
        adj_list = [list(set(neighbors)) for neighbors in adj_list]
        
        # Call METIS
        _, partition_ids = metis.part_graph(adj_list, self.num_partitions)
        partition_ids = torch.tensor(partition_ids, dtype=torch.long)
        
        return self._create_partitions(data, partition_ids)
    
    def _cluster_partition(self, data: Data) -> Dict[str, Union[Tensor, List[Data]]]:
        """Cluster-based partitioning using PyG ClusterData."""
        if not HAS_CLUSTER:
            return self._random_partition(data)
        
        # Use ClusterData for partitioning
        cluster_data = ClusterData(
            data,
            num_parts=self.num_partitions,
            recursive=False,
            save_dir=self.save_dir,
        )
        
        # Extract partition assignments
        partition_ids = cluster_data.partition.node_perm
        
        return self._create_partitions(data, partition_ids)
    
    def _degree_partition(self, data: Data) -> Dict[str, Union[Tensor, List[Data]]]:
        """Degree-based partitioning for load balancing."""
        # Calculate node degrees
        node_degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)
        
        # Sort nodes by degree
        sorted_indices = torch.argsort(node_degrees, descending=True)
        
        # Assign nodes to partitions in round-robin fashion
        partition_ids = torch.zeros(data.num_nodes, dtype=torch.long)
        for i, node in enumerate(sorted_indices):
            partition_ids[node] = i % self.num_partitions
        
        return self._create_partitions(data, partition_ids)
    
    def _create_partitions(
        self, data: Data, partition_ids: Tensor
    ) -> Dict[str, Union[Tensor, List[Data]]]:
        """Create partition Data objects from partition assignments."""
        partitions = []
        partition_stats = {
            "num_nodes": [],
            "num_edges": [],
            "cross_partition_edges": 0,
        }
        
        # Create subgraphs for each partition
        for pid in range(self.num_partitions):
            # Get nodes in this partition
            mask = partition_ids == pid
            nodes = torch.where(mask)[0]
            
            if len(nodes) == 0:
                logger.warning(f"Partition {pid} is empty!")
                continue
            
            # Create node mapping
            node_map = torch.full((data.num_nodes,), -1, dtype=torch.long)
            node_map[nodes] = torch.arange(len(nodes))
            
            # Extract edges within partition
            edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
            edge_index = data.edge_index[:, edge_mask]
            
            # Remap edge indices
            edge_index = node_map[edge_index]
            
            # Create partition data
            partition_data = Data(
                x=data.x[nodes] if data.x is not None else None,
                edge_index=edge_index,
                y=data.y[nodes] if hasattr(data, 'y') and data.y is not None else None,
                num_nodes=len(nodes),
            )
            
            # Copy other attributes if they exist
            for key, value in data:
                if key not in ['x', 'edge_index', 'y', 'num_nodes'] and value is not None:
                    if isinstance(value, Tensor) and value.size(0) == data.num_nodes:
                        setattr(partition_data, key, value[nodes])
            
            partitions.append(partition_data)
            
            # Update statistics
            partition_stats["num_nodes"].append(len(nodes))
            partition_stats["num_edges"].append(edge_index.size(1))
        
        # Calculate cross-partition edges
        for i in range(data.edge_index.size(1)):
            src, dst = data.edge_index[:, i]
            if partition_ids[src] != partition_ids[dst]:
                partition_stats["cross_partition_edges"] += 1
        
        # Log partition statistics
        logger.info(f"Partitioning statistics:")
        logger.info(f"  Method: {self.method}")
        logger.info(f"  Number of partitions: {len(partitions)}")
        logger.info(f"  Nodes per partition: {partition_stats['num_nodes']}")
        logger.info(f"  Edges per partition: {partition_stats['num_edges']}")
        logger.info(f"  Cross-partition edges: {partition_stats['cross_partition_edges']}")
        logger.info(f"  Edge cut ratio: {partition_stats['cross_partition_edges'] / data.edge_index.size(1):.2%}")
        
        return {
            "partition_ids": partition_ids,
            "partitions": partitions,
            "partition_stats": partition_stats,
        }
    
    def save_partitions(
        self, partitions: List[Data], partition_ids: Tensor
    ) -> None:
        """Save partitions to disk."""
        if self.save_dir is None:
            return
        
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save partition assignments
        torch.save(partition_ids, save_dir / "partition_ids.pt")
        
        # Save each partition
        for i, partition in enumerate(partitions):
            torch.save(partition, save_dir / f"partition_{i}.pt")
        
        logger.info(f"Saved {len(partitions)} partitions to {save_dir}")
    
    def load_partitions(self) -> Tuple[Tensor, List[Data]]:
        """Load partitions from disk."""
        if self.save_dir is None:
            raise ValueError("No save directory specified")
        
        save_dir = Path(self.save_dir)
        
        # Load partition assignments
        partition_ids = torch.load(save_dir / "partition_ids.pt")
        
        # Load partitions
        partitions = []
        for i in range(self.num_partitions):
            partition_path = save_dir / f"partition_{i}.pt"
            if partition_path.exists():
                partitions.append(torch.load(partition_path))
        
        logger.info(f"Loaded {len(partitions)} partitions from {save_dir}")
        
        return partition_ids, partitions


def partition_graph_for_distributed(
    data: Data,
    num_partitions: int,
    method: str = "auto",
    save_dir: Optional[str] = None,
) -> Dict[str, Union[Tensor, List[Data]]]:
    """
    Convenience function to partition a graph for distributed training.
    
    Args:
        data: PyG Data object
        num_partitions: Number of partitions
        method: Partitioning method or "auto" for automatic selection
        save_dir: Directory to save partitions
        
    Returns:
        Partition results dictionary
    """
    # Auto-select method based on graph size and availability
    if method == "auto":
        if data.num_nodes < 10000:
            method = "random"
        elif HAS_METIS and data.num_nodes < 1000000:
            method = "metis"
        elif HAS_CLUSTER:
            method = "cluster"
        else:
            method = "degree"
    
    # Create partitioner
    partitioner = GraphPartitioner(
        method=method,
        num_partitions=num_partitions,
        save_dir=Path(save_dir) if save_dir else None,
    )
    
    # Partition graph
    results = partitioner.partition(data)
    
    # Save if requested
    if save_dir:
        partitioner.save_partitions(
            results["partitions"], results["partition_ids"]
        )
    
    return results
