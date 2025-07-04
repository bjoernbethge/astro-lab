"""
Open3D Mixin for Point Cloud Operations
=======================================

Provides point cloud processing capabilities using Open3D.
"""

import numpy as np
import open3d as o3d
import torch


class Open3DMixin:
    """
    Mixin providing Open3D point cloud and geometric operations.
    Requires self.coordinates as a [N, 3] torch.Tensor.
    """

    def to_point_cloud(self):
        """Convert coordinates to Open3D PointCloud object."""
        coords = self.coordinates.detach().cpu().numpy()  # type: ignore
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        return pcd

    def voxel_downsampling(self, voxel_size: float):
        """Return a new instance with voxel-downsampled points."""
        pcd = self.to_point_cloud()
        down = pcd.voxel_down_sample(voxel_size)
        coords = np.asarray(down.points)
        new_obj = self.__class__(coordinates=torch.from_numpy(coords).float())  # type: ignore
        return new_obj

    def statistical_outlier_removal(
        self, nb_neighbors: int = 20, std_ratio: float = 2.0
    ):
        """Return a new instance with statistical outliers removed."""
        pcd = self.to_point_cloud()
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        coords = np.asarray(pcd.points)[ind]
        new_obj = self.__class__(coordinates=torch.from_numpy(coords).float())  # type: ignore
        return new_obj

    def estimate_normals(self, radius: float = 1.0, max_nn: int = 30):
        """Estimate normals for the point cloud and return as numpy array."""
        pcd = self.to_point_cloud()
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
        return np.asarray(pcd.normals)

    def radius_neighbors(self, radius: float = 1.0):
        """Return indices of neighbors within a given radius for each point."""
        pcd = self.to_point_cloud()
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        coords = np.asarray(pcd.points)
        neighbors = []
        for i in range(len(coords)):
            _, idx, _ = kdtree.search_radius_vector_3d(coords[i], radius)
            neighbors.append(idx)
        return neighbors
