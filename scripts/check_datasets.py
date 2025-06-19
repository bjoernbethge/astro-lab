import torch
from astro_lab.data.datasets import GaiaGraphDataset, NSAGraphDataset, ExoplanetGraphDataset

print("🚀 Loading all processed datasets...")

print("\n=== GAIA DATASET ===")
gaia = GaiaGraphDataset()
g = gaia[0]
print(f"Nodes: {g.num_nodes:,}")
print(f"Edges: {g.num_edges:,}")
print(f"Node features shape: {g.x.shape}")
print(f"Edge features: {g.edge_attr.shape if hasattr(g, 'edge_attr') and g.edge_attr is not None else 'None'}")
print("🎯 EXACT Features (5): ra, dec, phot_g_mean_mag, bp_rp, parallax")

print("\n=== NSA DATASET (FULL) ===")
nsa = NSAGraphDataset(max_galaxies=150000)
n = nsa[0]
print(f"Nodes: {n.num_nodes:,}")
print(f"Edges: {n.num_edges:,}")
print(f"Node features shape: {n.x.shape}")
print(f"Edge features: {n.edge_attr.shape if hasattr(n, 'edge_attr') and n.edge_attr is not None else 'None'}")
print("🎯 EXACT Features (7): RA, DEC, ZDIST, PETROMAG_R, MASS, x_3d, y_3d, z_3d")
print(f"Has labels: {hasattr(n, 'y') and n.y is not None}")

print("\n=== EXOPLANET DATASET ===")
exo = ExoplanetGraphDataset()
e = exo[0]
print(f"Nodes: {e.num_nodes:,}")
print(f"Edges: {e.num_edges:,}")
print(f"Node features shape: {e.x.shape}")
print(f"Edge features: {e.edge_attr.shape if hasattr(e, 'edge_attr') and e.edge_attr is not None else 'None'}")
print("🎯 EXACT Features (9): pl_name, hostname, ra, dec, sy_dist, discoverymethod, disc_year, pl_rade, pl_masse")

print("\n📊 SUMMARY:")
total_nodes = g.num_nodes + n.num_nodes + e.num_nodes
total_edges = g.num_edges + n.num_edges + e.num_edges
print(f"Total astronomical objects: {total_nodes:,}")
print(f"Total connections: {total_edges:,}")
print(f"Average connections per object: {total_edges / total_nodes:.1f}")

print("\n🌟 DATASET BREAKDOWN:")
print(f"🌟 Gaia Stars: {g.num_nodes:,} ({g.num_nodes/total_nodes*100:.1f}%)")
print(f"🌌 NSA Galaxies: {n.num_nodes:,} ({n.num_nodes/total_nodes*100:.1f}%)")
print(f"🪐 Exoplanets: {e.num_nodes:,} ({e.num_nodes/total_nodes*100:.1f}%)")

print("\n🔬 TECHNICAL DETAILS:")
print(f"Gaia: k={gaia.k_neighbors} neighbors, magnitude_limit={gaia.magnitude_limit}")
print(f"NSA: k={nsa.k_neighbors} neighbors, distance_threshold={nsa.distance_threshold} Mpc")
print(f"Exoplanets: k={exo.k_neighbors} neighbors, max_distance={exo.max_distance} parsecs") 