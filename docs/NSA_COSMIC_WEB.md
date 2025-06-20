# 🌌 NSA Cosmic Web Analysis

Complete 3D cosmic web analysis of NASA-Sloan Atlas galaxies using advanced spatial clustering techniques.

## Overview
The NSA cosmic web analysis processes ~640,000 galaxies with redshift z < 0.15, covering comoving distances up to 640 Mpc. This represents the most comprehensive mapping of large-scale cosmic structure using extragalactic data.

## Usage

### CLI Processing
\\\ash
# Process NSA data with cosmic web analysis
python -m astro_lab.cli.preprocessing nsa --max-samples 640000 --enable-clustering --enable-statistics --output-dir results/nsa_cosmic_web

# Direct cosmic web script  
python process_nsa_cosmic_web.py
\\\

### Results
- Multi-scale clustering: 5-50 Mpc hierarchy
- Galaxy groups to superclusters identified
- Cosmic web backbone mapped
- Large-scale structure confirmed

## Data Products
- **nsa_coords_3d_mpc.pt**: 3D comoving coordinates
- **nsa_cosmic_web_summary.txt**: Analysis summary
- **cluster_labels_*.pt**: Multi-scale cluster assignments

## Scientific Significance
This analysis reveals the hierarchical nature of cosmic structure from galaxy clusters (5 Mpc) to the cosmic web backbone (50 Mpc), confirming Lambda-CDM cosmological predictions through galaxy distribution mapping.

