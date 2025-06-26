"""
Demo-Skript für AstroGNN

Zeigt die Verwendung des Frameworks mit synthetischen Daten.
"""

import torch
import numpy as np
from pathlib import Path
from rich.console import Console
from torch_geometric.data import Data
from tensordict import TensorDict

from .config import Config
from .model import AstroPointCloudGNN
from .data import AstroPointCloudDataset

console = Console()


def create_synthetic_cluster(n_points: int = 1024, cluster_type: int = 0) -> Data:
    """
    Erstellt einen synthetischen Sternhaufen
    
    Args:
        n_points: Anzahl der Sterne
        cluster_type: Typ des Clusters (0-6 für O-M Sterne)
    
    Returns:
        PyG Data Objekt
    """
    # Generiere Positionen (Gauss-verteilt)
    center = np.random.randn(3) * 10
    positions = np.random.randn(n_points, 3) + center
    
    # Generiere Features basierend auf Sterntyp
    features = np.zeros((n_points, 4))
    
    # Magnitude (heller für frühe Typen)
    features[:, 0] = np.random.normal(5.0 - cluster_type * 0.5, 1.0, n_points)
    
    # B-V Farbe (blauer für frühe Typen)
    features[:, 1] = np.random.normal(cluster_type * 0.3, 0.1, n_points)
    
    # Temperatur (heißer für frühe Typen)
    features[:, 2] = np.random.normal(10000 - cluster_type * 1000, 500, n_points)
    
    # Masse (schwerer für frühe Typen)
    features[:, 3] = np.random.normal(3.0 - cluster_type * 0.3, 0.2, n_points)
    
    # Konvertiere zu Tensoren
    pos_tensor = torch.tensor(positions, dtype=torch.float32)
    feat_tensor = torch.tensor(features, dtype=torch.float32)
    label_tensor = torch.tensor(cluster_type, dtype=torch.long)
    
    return Data(x=feat_tensor, pos=pos_tensor, y=label_tensor)


def demo_model():
    """Demonstriert die Modell-Funktionalität"""
    console.print("[bold blue]AstroGNN Demo[/bold blue]\n")
    
    # 1. Konfiguration
    console.print("[yellow]1. Erstelle Konfiguration[/yellow]")
    config = Config(
        experiment_name="demo_run",
        model=Config.model.__class__(
            input_features=4,  # 4 Features pro Stern
            output_classes=7,  # 7 Sterntypen
            hidden_dim=64,     # Kleiner für Demo
            k_neighbors=8
        )
    )
    console.print("✓ Konfiguration erstellt")
    
    # 2. Modell
    console.print("\n[yellow]2. Initialisiere Modell[/yellow]")
    model = AstroPointCloudGNN(
        num_features=config.model.input_features,
        num_classes=config.model.output_classes,
        hidden_dim=config.model.hidden_dim,
        k_neighbors=config.model.k_neighbors
    )
    console.print(f"✓ Modell erstellt mit {sum(p.numel() for p in model.parameters())} Parametern")
    
    # 3. Synthetische Daten
    console.print("\n[yellow]3. Generiere synthetische Sternhaufen[/yellow]")
    clusters = []
    cluster_names = ['O-Typ', 'B-Typ', 'A-Typ', 'F-Typ', 'G-Typ', 'K-Typ', 'M-Typ']
    
    for i, name in enumerate(cluster_names):
        cluster = create_synthetic_cluster(n_points=512, cluster_type=i)
        clusters.append(cluster)
        console.print(f"  ✓ {name} Haufen generiert")
    
    # 4. Forward Pass Demo
    console.print("\n[yellow]4. Teste Forward Pass[/yellow]")
    model.eval()
    
    with torch.no_grad():
        # Einzelner Cluster
        test_cluster = clusters[2]  # A-Typ Stern
        
        # Erstelle Graph
        data = model.create_graph_from_pointcloud(
            test_cluster.pos, 
            test_cluster.x
        )
        
        # Forward Pass
        logits = model(data)
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=-1)
        
        console.print(f"  Input: {cluster_names[2]} Haufen")
        console.print(f"  Prediction: {cluster_names[pred.item()]}")
        console.print(f"  Konfidenz: {probs.max().item():.2%}")
    
    # 5. TensorDict Demo
    console.print("\n[yellow]5. Teste TensorDict Interface[/yellow]")
    
    # Erstelle TensorDict
    td = TensorDict({
        "positions": test_cluster.pos,
        "features": test_cluster.x,
        "labels": test_cluster.y.unsqueeze(0)
    }, batch_size=[1])
    
    # Forward mit TensorDict
    output_td = model.forward_tensordict(td)
    
    console.print("  TensorDict Output:")
    console.print(f"    - logits: {output_td['logits'].shape}")
    console.print(f"    - predictions: {output_td['predictions'].item()}")
    console.print(f"    - embeddings: {output_td['embeddings'].shape}")
    
    # 6. Batch Processing
    console.print("\n[yellow]6. Teste Batch Processing[/yellow]")
    
    # Erstelle Batch
    from torch_geometric.data import Batch
    batch = Batch.from_data_list(clusters[:3])
    
    # Forward Pass
    logits = model(batch)
    preds = torch.argmax(logits, dim=-1)
    
    console.print(f"  Batch Größe: {batch.num_graphs}")
    console.print(f"  Predictions: {preds.tolist()}")
    console.print(f"  Ground Truth: {batch.y.tolist()}")
    
    # 7. Feature Extraction
    console.print("\n[yellow]7. Extrahiere Embeddings[/yellow]")
    
    embeddings = model.get_embeddings(batch)
    console.print(f"  Embedding Shape: {embeddings.shape}")
    console.print("  Verwendbar für:")
    console.print("    - t-SNE/UMAP Visualisierung")
    console.print("    - Ähnlichkeitssuche")
    console.print("    - Transfer Learning")
    
    console.print("\n[bold green]✓ Demo erfolgreich abgeschlossen![/bold green]")
    console.print("\nNächste Schritte:")
    console.print("1. Erstelle echte Daten im HDF5 Format")
    console.print("2. Passe die Konfiguration an")
    console.print("3. Starte das Training mit: python -m astro_lab.astro_gnn train")


if __name__ == "__main__":
    demo_model()
