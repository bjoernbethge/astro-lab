"""
Demo für AstroGNN mit echten Survey-Daten

Zeigt die Integration mit Gaia, SDSS, NSA und TNG50 Daten.
"""

import torch
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .config import Config
from .model import AstroPointCloudGNN
from .surveys import SurveyDataLoader

console = Console()


def visualize_pointcloud_3d(
    positions: torch.Tensor,
    labels: torch.Tensor = None,
    title: str = "Astronomical Point Cloud",
    save_path: str = None
):
    """
    Visualisiert eine 3D Punktwolke.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Konvertiere zu numpy
    pos = positions.numpy()
    
    # Farben basierend auf Labels
    if labels is not None:
        colors = plt.cm.tab10(labels.numpy())
    else:
        colors = 'blue'
    
    # Plot
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], 
              c=colors, s=1, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def demo_gaia_stars():
    """Demo: Gaia Sternhaufen-Analyse"""
    console.print("\n[bold blue]Demo: Gaia DR3 Sternhaufen[/bold blue]")
    
    # Lade Gaia Daten
    loader = SurveyDataLoader(
        survey="gaia",
        num_points=1024,
        k_neighbors=16
    )
    
    # Lade kleine Probe
    data = loader.load_and_preprocess(max_samples=5000)
    
    console.print(f"✅ Geladen: {data['survey_info']['num_objects']} Sterne")
    
    # Visualisiere Probe
    sample_idx = torch.randperm(len(data['positions']))[:1000]
    visualize_pointcloud_3d(
        data['positions'][sample_idx],
        data['labels'][sample_idx] if data['labels'] is not None else None,
        title="Gaia Sternverteilung (1000 Sterne)",
        save_path="gaia_stars_3d.png"
    )
    
    # Trainiere Mini-Modell
    console.print("\n[yellow]Trainiere Mini-Modell für Sternklassifikation...[/yellow]")
    
    config = Config(
        experiment_name="gaia_demo",
        model=Config.model.__class__(
            input_features=data['features'].shape[1],
            output_classes=7,  # O, B, A, F, G, K, M
            hidden_dim=64,
            k_neighbors=8
        )
    )
    
    model = AstroPointCloudGNN(
        num_features=config.model.input_features,
        num_classes=config.model.output_classes,
        hidden_dim=config.model.hidden_dim,
        k_neighbors=config.model.k_neighbors
    )
    
    console.print(f"✅ Modell erstellt: {sum(p.numel() for p in model.parameters())} Parameter")
    
    # Quick Test
    test_batch = loader.create_dataset(split="full")[0]
    with torch.no_grad():
        output = model(test_batch)
        console.print(f"✅ Test Forward Pass: Output shape {output.shape}")


def demo_sdss_galaxies():
    """Demo: SDSS Galaxien-Analyse"""
    console.print("\n[bold blue]Demo: SDSS Galaxien[/bold blue]")
    
    # Lade SDSS Daten
    loader = SurveyDataLoader(
        survey="sdss",
        num_points=512,
        k_neighbors=8
    )
    
    # Lade kleine Probe
    data = loader.load_and_preprocess(max_samples=2000)
    
    console.print(f"✅ Geladen: {data['survey_info']['num_objects']} Galaxien")
    
    # Zeige Feature-Statistiken
    features = data['features']
    feature_names = data['survey_info']['feature_cols']
    
    table = Table(title="SDSS Feature Statistiken")
    table.add_column("Feature", style="cyan")
    table.add_column("Min", style="yellow")
    table.add_column("Max", style="yellow")
    table.add_column("Mean", style="green")
    table.add_column("Std", style="magenta")
    
    for i, name in enumerate(feature_names[:5]):  # Erste 5 Features
        col = features[:, i]
        table.add_row(
            name,
            f"{col.min():.3f}",
            f"{col.max():.3f}",
            f"{col.mean():.3f}",
            f"{col.std():.3f}"
        )
    
    console.print(table)


def demo_tng50_simulation():
    """Demo: TNG50 Simulations-Analyse"""
    console.print("\n[bold blue]Demo: TNG50 Kosmologische Simulation[/bold blue]")
    
    # Lade TNG50 Daten
    loader = SurveyDataLoader(
        survey="tng50",
        num_points=2048,
        k_neighbors=32
    )
    
    try:
        # Lade Probe
        data = loader.load_and_preprocess(max_samples=10000)
        
        console.print(f"✅ Geladen: {data['survey_info']['num_objects']} Partikel")
        
        # Visualisiere Dichteverteilung
        positions = data['positions'][:5000]
        features = data['features'][:5000]
        
        # Verwende Masse als Farbkodierung
        if features.shape[1] > 0:
            masses = features[:, 0]  # Erste Feature ist normalerweise Masse
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                c=np.log10(masses + 1e-10), 
                cmap='viridis', 
                s=0.5, 
                alpha=0.6
            )
            
            plt.colorbar(scatter, label='log10(Mass)')
            ax.set_xlabel('X [kpc/h]')
            ax.set_ylabel('Y [kpc/h]')
            ax.set_zlabel('Z [kpc/h]')
            ax.set_title('TNG50 Materieverteilung')
            
            plt.savefig("tng50_matter_distribution.png", dpi=150)
            plt.show()
            
    except Exception as e:
        console.print(f"[red]Fehler beim Laden von TNG50: {e}[/red]")
        console.print("[yellow]Stelle sicher, dass TNG50 Daten vorhanden sind[/yellow]")


def demo_cross_survey_comparison():
    """Demo: Vergleich zwischen verschiedenen Surveys"""
    console.print("\n[bold blue]Demo: Cross-Survey Vergleich[/bold blue]")
    
    surveys = ["gaia", "sdss", "nsa"]
    survey_data = {}
    
    # Lade Daten von verschiedenen Surveys
    for survey in surveys:
        try:
            loader = SurveyDataLoader(survey=survey)
            data = loader.load_and_preprocess(max_samples=1000)
            survey_data[survey] = data
            console.print(f"✅ {survey.upper()}: {data['survey_info']['num_objects']} Objekte")
        except Exception as e:
            console.print(f"[red]Fehler bei {survey}: {e}[/red]")
    
    # Vergleichstabelle
    if survey_data:
        table = Table(title="Survey Vergleich")
        table.add_column("Survey", style="cyan")
        table.add_column("Objekte", style="yellow")
        table.add_column("Pos Dim", style="green")
        table.add_column("Feature Dim", style="magenta")
        table.add_column("Koordinaten", style="blue")
        
        for survey, data in survey_data.items():
            info = data['survey_info']
            table.add_row(
                survey.upper(),
                f"{info['num_objects']:,}",
                str(info['position_dim']),
                str(info['feature_dim']),
                ", ".join(info['position_cols'][:3])
            )
        
        console.print(table)


def main():
    """Hauptfunktion für Survey-Demos"""
    console.print("[bold green]AstroGNN Survey Integration Demo[/bold green]")
    console.print("=" * 50)
    
    demos = {
        "1": ("Gaia DR3 Sternhaufen", demo_gaia_stars),
        "2": ("SDSS Galaxien", demo_sdss_galaxies),
        "3": ("TNG50 Simulation", demo_tng50_simulation),
        "4": ("Cross-Survey Vergleich", demo_cross_survey_comparison),
        "5": ("Alle Demos", None)
    }
    
    # Menü
    console.print("\nVerfügbare Demos:")
    for key, (name, _) in demos.items():
        console.print(f"  [{key}] {name}")
    
    choice = console.input("\nWähle Demo (1-5): ")
    
    if choice == "5":
        # Führe alle Demos aus
        for key, (name, func) in demos.items():
            if key != "5" and func:
                console.print(f"\n{'='*50}")
                func()
    elif choice in demos and demos[choice][1]:
        demos[choice][1]()
    else:
        console.print("[red]Ungültige Auswahl![/red]")
    
    console.print("\n[bold green]Demo abgeschlossen![/bold green]")


if __name__ == "__main__":
    main()
