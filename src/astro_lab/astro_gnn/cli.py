"""
Command Line Interface für AstroGNN

Einfaches CLI für Training, Evaluation und Inference des AstroGNN Modells.
"""

import click
import torch
import pytorch_lightning as pl
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track

from .config import Config
from .model import AstroPointCloudGNN
from .data import AstroPointCloudDataset, create_dataloaders
from .trainer import AstroGNNTrainer, train_model

# Try to import survey integration
try:
    from .surveys import SurveyDataLoader
    HAS_SURVEY_INTEGRATION = True
except ImportError:
    HAS_SURVEY_INTEGRATION = False

console = Console()


@click.group()
def cli():
    """AstroGNN - Graph Neural Networks für Astronomische Punktwolken"""
    pass


@cli.command()
@click.option('--config', '-c', default='config.yaml', help='Pfad zur Konfigurationsdatei')
@click.option('--survey', '-s', type=click.Choice(['gaia', 'sdss', 'nsa', 'tng50']), help='Survey-Daten verwenden')
@click.option('--max-samples', '-m', type=int, default=None, help='Maximale Anzahl Samples')
@click.option('--resume', '-r', default=None, help='Checkpoint zum Fortsetzen')
def train(config: str, survey: str, max_samples: int, resume: str):
    """Trainiere das AstroGNN Modell"""
    console.print("[bold blue]AstroGNN Training[/bold blue]")
    
    # Lade Konfiguration
    cfg = Config.from_yaml(config)
    console.print(f"[green]✓[/green] Konfiguration geladen: {config}")
    
    # Wenn Survey angegeben, nutze Survey-Daten
    if survey and HAS_SURVEY_INTEGRATION:
        console.print(f"[yellow]Lade {survey.upper()} Survey-Daten...[/yellow]")
        
        survey_loader = SurveyDataLoader(
            survey=survey,
            num_points=cfg.data.num_points,
            k_neighbors=cfg.model.k_neighbors,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers
        )
        
        # Erstelle DataLoader
        train_loader, val_loader, test_loader = survey_loader.create_dataloaders(
            max_samples=max_samples,
            train_ratio=cfg.data.train_split,
            val_ratio=cfg.data.val_split
        )
        
        # Zeige Survey-Info
        survey_info = survey_loader.get_survey_info()
        console.print(f"[green]✓[/green] {survey.upper()} Survey geladen:")
        console.print(f"  - Objekte: {survey_info['num_objects']:,}")
        console.print(f"  - Position Dim: {survey_info['position_dim']}")
        console.print(f"  - Feature Dim: {survey_info['feature_dim']}")
        
    else:
        # Standard: Erstelle Dataset
        console.print("[yellow]Lade Daten...[/yellow]")
        dataset = AstroPointCloudDataset(
            root=cfg.data_path,
            num_points=cfg.data.num_points,
            normalize=cfg.data.normalize
        )
        
        # Erstelle DataLoader
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset,
            batch_size=cfg.data.batch_size,
            train_split=cfg.data.train_split,
            val_split=cfg.data.val_split,
            num_workers=cfg.data.num_workers,
            seed=cfg.seed
        )
        
        console.print(f"[green]✓[/green] Dataset geladen: {len(dataset)} Samples")
    
    console.print(f"  - Training: {len(train_loader.dataset)} Samples")
    console.print(f"  - Validation: {len(val_loader.dataset)} Samples")
    console.print(f"  - Test: {len(test_loader.dataset)} Samples")
    
    # Erstelle Trainer und Modell
    trainer, model = train_model(cfg)
    
    # Training
    console.print("\n[bold green]Starte Training...[/bold green]")
    
    if resume:
        console.print(f"[yellow]Setze Training fort von: {resume}[/yellow]")
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume)
    else:
        trainer.fit(model, train_loader, val_loader)
    
    # Test
    console.print("\n[bold blue]Evaluiere auf Test Set...[/bold blue]")
    trainer.test(model, test_loader)
    
    console.print("[bold green]✓ Training abgeschlossen![/bold green]")


@cli.command()
@click.option('--config', '-c', default='config.yaml', help='Pfad zur Konfigurationsdatei')
@click.option('--checkpoint', '-ckpt', required=True, help='Pfad zum Modell-Checkpoint')
@click.option('--data-path', '-d', default=None, help='Pfad zu Test-Daten')
def evaluate(config: str, checkpoint: str, data_path: str):
    """Evaluiere ein trainiertes Modell"""
    console.print("[bold blue]AstroGNN Evaluation[/bold blue]")
    
    # Lade Konfiguration
    cfg = Config.from_yaml(config)
    
    # Lade Modell
    console.print(f"[yellow]Lade Modell von: {checkpoint}[/yellow]")
    model = AstroGNNTrainer.load_from_checkpoint(checkpoint, config=cfg)
    model.eval()
    
    # Lade Daten
    data_path = data_path or cfg.data_path
    dataset = AstroPointCloudDataset(
        root=data_path,
        num_points=cfg.data.num_points,
        normalize=cfg.data.normalize
    )
    
    _, _, test_loader = create_dataloaders(
        dataset,
        batch_size=cfg.data.batch_size,
        train_split=0,
        val_split=0,
        num_workers=cfg.data.num_workers
    )
    
    # Evaluiere
    trainer = pl.Trainer(
        accelerator="gpu" if cfg.device == "cuda" else "cpu",
        devices=1
    )
    
    results = trainer.test(model, test_loader)
    
    # Zeige Ergebnisse
    table = Table(title="Evaluation Ergebnisse")
    table.add_column("Metrik", style="cyan")
    table.add_column("Wert", style="magenta")
    
    for key, value in results[0].items():
        table.add_row(key, f"{value:.4f}")
    
    console.print(table)


@cli.command()
@click.option('--checkpoint', '-ckpt', required=True, help='Pfad zum Modell-Checkpoint')
@click.option('--input', '-i', required=True, help='Pfad zu Input-Daten')
@click.option('--output', '-o', default='predictions.csv', help='Output-Datei')
def predict(checkpoint: str, input: str, output: str):
    """Führe Inference auf neuen Daten aus"""
    console.print("[bold blue]AstroGNN Prediction[/bold blue]")
    
    # Lade Modell
    console.print(f"[yellow]Lade Modell von: {checkpoint}[/yellow]")
    
    # Extrahiere Config aus Checkpoint
    ckpt = torch.load(checkpoint)
    cfg = Config(**ckpt['hyper_parameters']['config'])
    
    model = AstroGNNTrainer.load_from_checkpoint(checkpoint, config=cfg)
    model.eval()
    
    # Lade Daten
    # Hier würde man echte Daten laden
    # Für Demo verwenden wir das Dataset
    dataset = AstroPointCloudDataset(
        root=input,
        num_points=cfg.data.num_points,
        normalize=cfg.data.normalize
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )
    
    # Predictions
    predictions = []
    embeddings = []
    
    console.print("[yellow]Führe Predictions aus...[/yellow]")
    
    with torch.no_grad():
        for batch in track(dataloader, description="Processing..."):
            # Move to device
            if cfg.device == "cuda" and torch.cuda.is_available():
                batch = batch.cuda()
            
            # Predict
            result = model.predict_step(batch, 0)
            
            predictions.extend(result['predictions'].cpu().numpy())
            embeddings.extend(result['embeddings'].cpu().numpy())
    
    # Speichere Ergebnisse
    import pandas as pd
    
    df = pd.DataFrame({
        'prediction': predictions,
        'embedding_0': [e[0] for e in embeddings],
        'embedding_1': [e[1] for e in embeddings],
        # ... mehr embedding dimensions wenn nötig
    })
    
    df.to_csv(output, index=False)
    console.print(f"[green]✓[/green] Predictions gespeichert in: {output}")


@cli.command()
def create_config():
    """Erstelle eine Beispiel-Konfigurationsdatei"""
    console.print("[bold blue]Erstelle Beispiel-Konfiguration[/bold blue]")
    
    # Erstelle Standard-Config
    config = Config()
    
    # Speichere als YAML
    config.to_yaml("config_example.yaml")
    
    console.print("[green]✓[/green] Beispiel-Konfiguration erstellt: config_example.yaml")
    console.print("\nBearbeiten Sie die Datei für Ihre Bedürfnisse und starten Sie das Training mit:")
    console.print("[bold]python -m astro_lab.astro_gnn.cli train -c config_example.yaml[/bold]")


@cli.command()
@click.option('--survey', '-s', type=click.Choice(['gaia', 'sdss', 'nsa', 'tng50']), required=True, help='Survey Name')
@click.option('--max-samples', '-m', type=int, default=1000, help='Maximale Anzahl Samples für Analyse')
def explore_survey(survey: str, max_samples: int):
    """Erkunde Survey-Daten"""
    if not HAS_SURVEY_INTEGRATION:
        console.print("[red]Survey-Integration nicht verfügbar![/red]")
        return
    
    console.print(f"[bold blue]Erkunde {survey.upper()} Survey[/bold blue]")
    
    # Lade Survey
    loader = SurveyDataLoader(survey=survey)
    data = loader.load_and_preprocess(max_samples=max_samples)
    
    # Zeige Statistiken
    info = data["survey_info"]
    positions = data["positions"]
    features = data["features"]
    labels = data["labels"]
    
    # Erstelle Tabelle
    table = Table(title=f"{survey.upper()} Survey Übersicht")
    table.add_column("Eigenschaft", style="cyan")
    table.add_column("Wert", style="magenta")
    
    table.add_row("Anzahl Objekte", f"{info['num_objects']:,}")
    table.add_row("Position Dimensionen", str(info['position_dim']))
    table.add_row("Feature Dimensionen", str(info['feature_dim']))
    table.add_row("Position Spalten", ", ".join(info['position_cols']))
    table.add_row("Feature Spalten", ", ".join(info['feature_cols'][:5]) + ("..." if len(info['feature_cols']) > 5 else ""))
    
    # Position Statistiken
    pos_min = positions.min(dim=0).values
    pos_max = positions.max(dim=0).values
    pos_mean = positions.mean(dim=0)
    
    table.add_row("Position Min", f"[{pos_min[0]:.2f}, {pos_min[1]:.2f}, {pos_min[2]:.2f}]")
    table.add_row("Position Max", f"[{pos_max[0]:.2f}, {pos_max[1]:.2f}, {pos_max[2]:.2f}]")
    table.add_row("Position Mean", f"[{pos_mean[0]:.2f}, {pos_mean[1]:.2f}, {pos_mean[2]:.2f}]")
    
    # Label Statistiken
    if labels is not None:
        unique_labels = torch.unique(labels)
        table.add_row("Anzahl Klassen", str(len(unique_labels)))
        table.add_row("Klassen", str(unique_labels.tolist()))
    
    console.print(table)
    
    # Optional: Speichere Sample für Visualisierung
    console.print("\n[yellow]Speichere Sample für Visualisierung...[/yellow]")
    sample_path = Path(f"{survey}_sample.pt")
    torch.save({
        "positions": positions[:1000],
        "features": features[:1000],
        "labels": labels[:1000] if labels is not None else None
    }, sample_path)
    console.print(f"[green]✓[/green] Sample gespeichert: {sample_path}")


if __name__ == "__main__":
    cli()
