import marimo

__generated_with = "0.14.0"
app = marimo.App()


@app.cell
def import_dependencies():
    import marimo as mo

    from astro_lab.config import get_data_paths
    from astro_lab.data.dataset.lightning import AstroLabDataModule
    from astro_lab.models import AstroModel
    from astro_lab.training import AstroTrainer

    return AstroLabDataModule, AstroModel, AstroTrainer, mo


@app.cell
def ui_training_config(mo):
    # UI for training config
    epochs = mo.ui.slider(5, 100, value=20, label="Max Epochs")
    lr = mo.ui.number(value=1e-3, label="Learning Rate")
    batch_size = mo.ui.slider(8, 256, value=32, label="Batch Size")
    hidden_dim = mo.ui.slider(32, 512, value=128, label="Hidden Dim")
    num_layers = mo.ui.slider(1, 8, value=3, label="Num Layers")
    conv_type = mo.ui.dropdown(
        ["gcn", "gat", "sage", "gin"], value="gcn", label="Conv Type"
    )
    survey = mo.ui.dropdown(["gaia"], value="gaia", label="Survey")
    return batch_size, conv_type, epochs, hidden_dim, lr, num_layers, survey


@app.cell
def show_config(
    batch_size,
    conv_type,
    epochs,
    hidden_dim,
    lr,
    mo,
    num_layers,
    survey,
):
    # Show config summary
    mo.output.clear()
    mo.output.append(
        mo.md(
            f"**Training Config:**\n- Epochs: {epochs.value}\n- LR: {lr.value}\n- Batch Size: {batch_size.value}\n- Hidden Dim: {hidden_dim.value}\n- Num Layers: {num_layers.value}\n- Conv: {conv_type.value}\n- Survey: {survey.value}"
        )
    )
    config = dict(
        max_epochs=epochs.value,
        learning_rate=lr.value,
        batch_size=batch_size.value,
        hidden_dim=hidden_dim.value,
        num_layers=num_layers.value,
        conv_type=conv_type.value,
        survey=survey.value,
    )
    return (config,)


@app.cell
def load_data(Astr, AstroLabDataModule, config, mo):
    # Only use real GaiaDataset, fail if not available
    import importlib.util

    gaia_mod = importlib.util.find_spec("astro_lab.data.collectors.gaia")
    if gaia_mod is None:
        raise ImportError(
            "GaiaDataset is not available. Please ensure astro_lab.data.collectors.gaia is installed and accessible."
        )
    from astro_lab.data.collectors.gaia import GaiaDataset

    dataset = Astr(max_samples=1000)
    datamodule = AstroLabDataModule(dataset, batch_size=config["batch_size"])
    datamodule.setup()
    info = {"num_features": 8, "num_classes": 2}
    if hasattr(datamodule, "get_info"):
        try:
            info = datamodule.get_info()
        except Exception:
            pass
    mo.output.append(
        mo.md(
            f"**Data loaded:** Features: {info['num_features']}, Classes: {info['num_classes']}"
        )
    )
    return datamodule, info


@app.cell
def create_model(AstroModel, config, info, mo):
    # Model creation
    model = AstroModel(
        num_features=info["num_features"],
        num_classes=info["num_classes"],
        hidden_dim=config["hidden_dim"],
        learning_rate=config["learning_rate"],
        task="node_classification",
    )
    mo.output.append(mo.md("**Model Summary:**"))
    mo.output.append(
        mo.ui.text_area(value=model.get_model_summary(), label="Model Summary")
    )
    return (model,)


@app.cell
def train_model(AstroTrainer, config, datamodule, mo, model, torch):
    # Training
    mo.output.append(mo.md("**Training...**"))
    trainer = AstroTrainer(
        max_epochs=config["max_epochs"],
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=5,
    )
    trainer.fit(model, datamodule)
    mo.output.append(mo.md("**Training complete!**"))
    return (trainer,)


@app.cell
def test_model(datamodule, mo, model, trainer):
    # Test
    mo.output.append(mo.md("**Testing...**"))
    results = trainer.test(model, datamodule)
    mo.output.append(mo.md(f"**Test Results:** {results}"))
    return


if __name__ == "__main__":
    app.run()
