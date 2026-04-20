"""Marimo notebook: Gaia real-data training using shared AstroLab Marimo UI widgets."""

import marimo

__generated_with = "0.14.0"
app = marimo.App()


@app.cell
def imports():
    import marimo as mo

    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import MLFlowLogger

    from astro_lab.config import get_data_paths
    from astro_lab.data import create_datamodule
    from astro_lab.models import create_model
    from astro_lab.ui.components.config import (
        create_data_config,
        gaia_real_data_config_dict,
    )
    from astro_lab.ui.components.training_config import create_training_config

    return (
        MLFlowLogger,
        ModelCheckpoint,
        EarlyStopping,
        create_data_config,
        create_model,
        create_training_config,
        gaia_real_data_config_dict,
        get_data_paths,
        mo,
        pl,
    )


@app.cell
def intro(mo):
    mo.md(
        """
# Train GNN on real Gaia (and other surveys) — real data pipeline

Uses the **same Marimo UI building blocks** as the main AstroLab dashboard:
`create_data_config()` and `create_training_config()` from `astro_lab.ui.components`.

Pipeline matches `examples/train_gaia_real_data.py`: `create_datamodule` → prepare/setup →
`create_model` → Lightning `Trainer` (checkpoints, early stopping, MLflow).

Reactive cells re-run when widgets change (including training). Lower **Max samples** while iterating.
"""
    )
    return


@app.cell
def data_config_ui(mo, create_data_config):
    mo.md("### Data & graph")
    ui = create_data_config()
    ui
    return (ui,)


@app.cell
def training_config_ui(mo, create_training_config):
    mo.md("### Training")
    ui = create_training_config()
    ui
    return (ui,)


@app.cell
def pipeline_config(data_config_ui, gaia_real_data_config_dict, training_config_ui):
    return gaia_real_data_config_dict(
        data_config_ui.value,
        training_config_ui.value,
    )


@app.cell
def prepare_data(create_datamodule, mo, pipeline_config):
    mo.md("### Loading data")
    dm = create_datamodule(
        survey=pipeline_config["survey"],
        task=pipeline_config["task"],
        max_samples=pipeline_config["max_samples"],
        num_workers=pipeline_config["num_workers"],
        k_neighbors=pipeline_config["k_neighbors"],
        graph_method="knn",
        astronomical_features=pipeline_config["astronomical_features"],
        cosmic_web_features=pipeline_config["cosmic_web_features"],
        multi_scale=pipeline_config["multi_scale"],
        batch_size=pipeline_config["batch_size"],
    )
    dm.prepare_data()
    dm.setup()
    info = dm.get_info()
    stats_md = (
        f"**Data ready** — nodes: {info['num_nodes']:,}, edges: {info['num_edges']:,}, "
        f"features: {info['num_features']}, classes: {info['num_classes']}"
    )
    if "graph_stats" in info:
        gs = info["graph_stats"]
        stats_md += (
            f"\n\nGraph stats: avg degree {gs['avg_degree']:.2f}, "
            f"max degree {gs['max_degree']}"
        )
    mo.md(stats_md)
    return {"dm": dm, "info": info}


@app.cell
def build_model(create_model, mo, pipeline_config, prepare_data):
    info = prepare_data["info"]
    mo.md(f"### Model ({pipeline_config['conv_type'].upper()})")
    model = create_model(
        model_type="astro_model",
        in_channels=info["num_features"],
        hidden_channels=pipeline_config["hidden_dim"],
        out_channels=info["num_classes"],
        num_layers=pipeline_config["num_layers"],
        conv_type=pipeline_config["conv_type"],
        dropout=pipeline_config["dropout"],
        task=pipeline_config["task"],
        learning_rate=pipeline_config["learning_rate"],
        weight_decay=pipeline_config["weight_decay"],
        optimizer=pipeline_config["optimizer"],
        scheduler=pipeline_config["scheduler"],
    )
    mo.ui.text_area(value=model.get_model_summary(), label="Model summary")
    return (model,)


@app.cell
def train_and_test(
    EarlyStopping,
    MLFlowLogger,
    ModelCheckpoint,
    get_data_paths,
    mo,
    model,
    pipeline_config,
    pl,
    prepare_data,
):
    dm = prepare_data["dm"]
    info = prepare_data["info"]
    mo.md("### Training")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints/gaia",
        filename="gaia-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
    )
    data_paths = get_data_paths()
    mlf_logger = MLFlowLogger(
        experiment_name="gaia_real_data",
        tracking_uri=f"file:///{data_paths['mlruns_dir']}",
        tags={
            "survey": pipeline_config["survey"],
            "task": pipeline_config["task"],
            "conv_type": pipeline_config["conv_type"],
            "data_size": info["num_nodes"],
            "model_preset": pipeline_config["model_preset"],
        },
    )
    trainer = pl.Trainer(
        max_epochs=pipeline_config["max_epochs"],
        accelerator=pipeline_config["accelerator"],
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=mlf_logger,
        log_every_n_steps=10,
        gradient_clip_val=pipeline_config["gradient_clip_val"],
        precision=pipeline_config["precision"],
        enable_progress_bar=True,
    )
    log_cfg = {
        k: v
        for k, v in pipeline_config.items()
        if k
        not in (
            "astronomical_features",
            "cosmic_web_features",
            "multi_scale",
        )
    }
    mlf_logger.log_hyperparams(log_cfg)
    mo.md(
        f"**Accelerator:** {pipeline_config['accelerator']} "
        f"(precision {pipeline_config['precision']}) — demo labels where applicable."
    )
    trainer.fit(model, dm)
    mo.md("### Test")
    test_out = trainer.test(model, dm)
    mo.md(
        f"**Done.** Best checkpoint: `{checkpoint_callback.best_model_path}`\n\n"
        f"**MLflow:** `mlflow ui --backend-store-uri {data_paths['mlruns_dir']}`\n\n"
        "**Advanced**\n"
        "- AstroLab catalog: `python scripts/generate_astrolab_catalog.py`\n"
        "- Conv types: gcn, gat, sage, gin (training form)\n"
        "- Raise **Max samples** when stable"
    )
    return trainer, test_out


if __name__ == "__main__":
    app.run()
