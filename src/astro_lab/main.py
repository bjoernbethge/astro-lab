import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium", sql_output="native")

with app.setup:
    import marimo as mo
    import astrophot as ap
    import torch_geometric as pyg
    import astro_lab as al
    import polars as pl
    import torch
    import warnings


@app.cell
def _(silent):
    import mlflow as ml
    ml.set_tracking_uri("http://localhost:5000/")
    ml.set_experiment("astro-lab-experiment")
    tags = ["astro-lab","train"]

    def train(dataset, context, tags, model):
        ml.pytorch.autolog(silent=silent.value)
        with ml.start_run() as run:
            ml.log_param("my", "param")
            ml.log_metric("score", 100)
            ml.log_input(dataset, context, tags, model)
    return


@app.cell
def _():
    from astro_lab.widget import BlenderImageWidget
    widget = BlenderImageWidget()
    widget.capture_live_render()

    # Widget im Notebook anzeigen
    widget
    return


@app.cell
def _():
    silent = mo.ui.checkbox(label="Silent")
    train_button = mo.ui.button(label="Train")
    train_ui = mo.vstack([silent,train_button])
    dir_browser = mo.ui.file_browser(initial_path="data",selection_mode="directory")
    return dir_browser, silent, train_ui


@app.cell(hide_code=True)
def _(dir_browser, train_ui):

    file_browser = mo.ui.file_browser(filetypes=[".parquet"], initial_path=dir_browser.path())
    accordion = mo.ui.tabs({
        "Data": mo.hstack([dir_browser,file_browser]),
        "Train": train_ui
    })
    panel = mo.callout(accordion)
    panel
    return (file_browser,)


@app.cell
def _(file_browser):

    pl_df = pl.read_parquet(file_browser.path())
    mo_df = mo.ui.dataframe(pl_df)
    mo_df
    return


if __name__ == "__main__":
    app.run()
