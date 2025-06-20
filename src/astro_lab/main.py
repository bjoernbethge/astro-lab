import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium", sql_output="native")

with app.setup:
    import marimo as mo
    import polars as pl
    import astro_lab as al


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
    mo.md(r""" """)
    return


@app.cell
def _():
    silent = mo.ui.checkbox(label="Silent")
    train_button = mo.ui.button(label="Train")
    train_ui = mo.vstack([silent,train_button])
    dir_browser = mo.ui.file_browser(initial_path="data",selection_mode="directory")
    return dir_browser, silent


@app.cell(hide_code=True)
def _(dir_browser):

    file_browser = mo.ui.file_browser(filetypes=[".parquet",".pt"], initial_path=dir_browser.path())
    menu = mo.accordion({
        "Data": mo.callout(mo.hstack([dir_browser,file_browser])),
        "Train": mo.hstack([dir_browser,file_browser])
    })
    menu
    return


@app.cell
def _():
    dataset_file = mo.ui.file(filetypes=[".parquet"], kind="area")
    dataset_file
    return (dataset_file,)


@app.cell
def _(dataset_file):
    mo.stop(dataset_file.value is None)
    df = pl.read_parquet(dataset_file.contents())
    mo.ui.dataframe(df)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
