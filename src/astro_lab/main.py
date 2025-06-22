import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium", sql_output="native")

with app.setup:
    import marimo as mo
    import polars as pl
    import astro_lab as al

    from astro_lab.data.manager import (
        download_gaia,
        download_bright_all_sky,
        load_gaia_bright_stars,
        list_catalogs,
        load_catalog
    )


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
    _df = list_catalogs()
    return


@app.cell(hide_code=True)
def _(alt):
    _chart = (
        alt.Chart(list_catalogs())
        .mark_bar()
        .transform_aggregate(count="count()", groupby=["type"])
        .transform_window(
            rank="rank()",
            sort=[
                alt.SortField("count", order="descending"),
                alt.SortField("type", order="ascending"),
            ],
        )
        .transform_filter(alt.datum.rank <= 10)
        .encode(
            y=alt.Y(
                "type:N",
                sort="-x",
                axis=alt.Axis(title=None),
            ),
            x=alt.X("count:Q", title="Datasets"),
            tooltip=[
                alt.Tooltip("type:N"),
                alt.Tooltip("count:Q", format=",.0f", title="Datasets"),
            ],
        )
        .properties(width="container")
        .configure_view(stroke=None)
        .configure_axis(grid=False)
    )
    _chart
    return


@app.cell
def _():
    import altair as alt
    return (alt,)


@app.cell
def _():
    silent = mo.ui.checkbox(label="Silent")
    train_button = mo.ui.button(label="Train")
    train_ui = mo.vstack([silent,train_button])
    dir_browser = mo.ui.file_browser(initial_path="data",selection_mode="directory")
    return dir_browser, silent, train_ui


@app.cell(hide_code=True)
def _(dir_browser, train_ui):

    file_browser = mo.ui.file_browser(filetypes=[".parquet",".pt"], initial_path=dir_browser.path(), restrict_navigation=True)
    menu = mo.accordion({
        "Data": mo.hstack([dir_browser,file_browser]),
        "Train": train_ui
    })
    menu
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
