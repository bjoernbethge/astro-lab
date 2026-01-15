import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium", sql_output="native")


@app.cell
def _():
    import marimo as mo
    from astropy.io import fits
    import numpy as np
    import polars as pl
    import astrophot as ap
    return mo, pl


@app.cell
def _(mo):


    fits_file = mo.ui.file(filetypes=[".parquet"])

    return (fits_file,)


@app.cell
def _(fits_file):
    fits_file
    return


@app.cell
def _(fits_file, mo, pl):
    linear_mapping = {
        "raLIN": "ra",
        "decLIN": "dec"
    }
    if fits_file.value:
        df = pl.read_parquet(fits_file.contents(0))
        df_next = df.rename(linear_mapping)
        editor = mo.ui.dataframe(df_next.limit(10))
        mo.output.append(editor)
    else:
        df = None
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
