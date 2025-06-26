import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium", sql_output="native")

with app.setup:
    import marimo as mo

    from astro_lab.ui.dashboard import create_astrolab_dashboard


@app.cell
def _():
    create_astrolab_dashboard()
    return


if __name__ == "__main__":
    app.run()
