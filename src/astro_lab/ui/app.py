import marimo

__generated_with = "0.14.0"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    from astro_lab.ui.dashboard import create_astrolab_dashboard
    
    # Create the main dashboard
    dashboard = create_astrolab_dashboard()
    return mo, create_astrolab_dashboard, dashboard


@app.cell
def __(dashboard):
    # Display the dashboard
    dashboard
    return


if __name__ == "__main__":
    app.run()
