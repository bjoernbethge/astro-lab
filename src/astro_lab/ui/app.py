import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo

    from astro_lab.ui.pages.cosmic_web_page import create_cosmic_web_page
    from astro_lab.ui.pages.data_page import create_data_page
    from astro_lab.ui.pages.training_page import create_training_page

    tab_selector = mo.ui.tabs(
        {
            "📡 Data & Visualization": "data",
            "🌌 Cosmic Web Analysis": "cosmic_web",
            "🚀 Model Training": "training",
        }
    )

    return (
        mo,
        tab_selector,
        create_data_page,
        create_cosmic_web_page,
        create_training_page,
    )


@app.cell
def _(mo, tab_selector, create_data_page, create_cosmic_web_page, create_training_page):
    if tab_selector.value == "data":
        content = create_data_page()
    elif tab_selector.value == "cosmic_web":
        content = create_cosmic_web_page()
    elif tab_selector.value == "training":
        content = create_training_page()
    else:
        content = mo.md("# 🌌 AstroLab - Astronomical Graph Neural Networks")

    return content


if __name__ == "__main__":
    app.run()
