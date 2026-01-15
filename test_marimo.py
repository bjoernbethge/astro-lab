"""
Minimal Marimo Test App
"""

import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return mo.md("# Hello World")


@app.cell
def _():
    import marimo as mo

    return mo.ui.button(label="Click me")


if __name__ == "__main__":
    app.run()
