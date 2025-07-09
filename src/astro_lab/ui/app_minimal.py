import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return mo.md("Hello from minimal marimo app!")


if __name__ == "__main__":
    app.run()
