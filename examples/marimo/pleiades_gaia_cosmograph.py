import marimo

__generated_with = "0.14.0"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def intro_cell(mo):
    mo.md(
        """
    # Pleiades Cluster Visualization with Gaia Data and Cosmograph

    This demo loads processed Gaia data, filters for the Pleiades open cluster, and visualizes it interactively using Cosmograph in Marimo.
    """
    )
    return


@app.cell(hide_code=True)
def load_data_md_cell(mo):
    mo.md(
        """
    ## Load Processed Gaia Data

    We load the harmonized Gaia parquet file from the processed directory.
    """
    )
    return


@app.cell
def load_data_cell():
    from pathlib import Path

    import polars as pl

    gaia_parquet_path = Path("data/processed/gaia/gaia.parquet")
    if not gaia_parquet_path.exists():
        raise FileNotFoundError(
            f"Processed Gaia parquet not found: {gaia_parquet_path}"
        )
    gaia_df = pl.read_parquet(gaia_parquet_path)
    return gaia_df, pl


@app.cell
def filter_pleiades_md_cell(mo):
    mo.md(
        """
    ## Filter for the Pleiades Cluster

    The Pleiades cluster is centered at RA ≈ 56.75°, Dec ≈ +24.12° with a radius of about 2°. We select all Gaia sources within this region.
    """
    )
    return


@app.cell
def filter_pleiades_cell(gaia_df):
    import numpy as np

    pleiades_ra = 56.75  # degrees
    pleiades_dec = 24.12  # degrees
    pleiades_radius = 2.0  # degrees

    def angular_separation(ra1, dec1, ra2, dec2):
        ra1_rad = np.deg2rad(ra1)
        dec1_rad = np.deg2rad(dec1)
        ra2_rad = np.deg2rad(ra2)
        dec2_rad = np.deg2rad(dec2)
        delta_ra = ra2_rad - ra1_rad
        delta_dec = dec2_rad - dec1_rad
        a = (
            np.sin(delta_dec / 2) ** 2
            + np.cos(dec1_rad) * np.cos(dec2_rad) * np.sin(delta_ra / 2) ** 2
        )
        return 2 * np.arcsin(np.sqrt(a)) * (180.0 / np.pi)

    ra = gaia_df["ra"].to_numpy()
    dec = gaia_df["dec"].to_numpy()
    sep = angular_separation(ra, dec, pleiades_ra, pleiades_dec)
    pleiades_mask = sep < pleiades_radius
    pleiades_df = gaia_df.filter(pleiades_mask)
    return (pleiades_df,)


@app.cell
def prepare_coords_md_cell(mo):
    mo.md(
        """
    ## Prepare Data for Cosmograph

    We ensure the DataFrame has 3D coordinates (x, y, z). If not, we convert from (ra, dec, distance_pc).
    """
    )
    return


@app.cell
def prepare_coords_cell(pl, pleiades_df):
    if all(col in pleiades_df.columns for col in ["x", "y", "z"]):
        coords_df = pleiades_df
    else:
        from astro_lab.data.transforms.astronomical import spherical_to_cartesian

        if "distance_pc" in pleiades_df.columns:
            x, y, z = spherical_to_cartesian(
                pleiades_df["ra"].to_numpy(),
                pleiades_df["dec"].to_numpy(),
                pleiades_df["distance_pc"].to_numpy(),
            )


            coords_df = pleiades_df.with_columns(
                [
                    pl.Series("x", x),
                    pl.Series("y", y),
                    pl.Series("z", z),
                ]
            )
        else:
            raise ValueError("No distance information available for 3D coordinates.")
    return (coords_df,)


@app.cell
def visualize_md_cell(mo):
    mo.md(
        """
    ## Visualize the Pleiades Cluster with Cosmograph

    We use the AstroLab Cosmograph integration to create an interactive 3D visualization of the Pleiades cluster.
    """
    )
    return


@app.cell
def visualize_cell(coords_df):
    from astro_lab.widgets.alcg.convenience import create_cosmograph_from_dataframe

    viz = create_cosmograph_from_dataframe(
        coords_df,
        x_col="x",
        y_col="y",
        z_col="z",
        survey="gaia",
        node_size_range=[3, 10],
        point_color="gold",
        radius=2.0,
    )
    viz
    return


if __name__ == "__main__":
    app.run()
