import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
    # Gaia DR3 mit DuckDB Astro Extension

    Astronomische Analysen auf 2.4M Gaia-Sternen — direkt in SQL mit der
    [astro](https://github.com/synapticore-io/astro-duck) DuckDB Extension.
    """
    )
    return (mo,)


@app.cell
def setup():
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL astro FROM community")
    con.execute("LOAD astro")

    GAIA = "D:/projects/synapticore-io/astro-lab/data/raw/gaia/gaia_dr3_bright_all_sky_mag12.0.parquet"
    TWOMASS = "D:/projects/synapticore-io/astro-lab/data/raw/twomass/twomass_psc_mag4.0.parquet"

    con.execute(
        f"CREATE OR REPLACE VIEW gaia AS SELECT * FROM read_parquet('{GAIA}')"
    )
    con.execute(
        f"CREATE OR REPLACE VIEW twomass AS SELECT * FROM read_parquet('{TWOMASS}')"
    )

    n_gaia = con.execute("SELECT count(*) FROM gaia").fetchone()[0]
    n_2mass = con.execute("SELECT count(*) FROM twomass").fetchone()[0]
    print(f"Gaia DR3: {n_gaia:,} Sterne | 2MASS PSC: {n_2mass:,} Quellen")
    return GAIA, TWOMASS, con, n_gaia, n_2mass


# ── 1. HR-Diagramm ──────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 1. Hertzsprung-Russell-Diagramm

    Absolute G-Magnitude vs. effektive Temperatur — berechnet mit
    `astro_absolute_mag(phot_g_mean_mag, distance_pc)`.
    """
    )
    return


@app.cell
def hr_diagram(con, mo):
    import plotly.express as px

    hr = con.execute("""
        SELECT
            teff_gspphot AS teff,
            astro_absolute_mag(phot_g_mean_mag, distance_pc) AS abs_g,
            bp_rp
        FROM gaia
        WHERE teff_gspphot IS NOT NULL
          AND distance_pc > 0
          AND phot_g_mean_mag IS NOT NULL
          AND bp_rp IS NOT NULL
        USING SAMPLE 50000
    """).fetchdf()

    fig = px.scatter(
        hr,
        x="teff",
        y="abs_g",
        color="bp_rp",
        color_continuous_scale="RdYlBu_r",
        labels={"teff": "T_eff [K]", "abs_g": "M_G [mag]", "bp_rp": "BP-RP"},
        range_color=[-0.5, 3.5],
        opacity=0.3,
    )
    fig.update_layout(
        xaxis=dict(autorange="reversed"),
        yaxis=dict(autorange="reversed"),
        title="Gaia DR3 — HR-Diagramm (50k Sample)",
        height=600,
    )
    mo.ui.plotly(fig)
    return


# ── 2. Pleiades Cluster ─────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 2. Pleiades-Cluster (M45)

    Mitglieder-Selektion via `astro_angular_separation` + Parallax-Cut.
    """
    )
    return


@app.cell
def pleiades(con, mo):
    import plotly.express as px

    pleiades_df = con.execute("""
        SELECT
            ra, dec, parallax, phot_g_mean_mag, bp_rp,
            distance_pc,
            astro_angular_separation(ra, dec, 56.75, 24.12) AS sep_deg,
            astro_absolute_mag(phot_g_mean_mag, distance_pc) AS abs_g
        FROM gaia
        WHERE astro_angular_separation(ra, dec, 56.75, 24.12) < 3.0
          AND parallax BETWEEN 6.0 AND 9.0
          AND bp_rp IS NOT NULL
        ORDER BY phot_g_mean_mag
    """).fetchdf()

    fig = px.scatter(
        pleiades_df,
        x="bp_rp",
        y="abs_g",
        color="sep_deg",
        color_continuous_scale="Viridis_r",
        hover_data=["ra", "dec", "parallax", "distance_pc"],
        labels={
            "bp_rp": "BP-RP [mag]",
            "abs_g": "M_G [mag]",
            "sep_deg": "Abstand zum Zentrum [°]",
        },
        title=f"Pleiades CMD — {len(pleiades_df)} Mitglieder",
        opacity=0.7,
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
    mo.ui.plotly(fig)
    return


# ── 3. Photometrie-Verifikation ─────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 3. Photometrie-Check

    Vergleich: `astro_absolute_mag()` vs. vorberechnetes `abs_g_mag` aus dem Katalog.
    """
    )
    return


@app.cell
def photometry_check(con):
    con.execute("""
        SELECT
            round(avg(abs(astro_absolute_mag(phot_g_mean_mag, distance_pc) - abs_g_mag)), 6)
                AS mean_abs_diff,
            round(max(abs(astro_absolute_mag(phot_g_mean_mag, distance_pc) - abs_g_mag)), 6)
                AS max_abs_diff,
            count(*) AS n_compared
        FROM gaia
        WHERE abs_g_mag IS NOT NULL
          AND distance_pc > 0
          AND phot_g_mean_mag IS NOT NULL
    """).fetchdf()
    return


# ── 4. Gaia × 2MASS Cross-Match ─────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 4. Gaia × 2MASS Cross-Match

    Nearest-Neighbour-Match via `astro_angular_separation` mit 2 Arcsec Radius.
    """
    )
    return


@app.cell
def crossmatch(con):
    xmatch = con.execute("""
        WITH matches AS (
            SELECT
                g.source_id,
                g.ra AS gaia_ra, g.dec AS gaia_dec,
                g.phot_g_mean_mag AS g_mag,
                t.Jmag, t.Hmag, t.Kmag,
                astro_angular_separation(g.ra, g.dec, t.RAJ2000, t.DEJ2000) * 3600 AS sep_arcsec
            FROM gaia g, twomass t
            WHERE abs(g.ra - t.RAJ2000) < 0.01
              AND abs(g.dec - t.DEJ2000) < 0.01
              AND astro_angular_separation(g.ra, g.dec, t.RAJ2000, t.DEJ2000) < 2.0 / 3600.0
        )
        SELECT * FROM matches
        QUALIFY row_number() OVER (PARTITION BY source_id ORDER BY sep_arcsec) = 1
        ORDER BY sep_arcsec
        LIMIT 200
    """).fetchdf()

    print(f"{len(xmatch)} Matches gefunden")
    print(f"Median Separation: {xmatch['sep_arcsec'].median():.3f} arcsec")
    xmatch.head(10)
    return


# ── 5. Beobachtungsplanung ──────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 5. Was ist jetzt am Himmel?

    Beobachtungsplanung mit `astro_lmst` und `astro_altaz_from_radec` —
    die hellsten Sterne über dem Horizont, jetzt gerade, von Hamburg aus.
    """
    )
    return


@app.cell
def observing_now(con):
    bright_visible = con.execute("""
        WITH obs AS (
            SELECT
                astro_lmst(astro_jd_from_timestamp(now()), 10.0) AS lmst_h,
                53.55 AS lat
        )
        SELECT
            source_id,
            round(ra, 4) AS ra,
            round(dec, 4) AS "dec",
            round(phot_g_mean_mag::DOUBLE, 2) AS g_mag,
            round((astro_altaz_from_radec(ra, dec, (SELECT lmst_h FROM obs), 53.55)).alt_deg, 1) AS altitude,
            round((astro_altaz_from_radec(ra, dec, (SELECT lmst_h FROM obs), 53.55)).az_deg, 1) AS azimuth,
            round(astro_hour_angle(ra, (SELECT lmst_h FROM obs)), 2) AS hour_angle
        FROM gaia
        WHERE (astro_altaz_from_radec(ra, dec, (SELECT lmst_h FROM obs), 53.55)).alt_deg > 20
        ORDER BY phot_g_mean_mag ASC
        LIMIT 25
    """).fetchdf()

    print("Hellste Sterne über Hamburg (Alt > 20°):")
    bright_visible
    return


# ── 6. Galaktische Verteilung ────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 6. Galaktische Koordinaten & 3D-Positionen

    `astro_radec_to_xyz` berechnet kartesische Positionen aus RA/Dec/Distanz.
    """
    )
    return


@app.cell
def galactic_distribution(con, mo):
    import plotly.express as px

    gal = con.execute("""
        SELECT
            l AS gal_lon,
            b AS gal_lat,
            (astro_radec_to_xyz(ra, dec, distance_pc)).x_m AS x_pc,
            (astro_radec_to_xyz(ra, dec, distance_pc)).y_m AS y_pc,
            phot_g_mean_mag
        FROM gaia
        WHERE distance_pc BETWEEN 1 AND 500
          AND distance_pc IS NOT NULL
        USING SAMPLE 30000
    """).fetchdf()

    fig = px.scatter(
        gal,
        x="gal_lon",
        y="gal_lat",
        color="phot_g_mean_mag",
        color_continuous_scale="hot_r",
        range_color=[4, 12],
        labels={
            "gal_lon": "Galaktische Länge l [°]",
            "gal_lat": "Galaktische Breite b [°]",
            "phot_g_mean_mag": "G [mag]",
        },
        title="Gaia DR3 — Galaktische Verteilung (30k Sample)",
        opacity=0.3,
    )
    fig.update_layout(height=400)
    mo.ui.plotly(fig)
    return


if __name__ == "__main__":
    app.run()
