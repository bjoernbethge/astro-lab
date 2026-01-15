"""
Cosmic Web Page - Nutzt echte COMPONENTS
"""

import marimo as mo

# Import der echten COMPONENTS
from astro_lab.ui.components.analyzer import create_analyzer
from astro_lab.ui.components.survey_manager import get_loaded_data


def create_cosmic_web_page():
    """Cosmic Web Page - nutzt echte Components"""

    # Check ob Daten geladen sind
    try:
        loaded_data = get_loaded_data()
        has_data = loaded_data is not None and len(loaded_data) > 0
    except:
        has_data = False

    if not has_data:
        return mo.vstack(
            [
                mo.md("# 🌌 Cosmic Web Analysis"),
                mo.md("⚠️ **Keine Daten geladen!**"),
                mo.md("Gehe zum **Data** Tab und lade Daten."),
                mo.md("Dann komm zurück für Cosmic Web Analyse."),
            ]
        )

    # Verwende echte, funktionierende Components
    analyzer = create_analyzer()

    return mo.vstack(
        [
            mo.md("# 🌌 Cosmic Web Analysis"),
            mo.md("Analysiere kosmische Strukturen in den geladenen Daten."),
            mo.md("---"),
            analyzer,
        ]
    )
