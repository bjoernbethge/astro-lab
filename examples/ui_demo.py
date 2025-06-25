#!/usr/bin/env python3
"""
AstroLab UI Demo
===============

Demonstration der integrierten marimo UI mit dem echten AstroLab Config-System.
Zeigt, wie die UI mit ConfigLoader und data_config arbeitet.
"""

import sys
from pathlib import Path

import marimo as mo

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from astro_lab.ui import (
    handle_component_actions,
    ui_config_loader,
    ui_data_paths,
    ui_model_selector,
    ui_quick_config,
    ui_survey_selector,
    ui_system_status,
)


def create_demo_ui():
    """Erstelle die Demo-UI mit integrierten Config-Komponenten."""

    # Welcome section
    welcome = mo.md("""
    # 🌟 AstroLab UI Demo
    
    Diese Demo zeigt die Integration der marimo UI mit dem echten AstroLab Config-System:
    
    - ✅ Verwendet ConfigLoader statt paralleles Settings-System
    - ✅ Integriert mit data_config für Pfad-Management
    - ✅ Arbeitet mit echten Model- und Training-Configs
    - ✅ Unterstützt Experiment-Management
    """)

    # Configuration section
    config_section = mo.md("## 📂 Configuration Management")
    config_loader = ui_config_loader()
    config_status = ui_system_status()

    # Quick setup section
    quick_section = mo.md("## 🚀 Quick Setup")
    quick_config = ui_quick_config()

    # Data management section
    data_section = mo.md("## 📊 Data Management")
    data_paths = ui_data_paths()
    survey_selector = ui_survey_selector()

    # Model configuration section
    model_section = mo.md("## 🤖 Model Configuration")
    model_selector = ui_model_selector()

    # Combine all components
    components = {
        "config_loader": config_loader,
        "quick_config": quick_config,
        "data_paths": data_paths,
        "survey_selector": survey_selector,
        "model_selector": model_selector,
    }

    # Handle actions
    result = handle_component_actions(components)

    # Display result if any
    result_display = mo.md("")
    if result:
        result_text = f"""
        ## ✅ Action Result
        
        ```python
        {result}
        ```
        """
        result_display = mo.md(result_text)

    # Layout the UI
    layout = mo.vstack(
        [
            welcome,
            config_section,
            mo.hstack([config_loader, config_status]),
            quick_section,
            quick_config,
            data_section,
            mo.hstack([data_paths, survey_selector]),
            model_section,
            model_selector,
            result_display,
        ]
    )

    return layout


def demo_config_integration():
    """Demo der Config-System Integration."""

    info = mo.md("""
    ## 🔧 Config System Integration
    
    Das neue UI-System arbeitet direkt mit den bestehenden AstroLab-Komponenten:
    
    ### ConfigLoader Integration
    - Lädt echte YAML-Configs aus `configs/`
    - Unterstützt Experiment-spezifische Konfigurationen
    - Automatisches Setup von MLflow und Pfaden
    
    ### DataConfig Integration  
    - Verwendet `data_config` für Pfad-Management
    - Erstellt Survey-spezifische Verzeichnisse
    - Verwaltet Experiment-Struktur
    
    ### Model Config Integration
    - Greift auf echte Model-Configs zu (`CONFIGS`)
    - Unterstützt vordefinierte Training-Configs
    - Integriert mit Factory-Funktionen
    
    ### Beispiel-Code:
    ```python
    from astro_lab.ui.settings import ui_config
    
    # Konfiguration laden
    config = ui_config.load_config("configs/gaia_training.yaml", "experiment_1")
    
    # Experiment-Verzeichnisse erstellen
    paths = ui_config.setup_experiment("gaia_stellar_v1")
    
    # Survey-Info abrufen
    survey_info = ui_config.get_survey_info("gaia")
    
    # Model-Config laden
    model_config = get_predefined_config("gaia_classifier")
    ```
    """)

    return info


if __name__ == "__main__":
    print("🌟 AstroLab UI Demo")
    print("=" * 50)

    # Show config system info
    print("\n📂 Config System Status:")

    try:
        from astro_lab.ui.settings import ui_config

        # Show available configs
        configs = ui_config.available_configs
        print(f"Available surveys: {configs['surveys']}")
        print(f"Available models: {configs['models']}")
        print(f"Available training configs: {configs['training']}")

        # Show data config
        data_config = ui_config.get_data_config()
        print(f"Data directory: {data_config['base_dir']}")

    except Exception as e:
        print(f"❌ Error loading config system: {e}")

    print("\n🚀 Starting marimo UI...")
    print("Run this file with: marimo run examples/ui_demo.py")

    # Create and return the UI for marimo
    demo_ui = create_demo_ui()
    config_info = demo_config_integration()

    # Combine everything
    app = mo.vstack(
        [
            demo_ui,
            mo.md("---"),
            config_info,
        ]
    )
