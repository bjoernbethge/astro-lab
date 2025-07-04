#!/usr/bin/env python3
"""
Integrations-Test für das aufgeräumte AstroLab Training System
============================================================

Testet die vollständige Integration zwischen:
- Models Module (aufgeräumt)
- Training Module
- Hardware Detection
- Memory Management

Dieser Test stellt sicher, dass alle Komponenten nahtlos zusammenarbeiten.
"""

import logging
import sys
import traceback
from pathlib import Path

# Add src to path for testing
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_basic_imports():
    """Test 1: Grundlegende Import-Validierung"""
    logger.info("🔍 Test 1: Teste grundlegende Imports...")

    try:
        # Models imports
        from astro_lab.models import AstroModel

        logger.info("✅ Models imports erfolgreich")

        # Training imports
        from astro_lab.training import AstroTrainer

        logger.info("✅ Training imports erfolgreich")

        # Core dependencies
        from astro_lab.config import get_training_config
        from astro_lab.memory import clear_cuda_cache, get_memory_info

        logger.info("✅ Core dependencies imports erfolgreich")

        return True

    except ImportError as e:
        logger.error(f"❌ Import-Fehler: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unerwarteter Fehler: {e}")
        return False


def test_hardware_detection():
    """Test 2: Hardware-Erkennung"""
    logger.info("🔍 Test 2: Teste Hardware-Erkennung...")

    try:
        from astro_lab.training import detect_hardware, setup_device

        # Hardware Detection
        hardware_info = detect_hardware()
        logger.info("Hardware erkannt:")
        logger.info(f"  CUDA verfügbar: {hardware_info['cuda_available']}")
        logger.info(f"  GPU Count: {hardware_info['cuda_device_count']}")
        if hardware_info["cuda_available"]:
            logger.info(f"  GPU: {hardware_info['cuda_device_name']}")
            logger.info(f"  VRAM: {hardware_info['cuda_memory_gb']:.1f}GB")
            logger.info(
                f"  Empfohlene Batch Size: {hardware_info['recommended_batch_size']}"
            )

        # Device Setup
        device = setup_device()
        logger.info(f"  Device: {device}")

        logger.info("✅ Hardware-Erkennung erfolgreich")
        return True

    except Exception as e:
        logger.error(f"❌ Hardware-Erkennungs-Fehler: {e}")
        return False


def test_model_factory():
    """Test 3: Model Factory"""
    logger.info("🔍 Test 3: Teste Model Factory...")

    try:
        from astro_lab.models import (
            create_cosmic_web_model,
            create_galaxy_model,
            create_stellar_model,
        )

        # Liste verfügbare Modelle
        available_models = get_available_models()
        logger.info(f"Verfügbare Modelle: {available_models[:5]}...")  # Erste 5

        # Teste verschiedene Modell-Erstellungen
        test_cases = [
            {
                "model_type": "graph",
                "num_features": 7,
                "num_classes": 3,
                "hidden_dim": 64,
            },
            {
                "model_type": "node",
                "num_features": 10,
                "num_classes": 5,
                "hidden_dim": 128,
            },
            {
                "model_type": "point_cloud",
                "num_features": 6,
                "num_classes": 4,
                "scale": "small",
            },
            {
                "model_type": "cosmic_web",
                "num_features": 7,
                "num_classes": 2,
                "multi_scale": True,
            },
        ]

        for i, case in enumerate(test_cases):
            logger.info(f"  Teste Modell {i + 1}: {case['model_type']}")
            model = create_model(**case)
            logger.info(f"    ✓ Erstellt: {type(model).__name__}")

            # Test model summary
            if hasattr(model, "get_model_summary"):
                summary = model.get_model_summary()
                logger.info(f"    ✓ Parameter: {summary['total_parameters']:,}")

        logger.info("✅ Model Factory erfolgreich")
        return True

    except Exception as e:
        logger.error(f"❌ Model Factory Fehler: {e}")
        traceback.print_exc()
        return False


def test_training_setup():
    """Test 4: Training Setup ohne echte Daten"""
    logger.info("🔍 Test 4: Teste Training Setup...")

    try:
        from astro_lab.config import get_training_config
        from astro_lab.training import AstroTrainer

        #
        config = get_training_config()

        # Trainer erstellen
        logger.info("  Erstelle AstroTrainer...")
        trainer = AstroTrainer(config)
        logger.info("  ✓ AstroTrainer erstellt")

        # Config validieren
        trainer_config = trainer.get_config()
        logger.info(f"  ✓ Config validiert: {len(trainer_config)} Parameter")

        logger.info("✅ Training Setup erfolgreich")
        return True

    except Exception as e:
        logger.error(f"❌ Training Setup Fehler: {e}")
        traceback.print_exc()
        return False


def test_memory_management():
    """Test 6: Memory Management"""
    logger.info("🔍 Test 6: Teste Memory Management...")

    try:
        import torch

        from astro_lab.memory import (
            clear_cuda_cache,
            get_memory_info,
            memory_management,
        )

        if torch.cuda.is_available():
            # Memory Info
            memory_info = get_memory_info()
            logger.info(f"  GPU: {memory_info['device_name']}")
            logger.info(f"  Total VRAM: {memory_info['memory_total_gb']:.1f}GB")
            logger.info(f"  Allocated: {memory_info['memory_allocated_gb']:.3f}GB")

            # Test Memory Management Context
            with memory_management():
                # Erstelle temporäre Tensoren
                temp_tensor = torch.randn(1000, 1000, device="cuda")
                logger.info("    ✓ Memory Context Manager funktioniert")

            # Clear Cache
            clear_cuda_cache()
            logger.info("    ✓ CUDA Cache geleert")

        else:
            logger.info("  CPU-only Modus - Memory Management begrenzt verfügbar")

        logger.info("✅ Memory Management erfolgreich")
        return True

    except Exception as e:
        logger.error(f"❌ Memory Management Fehler: {e}")
        return False


def test_component_compatibility():
    """Test 7: Komponenten-Kompatibilität"""
    logger.info("🔍 Test 7: Teste Komponenten-Kompatibilität...")

    try:
        from astro_lab.models.components import (
            EnhancedMLPBlock,
            ModernGraphEncoder,
            create_output_head,
        )
        from astro_lab.models.layers import (
            FlexibleGraphConv,
            MultiScalePooling,
        )

        # Teste Komponenten-Erstellung
        mlp = EnhancedMLPBlock(in_dim=64, out_dim=32)
        logger.info("  ✓ EnhancedMLPBlock erstellt")

        # Teste Output Head Factory
        head = create_output_head("classification", input_dim=64, output_dim=5)
        logger.info("  ✓ Output Head erstellt")

        logger.info("✅ Komponenten-Kompatibilität erfolgreich")
        return True

    except Exception as e:
        logger.error(f"❌ Komponenten-Kompatibilität Fehler: {e}")
        traceback.print_exc()
        return False


def run_integration_tests():
    """Führe alle Integrations-Tests aus"""
    logger.info("🚀 Starte AstroLab Training Integration Tests")
    logger.info("=" * 60)

    tests = [
        ("Basic Imports", test_basic_imports),
        ("Hardware Detection", test_hardware_detection),
        ("Model Factory", test_model_factory),
        ("Training Setup", test_training_setup),
        ("Memory Management", test_memory_management),
        ("Component Compatibility", test_component_compatibility),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))

        logger.info("-" * 40)

    # Zusammenfassung
    passed = sum(1 for _, result in results if result)
    total = len(results)

    logger.info("=" * 60)
    logger.info("📊 TEST ZUSAMMENFASSUNG")
    logger.info("=" * 60)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:<25} {status}")

    logger.info("-" * 60)
    logger.info(f"Gesamt: {passed}/{total} Tests bestanden")

    if passed == total:
        logger.info("🎉 ALLE TESTS ERFOLGREICH!")
        logger.info(
            "✅ Das aufgeräumte AstroLab System ist vollständig funktionsfähig!"
        )
        return True
    else:
        logger.error(f"💥 {total - passed} Tests fehlgeschlagen!")
        logger.error("❌ Integration benötigt weitere Arbeit.")
        return False


if __name__ == "__main__":
    """Hauptfunktion für direkten Test-Aufruf"""
    print("🌌 AstroLab Training Integration Test")
    print("Testing integration between cleaned models and training modules")
    print("=" * 80)

    success = run_integration_tests()

    if success:
        print("\n🎯 ERFOLG: Alle Integrations-Tests bestanden!")
        print("Das aufgeräumte AstroLab System ist bereit für den Einsatz.")
        sys.exit(0)
    else:
        print("\n💥 FEHLER: Einige Tests sind fehlgeschlagen!")
        print("Bitte die Fehlermeldungen überprüfen und beheben.")
        sys.exit(1)
