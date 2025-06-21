#!/usr/bin/env python3
"""
Test script for Cosmograph integration with AstroLab.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from astro_lab.utils.viz.cosmograph_bridge import CosmographBridge
from astro_lab.data.core import create_cosmic_web_loader

def test_cosmograph_integration():
    """Test Cosmograph integration with AstroLab data."""
    print("🧪 Testing Cosmograph Integration")
    print("=" * 35)
    
    try:
        # Test data loading
        print("📊 Loading test data...")
        results = create_cosmic_web_loader(
            survey="gaia",
            max_samples=100,
            scales_mpc=[5.0, 10.0]
        )
        
        print(f"✅ Loaded {results['n_objects']} objects")
        
        # Test CosmographBridge creation
        print("🌉 Creating CosmographBridge...")
        bridge = CosmographBridge()
        
        print("✅ CosmographBridge created successfully")
        
        # Test visualization creation
        print("🎨 Creating visualization...")
        widget = bridge.from_cosmic_web_results(
            results,
            survey_name="gaia",
            radius=2.0
        )
        
        print("✅ Visualization created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_survey_comparison():
    """Test multi-survey comparison."""
    print("\n🔄 Testing Survey Comparison")
    print("=" * 30)
    
    try:
        bridge = CosmographBridge()
        surveys = ["gaia", "sdss"]
        widgets = []
        
        for survey in surveys:
            print(f"📊 Loading {survey} data...")
            results = create_cosmic_web_loader(
                survey=survey,
                max_samples=50
            )
            
            widget = bridge.from_cosmic_web_results(
                results,
                survey_name=survey
            )
            widgets.append(widget)
            
            print(f"✅ {survey} widget created")
        
        print(f"✅ Created {len(widgets)} survey widgets")
        return True
        
    except Exception as e:
        print(f"❌ Survey comparison failed: {e}")
        return False

if __name__ == "__main__":
    print("🌌 Cosmograph Integration Test Suite")
    print("=" * 45)
    
    # Run tests
    test1_passed = test_cosmograph_integration()
    test2_passed = test_survey_comparison()
    
    # Summary
    print(f"\n📋 Test Results:")
    print(f"   Cosmograph Integration: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Survey Comparison: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 