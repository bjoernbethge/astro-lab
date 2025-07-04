"""
Showcase Operators for AlbPy
===========================

Provides operators for creating astronomical showcases using Node Groups.
"""

from . import (
    nebula_showcase,
    planetary_system,
    stellar_showcase,
)


def register():
    """Register all showcase operators."""
    stellar_showcase.register()
    planetary_system.register()
    nebula_showcase.register()


def unregister():
    """Unregister all showcase operators."""
    nebula_showcase.unregister()
    planetary_system.unregister()
    stellar_showcase.unregister()
