"""
Base Mixins for AstroLab Models
===============================

Core mixin functionality that other mixins can inherit from.
"""

from typing import Any, Dict, Optional



class BaseMixin:
    """Base class for all mixins with common functionality."""

    def get_mixin_config(self) -> Dict[str, Any]:
        """Get configuration specific to this mixin."""
        return {}

    def reset_mixin_state(self):
        """Reset any mixin-specific state."""
        pass


class ConfigurableMixin(BaseMixin):
    """Mixin for components with configurable behavior."""

    def __init__(self):
        self._mixin_config = {}

    def update_mixin_config(self, **kwargs):
        """Update mixin configuration."""
        self._mixin_config.update(kwargs)

    def get_mixin_config(self) -> Dict[str, Any]:
        """Get current mixin configuration."""
        return self._mixin_config.copy()


class StatefulMixin(BaseMixin):
    """Mixin for components that maintain internal state."""

    def __init__(self):
        self._mixin_state = {}

    def update_state(self, key: str, value: Any):
        """Update internal state."""
        self._mixin_state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value."""
        return self._mixin_state.get(key, default)

    def reset_mixin_state(self):
        """Clear all internal state."""
        self._mixin_state.clear()


class CacheableMixin(BaseMixin):
    """Mixin for components with caching capabilities."""

    def __init__(self):
        self._cache = {}
        self._cache_enabled = True

    def enable_cache(self):
        """Enable caching."""
        self._cache_enabled = True

    def disable_cache(self):
        """Disable caching and clear cache."""
        self._cache_enabled = False
        self.clear_cache()

    def clear_cache(self):
        """Clear all cached values."""
        self._cache.clear()

    def get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache if available."""
        if self._cache_enabled:
            return self._cache.get(key)
        return None

    def add_to_cache(self, key: str, value: Any):
        """Add value to cache."""
        if self._cache_enabled:
            self._cache[key] = value

    def reset_mixin_state(self):
        """Reset cache."""
        self.clear_cache()


class CombinedMixin(ConfigurableMixin, StatefulMixin, CacheableMixin):
    """Combined mixin that provides configuration, state, and caching capabilities."""

    def __init__(self):
        ConfigurableMixin.__init__(self)
        StatefulMixin.__init__(self)
        CacheableMixin.__init__(self)

    def get_mixin_config(self) -> Dict[str, Any]:
        """Get combined configuration from all mixins."""
        config = ConfigurableMixin.get_mixin_config(self)
        config.update({
            'cache_enabled': self._cache_enabled,
            'state_keys': list(self._mixin_state.keys()),
            'cache_keys': list(self._cache.keys())
        })
        return config

    def reset_mixin_state(self):
        """Reset state for all mixins."""
        ConfigurableMixin.reset_mixin_state(self)
        StatefulMixin.reset_mixin_state(self)
        CacheableMixin.reset_mixin_state(self)
