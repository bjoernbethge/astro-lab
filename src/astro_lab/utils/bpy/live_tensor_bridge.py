"""
Live Tensor Bridge for Blender - DISABLED
=========================================

DISABLED: This module attempted to connect to Blender application,
but we only need Blender API for data types, not live connection.
"""

import gc
import logging

logger = logging.getLogger(__name__)

# DISABLED: No Blender application connection needed
BLENDER_AVAILABLE = False
bpy = None
mathutils = None


class LiveTensorSocketBridge:
    """
    DISABLED: Live tensor bridge disabled.

    We only use Blender API for data types, not live application connection.
    """

    def __init__(self):
        """Initialize disabled bridge."""
        logger.debug("LiveTensorBridge disabled - we only use Blender API for types")
        self._active = False

    def _ensure_handler_is_running(self):
        """Disabled - no handler needed."""
        pass

    def link_tensor_to_socket(self, tensor, obj_name, mod_name, sock_name):
        """Disabled - no socket linking."""
        logger.debug("LiveTensorBridge disabled")
        return False

    def cleanup(self):
        """Disabled - no cleanup needed."""
        pass

    def _update_handler(self, scene, depsgraph):
        """Disabled - no update handler."""
        pass

    def _send_update_signal(self):
        """Disabled - no signals."""
        pass


# DISABLED: No global instance needed
live_bridge = None
