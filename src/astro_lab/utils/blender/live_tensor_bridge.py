"""
Bidirectional Live Tensor <-> Blender Node Socket Bridge
========================================================

Provides a mechanism to create a bidirectional link between a PyTorch tensor
in Python and an input socket on a Geometry Nodes modifier in Blender.
Changes are propagated in real-time in both directions.

- Python -> Blender: Uses a `depsgraph_update_post` handler.
- Blender -> Python: Uses the `bpy.msgbus` for event-based updates.
"""

import math
from typing import Dict, Any, Tuple, Callable
import torch

# Use centralized bpy import
from . import bpy, mathutils

# Import bpy components if available
if bpy is not None:
    from bpy.app.handlers import depsgraph_update_post
else:
    depsgraph_update_post = None


class LiveTensorSocketBridge:
    """
    Manages live, bidirectional links between PyTorch tensors and Blender sockets.
    """
    def __init__(self):
        self._handler_active = False
        self._is_updating = False  # Lock to prevent update loops
        # Format: { (obj, mod, sock): {'tensor': tensor, 'last_value': value} }
        self._linked_sockets: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        # Format: { (obj, mod, sock): (owner, path, callback) }
        self._subscriptions: Dict[Tuple[str, str, str], Tuple[Any, str, Callable]] = {}

        if bpy is not None:
            self._ensure_handler_is_running()

    def _ensure_handler_is_running(self):
        """Ensure the update handler is registered."""
        if self._handler_active or depsgraph_update_post is None:
            return

        # Register the handler
        depsgraph_update_post.append(self._update_handler)  # type: ignore
        self._handler_active = True

    def _update_handler(self, scene):  # type: ignore
        """Handle depsgraph updates to sync tensor changes to Blender."""
        if self._is_updating:
            return

        self._is_updating = True
        try:
            for (obj_name, mod_name, sock_name), data in self._linked_sockets.items():
                tensor = data['tensor']
                last_value = data['last_value']
                
                # Check if tensor has changed
                if not torch.equal(tensor, last_value):
                    self._update_socket_value(obj_name, mod_name, sock_name, tensor)
                    data['last_value'] = tensor.clone()
        finally:
            self._is_updating = False

    def _update_socket_value(self, obj_name: str, mod_name: str, sock_name: str, tensor: torch.Tensor):
        """Update a specific socket value."""
        try:
            obj = bpy.data.objects.get(obj_name)
            if obj is None:
                return

            mod = obj.modifiers.get(mod_name)
            if mod is None or mod.type != 'NODES':
                return

            # Find the socket
            for sock in mod.node_group.inputs:
                if sock.name == sock_name:
                    # Convert tensor to appropriate type
                    if tensor.dim() == 0:  # Scalar
                        sock.default_value = float(tensor.item())
                    elif tensor.dim() == 1:  # Vector
                        sock.default_value = tuple(tensor.tolist())
                    elif tensor.dim() == 2:  # Matrix
                        sock.default_value = [tuple(row) for row in tensor.tolist()]
                    break
        except Exception as e:
            print(f"Error updating socket {sock_name}: {e}")

    def link_tensor_to_socket(self, tensor: torch.Tensor, obj_name: str, mod_name: str, sock_name: str):
        """
        Create a live, bidirectional link between a tensor and a Blender socket.
        """
        if bpy is None:
            print("⚠️ Blender not available. Cannot link tensor.")
            return

        key = (obj_name, mod_name, sock_name)
        self._linked_sockets[key] = {
            'tensor': tensor,
            'last_value': tensor.clone()
        }

        # Subscribe to socket changes
        self._subscribe_to_socket_changes(obj_name, mod_name, sock_name)

    def _subscribe_to_socket_changes(self, obj_name: str, mod_name: str, sock_name: str):
        """Subscribe to socket changes using bpy.msgbus."""
        if bpy is None or not hasattr(bpy, 'msgbus'):
            return

        try:
            obj = bpy.data.objects.get(obj_name)
            if obj is None:
                return

            mod = obj.modifiers.get(mod_name)
            if mod is None:
                return

            # Subscribe to socket changes
            bpy.msgbus.subscribe_rna(  # type: ignore
                key=mod.node_group.inputs[sock_name],
                owner=self,
                args=(obj_name, mod_name, sock_name),
                notify=self._on_socket_changed
            )
        except Exception as e:
            print(f"Error subscribing to socket changes: {e}")

    def _on_socket_changed(self, obj_name: str, mod_name: str, sock_name: str):
        """Handle socket changes from Blender."""
        if self._is_updating:
            return

        try:
            obj = bpy.data.objects.get(obj_name)
            if obj is None:
                return

            mod = obj.modifiers.get(mod_name)
            if mod is None:
                return

            # Get socket value
            sock = mod.node_group.inputs.get(sock_name)
            if sock is None:
                return

            # Update tensor
            key = (obj_name, mod_name, sock_name)
            if key in self._linked_sockets:
                data = self._linked_sockets[key]
                tensor = data['tensor']
                
                # Convert socket value to tensor
                if hasattr(sock, 'default_value'):
                    value = sock.default_value
                    if isinstance(value, (int, float)):
                        tensor.fill_(value)
                    elif isinstance(value, (list, tuple)):
                        new_tensor = torch.tensor(value, dtype=tensor.dtype, device=tensor.device)
                        tensor.copy_(new_tensor)
        except Exception as e:
            print(f"Error handling socket change: {e}")

    def unlink_tensor(self, obj_name: str, mod_name: str, sock_name: str):
        """Remove a tensor-socket link."""
        key = (obj_name, mod_name, sock_name)
        if key in self._linked_sockets:
            del self._linked_sockets[key]

        # Unsubscribe from changes
        if bpy is not None and hasattr(bpy, 'msgbus'):
            try:
                bpy.msgbus.clear_by_owner(self)  # type: ignore
            except Exception as e:
                print(f"Error unsubscribing from socket changes: {e}")

    def cleanup(self):
        """Clean up all links and handlers."""
        self._linked_sockets.clear()
        
        if bpy is not None and depsgraph_update_post is not None:
            try:
                if self._update_handler in depsgraph_update_post:  # type: ignore
                    depsgraph_update_post.remove(self._update_handler)  # type: ignore
                self._handler_active = False
            except Exception as e:
                print(f"Error cleaning up handler: {e}")


# Global state for the single instance and its handler
_update_handler_wrapper = None
live_bridge = LiveTensorSocketBridge() if bpy is not None else None 