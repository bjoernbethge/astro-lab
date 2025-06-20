"""
Bidirectional Live Tensor <-> Blender Node Socket Bridge
========================================================

Provides a mechanism to create a bidirectional link between a PyTorch tensor
in Python and an input socket on a Geometry Nodes modifier in Blender.
Changes are propagated in real-time in both directions.

- Python -> Blender: Uses a `depsgraph_update_post` handler.
- Blender -> Python: Uses the `bpy.msgbus` for event-based updates.
"""

from typing import Dict, Any, Tuple, Callable
import torch
import mathutils

try:
    import bpy
    from bpy.app.handlers import depsgraph_update_post
    BLENDER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    BLENDER_AVAILABLE = False
    bpy = None
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

        if BLENDER_AVAILABLE:
            self._ensure_handler_is_running()

    def link(
        self,
        tensor: torch.Tensor,
        obj_name: str,
        socket_name: str,
        modifier_name: str = "GeometryNodes",
    ):
        """
        Create a live, bidirectional link between a tensor and a Blender socket.
        """
        if not BLENDER_AVAILABLE:
            print("⚠️ Blender not available. Cannot link tensor.")
            return

        key = (obj_name, modifier_name, socket_name)
        if key in self._linked_sockets:
            print(f"🔗 Link for '{obj_name}.{socket_name}' already exists.")
            return

        # 1. Initial value sync (Python -> Blender)
        initial_value = self._get_tensor_value(tensor)
        if initial_value is None:
            return
        self._set_socket_value(obj_name, modifier_name, socket_name, initial_value)

        # 2. Store the link information
        self._linked_sockets[key] = {
            "tensor": tensor,
            "last_value": initial_value
        }

        # 3. Subscribe to Blender UI changes (Blender -> Python)
        self._subscribe_to_socket(key)

        print(f"🔗<=>🔗 Bidirectional link created for '{obj_name}.{socket_name}'.")
        self._ensure_handler_is_running()

    def unlink(self, obj_name: str, socket_name: str, modifier_name: str = "GeometryNodes"):
        """Remove a live link."""
        key = (obj_name, modifier_name, socket_name)
        if key in self._linked_sockets:
            del self._linked_sockets[key]
            self._unsubscribe_from_socket(key)
            print(f"🚫 Unlinked tensor from '{obj_name}.{socket_name}'.")
        
        if not self._linked_sockets:
            self._remove_handler()

    def _subscribe_to_socket(self, key: Tuple[str, str, str]):
        """Use msgbus to listen for changes on a modifier socket."""
        obj_name, modifier_name, socket_name = key
        
        obj = bpy.data.objects.get(obj_name)
        if not obj: return
        mod = obj.modifiers.get(modifier_name)
        if not mod: return

        # Construct the data path for the subscription
        path = f'modifiers["{modifier_name}"]["{socket_name}"]'

        # Create a unique callback for this specific key
        callback = lambda: self._blender_to_python_update(key)
        
        bpy.msgbus.subscribe_rna(
            key=(mod, path),
            owner=self,
            args=(),
            notify=callback,
        )
        self._subscriptions[key] = (mod, path, callback)

    def _unsubscribe_from_socket(self, key: Tuple[str, str, str]):
        """Remove a msgbus subscription."""
        if key in self._subscriptions:
            bpy.msgbus.clear_by_owner(self)
            del self._subscriptions[key]

    def _blender_to_python_update(self, key: Tuple[str, str, str]):
        """Callback triggered by msgbus when a Blender property changes."""
        if self._is_updating:
            return # Prevent feedback loop

        obj_name, modifier_name, socket_name = key
        link_data = self._linked_sockets.get(key)
        if not link_data:
            return

        self._is_updating = True
        try:
            new_value = self._get_socket_value(obj_name, modifier_name, socket_name)
            tensor = link_data["tensor"]
            
            if new_value is not None:
                # Update tensor in-place
                if tensor.numel() == 1:
                    tensor.fill_(new_value)
                else: # Handle vectors/colors
                    tensor_val = torch.tensor(new_value, dtype=tensor.dtype, device=tensor.device)
                    tensor.data = tensor_val.data

                # Update our last known value to prevent Python->Blender echo
                link_data["last_value"] = self._get_tensor_value(tensor)
        finally:
            self._is_updating = False
            
    def _set_socket_value(self, obj_name, modifier_name, socket_name, value):
        """Safely sets the value of a geometry node input socket."""
        obj = bpy.data.objects.get(obj_name)
        if not obj: return
        mod = obj.modifiers.get(modifier_name)
        if not mod: return
        try:
            if socket_name in mod:
                mod[socket_name] = value
        except (KeyError, AttributeError, TypeError):
            pass

    def _get_socket_value(self, obj_name, modifier_name, socket_name):
        """Safely gets the value of a geometry node input socket."""
        obj = bpy.data.objects.get(obj_name)
        if not obj: return None
        mod = obj.modifiers.get(modifier_name)
        if not mod: return None
        try:
            value = mod.get(socket_name)
            # Convert Blender math types to lists
            if isinstance(value, (mathutils.Vector, mathutils.Color, mathutils.Euler)):
                return list(value)
            return value
        except (KeyError, AttributeError):
            return None

    @staticmethod
    def _get_tensor_value(tensor: torch.Tensor):
        """Extracts a Python scalar or list from a tensor."""
        try:
            if tensor.numel() == 1:
                return tensor.item()
            else:
                return tensor.detach().cpu().tolist()
        except Exception:
            return None

    def _python_to_blender_update(self):
        """
        The core update function that runs on scene changes (depsgraph).
        Checks for changes in Python tensors and updates Blender.
        """
        if self._is_updating:
            return # Prevent feedback loop

        self._is_updating = True
        try:
            for key, data in list(self._linked_sockets.items()):
                obj_name, modifier_name, socket_name = key
                tensor = data["tensor"]
                last_value = data["last_value"]
                current_value = self._get_tensor_value(tensor)
                
                # Check for change (handles scalars and lists with tolerance)
                is_different = False
                if isinstance(current_value, list):
                    if not isinstance(last_value, list) or len(current_value) != len(last_value):
                        is_different = True
                    else:
                        for v1, v2 in zip(current_value, last_value):
                            if not math.isclose(v1, v2, rel_tol=1e-6):
                                is_different = True
                                break
                elif not math.isclose(current_value, last_value, rel_tol=1e-6):
                    is_different = True

                if is_different:
                    self._set_socket_value(obj_name, modifier_name, socket_name, current_value)
                    self._linked_sockets[key]["last_value"] = current_value
        finally:
            self._is_updating = False

    def _ensure_handler_is_running(self):
        """Make sure the update handler is registered and running."""
        global _update_handler_wrapper
        if not self._handler_active:
            # Create a wrapper that can be stored and removed reliably
            _update_handler_wrapper = lambda scene: self._python_to_blender_update()
            depsgraph_update_post.append(_update_handler_wrapper)
            self._handler_active = True
            print("📈 Live tensor bridge handler started.")

    def _remove_handler(self):
        """Remove the update handler if it's running."""
        global _update_handler_wrapper
        if self._handler_active and _update_handler_wrapper in depsgraph_update_post:
            depsgraph_update_post.remove(_update_handler_wrapper)
            self._handler_active = False
            print("📉 Live tensor bridge handler stopped.")

# Global state for the single instance and its handler
_update_handler_wrapper = None
live_bridge = LiveTensorSocketBridge() if BLENDER_AVAILABLE else None 