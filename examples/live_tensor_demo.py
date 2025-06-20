"""
Live Tensor to Blender Socket Demo
==================================

This script demonstrates the power of the `LiveTensorSocketBridge`.
It creates a procedural object in Blender and then links PyTorch tensors
to its Geometry Node inputs. By changing the tensor values in Python,
the Blender object is updated in real-time.

**To see this in action:**
1. Open Blender.
2. Run this script from your IDE or the command line.
3. Observe the created 'ProceduralGalaxy' object in Blender.
4. Watch the console output as the script modifies the tensors.
   The galaxy in Blender will change its size and star density accordingly.
"""

import torch
import time
import math
from pathlib import Path
import sys

# Add project root to Python path
try:
    from src.astro_lab.widgets.astro_lab import AstroLabWidget
except ImportError:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))
    from src.astro_lab.widgets.astro_lab import AstroLabWidget

def run_live_demo(widget: AstroLabWidget):
    """Executes the live tensor demonstration."""
    print("\n--- Running Live Tensor to Blender Socket Demo ---")

    # 1. Create a scene with a procedural object
    print("Step 1: Creating a procedural galaxy in Blender.")
    widget.al.core['reset_scene']()
    galaxy_obj = widget.al.advanced.create_procedural_galaxy(
        galaxy_type='spiral',
        num_stars=50000,
        radius=20.0
    )
    
    if galaxy_obj is None:
        print("❌ Could not create galaxy object. Aborting demo.")
        print("   Make sure you are running Blender 2.93+ with the 'Node: Is Shader' patch or Blender 3.0+.")
        return
        
    obj_name = galaxy_obj.name
    print(f"   - Created object: '{obj_name}'")

    # 2. Define the tensors in Python that will control the object
    print("Step 2: Creating PyTorch control tensors.")
    
    # We start these tensors with the initial values of the galaxy
    radius_tensor = torch.tensor(20.0, dtype=torch.float32)
    
    # Let's control the 'Arm Rotation' property
    arm_rotation_tensor = torch.tensor(0.5, dtype=torch.float32)

    # 3. Link the tensors to the Geometry Node sockets
    print("Step 3: Linking tensors to Blender's node sockets.")
    
    # Note: You can find the socket identifier by hovering over the input
    # in the Geometry Nodes modifier panel in Blender. It's often "Input_X"
    # where X is the position of the input.
    # For our 'create_procedural_galaxy' function:
    # - "Radius" is Input_2
    # - "Arm Rotation" is Input_5
    
    widget.al.live_bridge.link(radius_tensor, obj_name, "Input_2")
    widget.al.live_bridge.link(arm_rotation_tensor, obj_name, "Input_5")

    # 4. Animate the tensor values in a loop
    print("\nStep 4: Starting live animation loop (Python -> Blender). Press Ctrl+C to stop.")
    print("      (Watch the 'ProceduralGalaxy' object in Blender!)")
    print("\n✨ TRY THIS: While the animation is running, manually change the 'Radius' or")
    print("      'Arm Rotation' values in Blender's Geometry Nodes modifier panel.")
    print("      You will see your change for a moment before the script overwrites it.\n")
    
    start_time = time.time()
    try:
        # Run for a fixed duration for automated testing (e.g., 10 seconds)
        print("Running Python -> Blender animation for 10 seconds...")
        for _ in range(200): # 200 * 0.05s = 10s
            elapsed_time = time.time() - start_time
            
            # Python updates the tensor -> Blender gets updated
            new_radius = 20.0 + 5.0 * math.sin(elapsed_time * 0.5)
            radius_tensor.fill_(new_radius)
            
            new_rotation = 0.5 + elapsed_time * 0.1
            arm_rotation_tensor.fill_(new_rotation)

            # Read back the value from the tensor to show it's consistent
            # If you changed it in Blender, you'd see the change here for one frame
            r_val = radius_tensor.item()
            rot_val = arm_rotation_tensor.item()
            print(f"\rPython -> Blender | Radius: {r_val:.2f}, Arm Rotation: {rot_val:.2f}   ", end="")
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\nAnimation stopped by user.")
    
    print("\n\nPython -> Blender animation finished.")

    # 5. Demonstrate Blender -> Python sync
    print("\nStep 5: Demonstrating Blender -> Python synchronization.")
    print("      You now have 15 seconds to freely change the values in Blender.")
    print("      Watch the console to see the Python tensor update in real-time.\n")

    try:
        for i in range(150): # ~15 seconds
            r_val = radius_tensor.item()
            rot_val = arm_rotation_tensor.item()
            print(f"\rBlender -> Python | Radius: {r_val:.2f}, Arm Rotation: {rot_val:.2f}   ", end="")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass # Allow skipping this part
        
    finally:
        # 6. Unlink the tensors to stop the updates
        print("\n\nStep 6: Unlinking tensors.")
        widget.al.live_bridge.unlink(obj_name, "Input_2")
        widget.al.live_bridge.unlink(obj_name, "Input_5")
        print("✅ Demo finished.")


if __name__ == "__main__":
    print("Initializing AstroLabWidget for Live Demo...")
    widget = AstroLabWidget()

    if widget.blender_available():
        run_live_demo(widget)
    else:
        print("\n❌ Blender API not available.")
        print("   This demo requires a running Blender instance and the 'bpy' module.") 