"""
AlbPy Demo: Astronomische Szene mit Operatoren
==============================================

Dieses Skript erzeugt eine attraktive astronomische Szene:
- Mehrere Galaxien (verschiedene Typen, Positionen, Farben)
- Einen Sternhaufen/Pointcloud
- Kosmische Filamente
- Optimale Kamera/Licht-Position
- Schönes Compositing (Vignette, Glow)

Ausführung (in Blender-Konsole oder mit blender --background --python test_render.py):
Das Ergebnisbild wird als test_render_output.png gespeichert.
"""

import math
import os
import random
import sys

import bpy

import src.astro_lab.widgets.albpy as albpy

albpy.register()

# Szene aufräumen
bpy.ops.wm.read_factory_settings(use_empty=True)

# Kamera hinzufügen
cam = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj
cam_obj.location = (0, -40, 18)
cam_obj.rotation_euler = (math.radians(65), 0, 0)

# Licht hinzufügen (Sun + Soft Fill)
light_data = bpy.data.lights.new(name="KeyLight", type="SUN")
light_obj = bpy.data.objects.new(name="KeyLight", object_data=light_data)
bpy.context.scene.collection.objects.link(light_obj)
light_obj.location = (0, -30, 40)
light_data.energy = 5

fill_light_data = bpy.data.lights.new(name="FillLight", type="AREA")
fill_light_obj = bpy.data.objects.new(name="FillLight", object_data=fill_light_data)
bpy.context.scene.collection.objects.link(fill_light_obj)
fill_light_obj.location = (20, -10, 10)
fill_light_data.energy = 2
fill_light_data.size = 20

print("\n🌟 Erzeuge Stellar Showcase...")
try:
    bpy.ops.albpy.create_stellar_showcase(spacing=5.0, scale=1.5)
    print("✅ Stellar Showcase erfolgreich erstellt")
except Exception as e:
    print(f"❌ Fehler beim Erstellen des Stellar Showcase: {e}")

# Compositing-Node-Setup (Vignette + Glow)
bpy.context.scene.use_nodes = True
comp_tree = bpy.context.scene.node_tree
comp_tree.nodes.clear()
rl = comp_tree.nodes.new("CompositorNodeRLayers")
vignette_ng = bpy.data.node_groups.get("ALBPY_NG_Vignette")
glow_ng = bpy.data.node_groups.get("ALBPY_NG_StarGlow")
comp_out = comp_tree.nodes.new("CompositorNodeComposite")

last_node = rl
if vignette_ng:
    vignette_node = comp_tree.nodes.new("CompositorNodeGroup")
    vignette_node.node_tree = vignette_ng
    comp_tree.links.new(last_node.outputs[0], vignette_node.inputs[0])
    last_node = vignette_node
if glow_ng:
    glow_node = comp_tree.nodes.new("CompositorNodeGroup")
    glow_node.node_tree = glow_ng
    comp_tree.links.new(last_node.outputs[0], glow_node.inputs[0])
    last_node = glow_node
comp_tree.links.new(last_node.outputs[0], comp_out.inputs[0])

# Render-Settings
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 128
scene.render.resolution_x = 1600
scene.render.resolution_y = 1200
scene.render.filepath = os.path.abspath("test_render_output.png")

# Rendern
print("\n🎬 Rendering...")
bpy.ops.render.render(write_still=True)
print(f"✅ Rendered image saved as {scene.render.filepath}")

# Objekte in der Szene anzeigen
print("\n📊 Objekte in der Szene:")
for obj in bpy.context.scene.objects:
    print(f"  - {obj.name} ({obj.type})")

print("\n🎉 Demo abgeschlossen!")
