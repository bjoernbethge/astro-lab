# cycles

Part of `bpy.ops`
Module: `bpy.ops.cycles`

## Operators (3)

### `denoise_animation`

bpy.ops.cycles.denoise_animation(input_filepath="", output_filepath="")
Denoise rendered animation sequence using current scene and view layer settings. Requires denoising data passes and output to OpenEXR multilayer files

### `merge_images`

bpy.ops.cycles.merge_images(input_filepath1="", input_filepath2="", output_filepath="")
Combine OpenEXR multi-layer images rendered with different sample ranges into one image with reduced noise

### `use_shading_nodes`

bpy.ops.cycles.use_shading_nodes()
Enable nodes on a material, world or light
