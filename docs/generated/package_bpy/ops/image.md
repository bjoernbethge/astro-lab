# image

Part of `bpy.ops`
Module: `bpy.ops.image`

## Operators (48)

### `add_render_slot`

bpy.ops.image.add_render_slot()
Add a new render slot

### `change_frame`

bpy.ops.image.change_frame(frame=0)
Interactively change the current frame number

### `clear_render_border`

bpy.ops.image.clear_render_border()
Clear the boundaries of the render region and disable render region

### `clear_render_slot`

bpy.ops.image.clear_render_slot()
Clear the currently selected render slot

### `clipboard_copy`

bpy.ops.image.clipboard_copy()
Copy the image to the clipboard

### `clipboard_paste`

bpy.ops.image.clipboard_paste()
Paste new image from the clipboard

### `convert_to_mesh_plane`

bpy.ops.image.convert_to_mesh_plane(interpolation='Linear', extension='CLIP', alpha_mode='STRAIGHT', use_auto_refresh=True, relative=True, shader='PRINCIPLED', emit_strength=1, use_transparency=True, render_method='DITHERED', use_backface_culling=False, show_transparent_back=True, overwrite_material=True, name_from='OBJECT', delete_ref=True)
Convert selected reference images to textured mesh plane

### `curves_point_set`

bpy.ops.image.curves_point_set(point='BLACK_POINT', size=1)
Set black point or white point for curves

### `cycle_render_slot`

bpy.ops.image.cycle_render_slot(reverse=False)
Cycle through all non-void render slots

### `external_edit`

bpy.ops.image.external_edit(filepath="")
Edit image in an external application

### `file_browse`

bpy.ops.image.file_browse(filepath="", hide_props_region=True, check_existing=False, filter_blender=False, filter_backup=False, filter_image=True, filter_movie=True, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, relative_path=True, show_multiview=False, use_multiview=False, display_type='DEFAULT', sort_method='DEFAULT')
Open an image file browser, hold Shift to open the file, Alt to browse containing directory

### `flip`

bpy.ops.image.flip(use_flip_x=False, use_flip_y=False)
Flip the image

### `import_as_mesh_planes`

bpy.ops.image.import_as_mesh_planes(interpolation='Linear', extension='CLIP', alpha_mode='STRAIGHT', use_auto_refresh=True, relative=True, shader='PRINCIPLED', emit_strength=1, use_transparency=True, render_method='DITHERED', use_backface_culling=False, show_transparent_back=True, overwrite_material=True, filepath="", align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), files=[], directory="", filter_image=True, filter_movie=True, filter_folder=True, force_reload=False, image_sequence=False, offset=True, offset_axis='+X', offset_amount=0.1, align_axis='CAM_AX', prev_align_axis='NONE', align_track=False, size_mode='ABSOLUTE', fill_mode='FILL', height=1, factor=600)
Create mesh plane(s) from image files with the appropriate aspect ratio

### `invert`

bpy.ops.image.invert(invert_r=False, invert_g=False, invert_b=False, invert_a=False)
Invert image's channels

### `match_movie_length`

bpy.ops.image.match_movie_length()
Set image's user's length to the one of this video

### `new`

bpy.ops.image.new(name="Untitled", width=1024, height=1024, color=(0, 0, 0, 1), alpha=True, generated_type='BLANK', float=False, use_stereo_3d=False, tiled=False)
Create a new image

### `open`

bpy.ops.image.open(allow_path_tokens=True, filepath="", directory="", files=[], hide_props_region=True, check_existing=False, filter_blender=False, filter_backup=False, filter_image=True, filter_movie=True, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, relative_path=True, show_multiview=False, use_multiview=False, display_type='DEFAULT', sort_method='DEFAULT', use_sequence_detection=True, use_udim_detecting=True)
Open image

### `open_images`

bpy.ops.image.open_images(directory="", files=[], relative_path=True, use_sequence_detection=True, use_udim_detection=True)
(undocumented operator)

### `pack`

bpy.ops.image.pack()
Pack an image as embedded data into the .blend file

### `project_apply`

bpy.ops.image.project_apply()
Project edited image back onto the object

### `project_edit`

bpy.ops.image.project_edit()
Edit a snapshot of the 3D Viewport in an external image editor

### `read_viewlayers`

bpy.ops.image.read_viewlayers()
Read all the current scene's view layers from cache, as needed

### `reload`

bpy.ops.image.reload()
Reload current image from disk

### `remove_render_slot`

bpy.ops.image.remove_render_slot()
Remove the current render slot

### `render_border`

bpy.ops.image.render_border(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True)
Set the boundaries of the render region and enable render region

### `replace`

bpy.ops.image.replace(filepath="", hide_props_region=True, check_existing=False, filter_blender=False, filter_backup=False, filter_image=True, filter_movie=True, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, relative_path=True, show_multiview=False, use_multiview=False, display_type='DEFAULT', sort_method='DEFAULT')
Replace current image by another one from disk

### `resize`

bpy.ops.image.resize(size=(0, 0), all_udims=False)
Resize the image

### `rotate_orthogonal`

bpy.ops.image.rotate_orthogonal(degrees='90')
Rotate the image

### `sample`

bpy.ops.image.sample(size=1)
Use mouse to sample a color in current image

### `sample_line`

bpy.ops.image.sample_line(xstart=0, xend=0, ystart=0, yend=0, flip=False, cursor=5)
Sample a line and show it in Scope panels

### `save`

bpy.ops.image.save()
Save the image with current name and settings

### `save_all_modified`

bpy.ops.image.save_all_modified()
Save all modified images

### `save_as`

bpy.ops.image.save_as(save_as_render=False, copy=False, allow_path_tokens=True, filepath="", check_existing=True, filter_blender=False, filter_backup=False, filter_image=True, filter_movie=True, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, relative_path=True, show_multiview=False, use_multiview=False, display_type='DEFAULT', sort_method='DEFAULT')
Save the image with another name and/or settings

### `save_sequence`

bpy.ops.image.save_sequence()
Save a sequence of images

### `tile_add`

bpy.ops.image.tile_add(number=1002, count=1, label="", fill=True, color=(0, 0, 0, 1), generated_type='BLANK', width=1024, height=1024, float=False, alpha=True)
Adds a tile to the image

### `tile_fill`

bpy.ops.image.tile_fill(color=(0, 0, 0, 1), generated_type='BLANK', width=1024, height=1024, float=False, alpha=True)
Fill the current tile with a generated image

### `tile_remove`

bpy.ops.image.tile_remove()
Removes a tile from the image

### `unpack`

bpy.ops.image.unpack(method='USE_LOCAL', id="")
Save an image packed in the .blend file to disk

### `view_all`

bpy.ops.image.view_all(fit_view=False)
View the entire image

### `view_center_cursor`

bpy.ops.image.view_center_cursor()
Center the view so that the cursor is in the middle of the view

### `view_cursor_center`

bpy.ops.image.view_cursor_center(fit_view=False)
Set 2D Cursor To Center View location

### `view_pan`

bpy.ops.image.view_pan(offset=(0, 0))
Pan the view

### `view_selected`

bpy.ops.image.view_selected()
View all selected UVs

### `view_zoom`

bpy.ops.image.view_zoom(factor=0, use_cursor_init=True)
Zoom in/out the image

### `view_zoom_border`

bpy.ops.image.view_zoom_border(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True, zoom_out=False)
Zoom in the view to the nearest item contained in the border

### `view_zoom_in`

bpy.ops.image.view_zoom_in(location=(0, 0))
Zoom in the image (centered around 2D cursor)

### `view_zoom_out`

bpy.ops.image.view_zoom_out(location=(0, 0))
Zoom out the image (centered around 2D cursor)

### `view_zoom_ratio`

bpy.ops.image.view_zoom_ratio(ratio=0)
Set zoom ratio of the view
