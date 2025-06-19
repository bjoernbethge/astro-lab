# paint

Part of `bpy.ops`
Module: `bpy.ops.paint`

## Operators (54)

### `add_simple_uvs`

bpy.ops.paint.add_simple_uvs()
Add cube map UVs on mesh

### `add_texture_paint_slot`

bpy.ops.paint.add_texture_paint_slot(type='BASE_COLOR', slot_type='IMAGE', name="Untitled", color=(0, 0, 0, 1), width=1024, height=1024, alpha=True, generated_type='BLANK', float=False, domain='POINT', data_type='FLOAT_COLOR')
Add a paint slot

### `brush_colors_flip`

bpy.ops.paint.brush_colors_flip()
Swap primary and secondary brush colors

### `face_select_all`

bpy.ops.paint.face_select_all(action='TOGGLE')
Change selection for all faces

### `face_select_hide`

bpy.ops.paint.face_select_hide(unselected=False)
Hide selected faces

### `face_select_less`

bpy.ops.paint.face_select_less(face_step=True)
Deselect Faces connected to existing selection

### `face_select_linked`

bpy.ops.paint.face_select_linked()
Select linked faces

### `face_select_linked_pick`

bpy.ops.paint.face_select_linked_pick(deselect=False)
Select linked faces under the cursor

### `face_select_loop`

bpy.ops.paint.face_select_loop(select=True, extend=False)
Select face loop under the cursor

### `face_select_more`

bpy.ops.paint.face_select_more(face_step=True)
Select Faces connected to existing selection

### `face_vert_reveal`

bpy.ops.paint.face_vert_reveal(select=True)
Reveal hidden faces and vertices

### `grab_clone`

bpy.ops.paint.grab_clone(delta=(0, 0))
Move the clone source image

### `hide_show`

bpy.ops.paint.hide_show(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True, action='HIDE', area='Inside', use_front_faces_only=False)
Hide/show some vertices

### `hide_show_all`

bpy.ops.paint.hide_show_all(action='HIDE')
Hide/show all vertices

### `hide_show_lasso_gesture`

bpy.ops.paint.hide_show_lasso_gesture(path=[], use_smooth_stroke=False, smooth_stroke_factor=0.75, smooth_stroke_radius=35, action='HIDE', area='Inside', use_front_faces_only=False)
Hide/show some vertices

### `hide_show_line_gesture`

bpy.ops.paint.hide_show_line_gesture(xstart=0, xend=0, ystart=0, yend=0, flip=False, cursor=5, action='HIDE', area='Inside', use_front_faces_only=False, use_limit_to_segment=False)
Hide/show some vertices

### `hide_show_masked`

bpy.ops.paint.hide_show_masked(action='HIDE')
Hide/show all masked vertices above a threshold

### `hide_show_polyline_gesture`

bpy.ops.paint.hide_show_polyline_gesture(path=[], action='HIDE', area='Inside', use_front_faces_only=False)
Hide/show some vertices

### `image_from_view`

bpy.ops.paint.image_from_view(filepath="")
Make an image from biggest 3D view for reprojection

### `image_paint`

bpy.ops.paint.image_paint(stroke=[], mode='NORMAL', pen_flip=False)
Paint a stroke into the image

### `mask_box_gesture`

bpy.ops.paint.mask_box_gesture(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True, use_front_faces_only=False, mode='VALUE', value=1)
Mask within a rectangle defined by the cursor

### `mask_flood_fill`

bpy.ops.paint.mask_flood_fill(mode='VALUE', value=0)
Fill the whole mask with a given value, or invert its values

### `mask_lasso_gesture`

bpy.ops.paint.mask_lasso_gesture(path=[], use_smooth_stroke=False, smooth_stroke_factor=0.75, smooth_stroke_radius=35, use_front_faces_only=False, mode='VALUE', value=1)
Mask within a shape defined by the cursor

### `mask_line_gesture`

bpy.ops.paint.mask_line_gesture(xstart=0, xend=0, ystart=0, yend=0, flip=False, cursor=5, use_front_faces_only=False, use_limit_to_segment=False, mode='VALUE', value=1)
Mask to one side of a line defined by the cursor

### `mask_polyline_gesture`

bpy.ops.paint.mask_polyline_gesture(path=[], use_front_faces_only=False, mode='VALUE', value=1)
Mask within a shape defined by the cursor

### `project_image`

bpy.ops.paint.project_image(image='<UNKNOWN ENUM>')
Project an edited render from the active camera back onto the object

### `sample_color`

bpy.ops.paint.sample_color(location=(0, 0), merged=False, palette=False)
Use the mouse to sample a color in the image

### `texture_paint_toggle`

bpy.ops.paint.texture_paint_toggle()
Toggle texture paint mode in 3D view

### `vert_select_all`

bpy.ops.paint.vert_select_all(action='TOGGLE')
Change selection for all vertices

### `vert_select_hide`

bpy.ops.paint.vert_select_hide(unselected=False)
Hide selected vertices

### `vert_select_less`

bpy.ops.paint.vert_select_less(face_step=True)
Deselect Vertices connected to existing selection

### `vert_select_linked`

bpy.ops.paint.vert_select_linked()
Select linked vertices

### `vert_select_linked_pick`

bpy.ops.paint.vert_select_linked_pick(select=True)
Select linked vertices under the cursor

### `vert_select_more`

bpy.ops.paint.vert_select_more(face_step=True)
Select Vertices connected to existing selection

### `vert_select_ungrouped`

bpy.ops.paint.vert_select_ungrouped(extend=False)
Select vertices without a group

### `vertex_color_brightness_contrast`

bpy.ops.paint.vertex_color_brightness_contrast(brightness=0, contrast=0)
Adjust vertex color brightness/contrast

### `vertex_color_dirt`

bpy.ops.paint.vertex_color_dirt(blur_strength=1, blur_iterations=1, clean_angle=3.14159, dirt_angle=0, dirt_only=False, normalize=True)
Generate a dirt map gradient based on cavity

### `vertex_color_from_weight`

bpy.ops.paint.vertex_color_from_weight()
Convert active weight into gray scale vertex colors

### `vertex_color_hsv`

bpy.ops.paint.vertex_color_hsv(h=0.5, s=1, v=1)
Adjust vertex color Hue/Saturation/Value

### `vertex_color_invert`

bpy.ops.paint.vertex_color_invert()
Invert RGB values

### `vertex_color_levels`

bpy.ops.paint.vertex_color_levels(offset=0, gain=1)
Adjust levels of vertex colors

### `vertex_color_set`

bpy.ops.paint.vertex_color_set(use_alpha=True)
Fill the active vertex color layer with the current paint color

### `vertex_color_smooth`

bpy.ops.paint.vertex_color_smooth()
Smooth colors across vertices

### `vertex_paint`

bpy.ops.paint.vertex_paint(stroke=[], mode='NORMAL', pen_flip=False, override_location=False)
Paint a stroke in the active color attribute layer

### `vertex_paint_toggle`

bpy.ops.paint.vertex_paint_toggle()
Toggle the vertex paint mode in 3D view

### `visibility_filter`

bpy.ops.paint.visibility_filter(action='GROW', iterations=1, auto_iteration_count=True)
Edit the visibility of the current mesh

### `visibility_invert`

bpy.ops.paint.visibility_invert()
Invert the visibility of all vertices

### `weight_from_bones`

bpy.ops.paint.weight_from_bones(type='AUTOMATIC')
Set the weights of the groups matching the attached armature's selected bones, using the distance between the vertices and the bones

### `weight_gradient`

bpy.ops.paint.weight_gradient(type='LINEAR', xstart=0, xend=0, ystart=0, yend=0, flip=False, cursor=5)
Draw a line to apply a weight gradient to selected vertices

### `weight_paint`

bpy.ops.paint.weight_paint(stroke=[], mode='NORMAL', pen_flip=False, override_location=False)
Paint a stroke in the current vertex group's weights

### `weight_paint_toggle`

bpy.ops.paint.weight_paint_toggle()
Toggle weight paint mode in 3D view

### `weight_sample`

bpy.ops.paint.weight_sample()
Use the mouse to sample a weight in the 3D view

### `weight_sample_group`

bpy.ops.paint.weight_sample_group()
Select one of the vertex groups available under current mouse position

### `weight_set`

bpy.ops.paint.weight_set()
Fill the active vertex group with the current paint weight
