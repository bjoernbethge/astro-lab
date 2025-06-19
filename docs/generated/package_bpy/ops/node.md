# node

Part of `bpy.ops`
Module: `bpy.ops.node`

## Operators (118)

### `add_collection`

bpy.ops.node.add_collection(name="", session_uid=0)
Add a collection info node to the current node editor

### `add_color`

bpy.ops.node.add_color(color=(0, 0, 0, 0), gamma=False, has_alpha=False)
Add a color node to the current node editor

### `add_file`

bpy.ops.node.add_file(filepath="", directory="", files=[], hide_props_region=True, check_existing=False, filter_blender=False, filter_backup=False, filter_image=True, filter_movie=True, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, relative_path=True, show_multiview=False, use_multiview=False, display_type='DEFAULT', sort_method='DEFAULT', name="", session_uid=0)
Add a file node to the current node editor

### `add_foreach_geometry_element_zone`

bpy.ops.node.add_foreach_geometry_element_zone(use_transform=False, settings=[], offset=(150, 0))
Add a For Each Geometry Element zone that allows executing nodes e.g. for each vertex separately

### `add_group`

bpy.ops.node.add_group(name="", session_uid=0, show_datablock_in_node=True)
Add an existing node group to the current node editor

### `add_group_asset`

bpy.ops.node.add_group_asset(asset_library_type='LOCAL', asset_library_identifier="", relative_asset_identifier="")
Add a node group asset to the active node tree

### `add_mask`

bpy.ops.node.add_mask(name="", session_uid=0)
Add a mask node to the current node editor

### `add_material`

bpy.ops.node.add_material(name="", session_uid=0)
Add a material node to the current node editor

### `add_node`

bpy.ops.node.add_node(use_transform=False, settings=[], type="")
Add a node to the active tree

### `add_object`

bpy.ops.node.add_object(name="", session_uid=0)
Add an object info node to the current node editor

### `add_repeat_zone`

bpy.ops.node.add_repeat_zone(use_transform=False, settings=[], offset=(150, 0))
Add a repeat zone that allows executing nodes a dynamic number of times

### `add_reroute`

bpy.ops.node.add_reroute(path=[], cursor=11)
Add a reroute node

### `add_simulation_zone`

bpy.ops.node.add_simulation_zone(use_transform=False, settings=[], offset=(150, 0))
Add simulation zone input and output nodes to the active tree

### `attach`

bpy.ops.node.attach()
Attach active node to a frame

### `backimage_fit`

bpy.ops.node.backimage_fit()
Fit the background image to the view

### `backimage_move`

bpy.ops.node.backimage_move()
Move node backdrop

### `backimage_sample`

bpy.ops.node.backimage_sample()
Use mouse to sample background image

### `backimage_zoom`

bpy.ops.node.backimage_zoom(factor=1.2)
Zoom in/out the background image

### `bake_node_item_add`

bpy.ops.node.bake_node_item_add()
Add item below active item

### `bake_node_item_move`

bpy.ops.node.bake_node_item_move(direction='UP')
Move active item

### `bake_node_item_remove`

bpy.ops.node.bake_node_item_remove()
Remove active item

### `capture_attribute_item_add`

bpy.ops.node.capture_attribute_item_add()
Add item below active item

### `capture_attribute_item_move`

bpy.ops.node.capture_attribute_item_move(direction='UP')
Move active item

### `capture_attribute_item_remove`

bpy.ops.node.capture_attribute_item_remove()
Remove active item

### `clear_viewer_border`

bpy.ops.node.clear_viewer_border()
Clear the boundaries for viewer operations

### `clipboard_copy`

bpy.ops.node.clipboard_copy()
Copy the selected nodes to the internal clipboard

### `clipboard_paste`

bpy.ops.node.clipboard_paste(offset=(0, 0))
Paste nodes from the internal clipboard to the active node tree

### `collapse_hide_unused_toggle`

bpy.ops.node.collapse_hide_unused_toggle()
Toggle collapsed nodes and hide unused sockets

### `connect_to_output`

bpy.ops.node.connect_to_output(run_in_geometry_nodes=True)
Connect active node to the active output node of the node tree

### `cryptomatte_layer_add`

bpy.ops.node.cryptomatte_layer_add()
Add a new input layer to a Cryptomatte node

### `cryptomatte_layer_remove`

bpy.ops.node.cryptomatte_layer_remove()
Remove layer from a Cryptomatte node

### `deactivate_viewer`

bpy.ops.node.deactivate_viewer()
Deactivate selected viewer node in geometry nodes

### `default_group_width_set`

bpy.ops.node.default_group_width_set()
Set the width based on the parent group node in the current context

### `delete`

bpy.ops.node.delete()
Remove selected nodes

### `delete_reconnect`

bpy.ops.node.delete_reconnect()
Remove nodes and reconnect nodes as if deletion was muted

### `detach`

bpy.ops.node.detach()
Detach selected nodes from parents

### `detach_translate_attach`

bpy.ops.node.detach_translate_attach(NODE_OT_detach={}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False}, NODE_OT_attach={})
Detach nodes, move and attach to frame

### `duplicate`

bpy.ops.node.duplicate(keep_inputs=False, linked=True)
Duplicate selected nodes

### `duplicate_move`

bpy.ops.node.duplicate_move(NODE_OT_duplicate={"keep_inputs":False, "linked":True}, NODE_OT_translate_attach={"TRANSFORM_OT_translate":{"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False}, "NODE_OT_attach":{}})
Duplicate selected nodes and move them

### `duplicate_move_keep_inputs`

bpy.ops.node.duplicate_move_keep_inputs(NODE_OT_duplicate={"keep_inputs":False, "linked":True}, NODE_OT_translate_attach={"TRANSFORM_OT_translate":{"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False}, "NODE_OT_attach":{}})
Duplicate selected nodes keeping input links and move them

### `duplicate_move_linked`

bpy.ops.node.duplicate_move_linked(NODE_OT_duplicate={"keep_inputs":False, "linked":True}, NODE_OT_translate_attach={"TRANSFORM_OT_translate":{"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False}, "NODE_OT_attach":{}})
Duplicate selected nodes, but not their node trees, and move them

### `enum_definition_item_add`

bpy.ops.node.enum_definition_item_add()
Add item below active item

### `enum_definition_item_move`

bpy.ops.node.enum_definition_item_move(direction='UP')
Move active item

### `enum_definition_item_remove`

bpy.ops.node.enum_definition_item_remove()
Remove active item

### `find_node`

bpy.ops.node.find_node()
Search for a node by name and focus and select it

### `foreach_geometry_element_zone_generation_item_add`

bpy.ops.node.foreach_geometry_element_zone_generation_item_add()
Add item below active item

### `foreach_geometry_element_zone_generation_item_move`

bpy.ops.node.foreach_geometry_element_zone_generation_item_move(direction='UP')
Move active item

### `foreach_geometry_element_zone_generation_item_remove`

bpy.ops.node.foreach_geometry_element_zone_generation_item_remove()
Remove active item

### `foreach_geometry_element_zone_input_item_add`

bpy.ops.node.foreach_geometry_element_zone_input_item_add()
Add item below active item

### `foreach_geometry_element_zone_input_item_move`

bpy.ops.node.foreach_geometry_element_zone_input_item_move(direction='UP')
Move active item

### `foreach_geometry_element_zone_input_item_remove`

bpy.ops.node.foreach_geometry_element_zone_input_item_remove()
Remove active item

### `foreach_geometry_element_zone_main_item_add`

bpy.ops.node.foreach_geometry_element_zone_main_item_add()
Add item below active item

### `foreach_geometry_element_zone_main_item_move`

bpy.ops.node.foreach_geometry_element_zone_main_item_move(direction='UP')
Move active item

### `foreach_geometry_element_zone_main_item_remove`

bpy.ops.node.foreach_geometry_element_zone_main_item_remove()
Remove active item

### `gltf_settings_node_operator`

bpy.ops.node.gltf_settings_node_operator()
Add a node to the active tree for glTF export

### `group_edit`

bpy.ops.node.group_edit(exit=False)
Edit node group

### `group_insert`

bpy.ops.node.group_insert()
Insert selected nodes into a node group

### `group_make`

bpy.ops.node.group_make()
Make group from selected nodes

### `group_separate`

bpy.ops.node.group_separate(type='COPY')
Separate selected nodes from the node group

### `group_ungroup`

bpy.ops.node.group_ungroup()
Ungroup selected nodes

### `hide_socket_toggle`

bpy.ops.node.hide_socket_toggle()
Toggle unused node socket display

### `hide_toggle`

bpy.ops.node.hide_toggle()
Toggle hiding of selected nodes

### `index_switch_item_add`

bpy.ops.node.index_switch_item_add()
Add bake item

### `index_switch_item_remove`

bpy.ops.node.index_switch_item_remove(index=0)
Remove an item from the index switch

### `insert_offset`

bpy.ops.node.insert_offset()
Automatically offset nodes on insertion

### `interface_item_duplicate`

bpy.ops.node.interface_item_duplicate()
Add a copy of the active item to the interface

### `interface_item_new`

bpy.ops.node.interface_item_new(item_type='INPUT')
Add a new item to the interface

### `interface_item_remove`

bpy.ops.node.interface_item_remove()
Remove active item from the interface

### `join`

bpy.ops.node.join()
Attach selected nodes to a new common frame

### `link`

bpy.ops.node.link(detach=False, drag_start=(0, 0), inside_padding=2, outside_padding=0, speed_ramp=1, max_speed=26, delay=0.5, zoom_influence=0.5)
Use the mouse to create a link between two nodes

### `link_make`

bpy.ops.node.link_make(replace=False)
Make a link between selected output and input sockets

### `link_viewer`

bpy.ops.node.link_viewer()
Link to viewer node

### `links_cut`

bpy.ops.node.links_cut(path=[], cursor=15)
Use the mouse to cut (remove) some links

### `links_detach`

bpy.ops.node.links_detach()
Remove all links to selected nodes, and try to connect neighbor nodes together

### `links_mute`

bpy.ops.node.links_mute(path=[], cursor=38)
Use the mouse to mute links

### `move_detach_links`

bpy.ops.node.move_detach_links(NODE_OT_links_detach={}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Move a node to detach links

### `move_detach_links_release`

bpy.ops.node.move_detach_links_release(NODE_OT_links_detach={}, NODE_OT_translate_attach={"TRANSFORM_OT_translate":{"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False}, "NODE_OT_attach":{}})
Move a node to detach links

### `mute_toggle`

bpy.ops.node.mute_toggle()
Toggle muting of selected nodes

### `new_geometry_node_group_assign`

bpy.ops.node.new_geometry_node_group_assign()
Create a new geometry node group and assign it to the active modifier

### `new_geometry_node_group_tool`

bpy.ops.node.new_geometry_node_group_tool()
Create a new geometry node group for a tool

### `new_geometry_nodes_modifier`

bpy.ops.node.new_geometry_nodes_modifier()
Create a new modifier with a new geometry node group

### `new_node_tree`

bpy.ops.node.new_node_tree(type='GeometryNodeTree', name="NodeTree")
Create a new node tree

### `node_color_preset_add`

bpy.ops.node.node_color_preset_add(name="", remove_name=False, remove_active=False)
Add or remove a Node Color Preset

### `node_copy_color`

bpy.ops.node.node_copy_color()
Copy color to all selected nodes

### `options_toggle`

bpy.ops.node.options_toggle()
Toggle option buttons display for selected nodes

### `output_file_add_socket`

bpy.ops.node.output_file_add_socket(file_path="Image")
Add a new input to a file output node

### `output_file_move_active_socket`

bpy.ops.node.output_file_move_active_socket(direction='DOWN')
Move the active input of a file output node up or down the list

### `output_file_remove_active_socket`

bpy.ops.node.output_file_remove_active_socket()
Remove the active input from a file output node

### `parent_set`

bpy.ops.node.parent_set()
Attach selected nodes

### `preview_toggle`

bpy.ops.node.preview_toggle()
Toggle preview display for selected nodes

### `read_viewlayers`

bpy.ops.node.read_viewlayers()
Read all render layers of all used scenes

### `render_changed`

bpy.ops.node.render_changed()
Render current scene, when input node's layer has been changed

### `repeat_zone_item_add`

bpy.ops.node.repeat_zone_item_add()
Add item below active item

### `repeat_zone_item_move`

bpy.ops.node.repeat_zone_item_move(direction='UP')
Move active item

### `repeat_zone_item_remove`

bpy.ops.node.repeat_zone_item_remove()
Remove active item

### `resize`

bpy.ops.node.resize()
Resize a node

### `select`

bpy.ops.node.select(extend=False, deselect=False, toggle=False, deselect_all=False, select_passthrough=False, location=(0, 0), socket_select=False, clear_viewer=False)
Select the node under the cursor

### `select_all`

bpy.ops.node.select_all(action='TOGGLE')
(De)select all nodes

### `select_box`

bpy.ops.node.select_box(tweak=False, xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True, mode='SET')
Use box selection to select nodes

### `select_circle`

bpy.ops.node.select_circle(x=0, y=0, radius=25, wait_for_input=True, mode='SET')
Use circle selection to select nodes

### `select_grouped`

bpy.ops.node.select_grouped(extend=False, type='TYPE')
Select nodes with similar properties

### `select_lasso`

bpy.ops.node.select_lasso(tweak=False, path=[], use_smooth_stroke=False, smooth_stroke_factor=0.75, smooth_stroke_radius=35, mode='SET')
Select nodes using lasso selection

### `select_link_viewer`

bpy.ops.node.select_link_viewer(NODE_OT_select={"extend":False, "deselect":False, "toggle":False, "deselect_all":False, "select_passthrough":False, "location":(0, 0), "socket_select":False, "clear_viewer":False}, NODE_OT_link_viewer={})
Select node and link it to a viewer node

### `select_linked_from`

bpy.ops.node.select_linked_from()
Select nodes linked from the selected ones

### `select_linked_to`

bpy.ops.node.select_linked_to()
Select nodes linked to the selected ones

### `select_same_type_step`

bpy.ops.node.select_same_type_step(prev=False)
Activate and view same node type, step by step

### `shader_script_update`

bpy.ops.node.shader_script_update()
Update shader script node with new sockets and options from the script

### `simulation_zone_item_add`

bpy.ops.node.simulation_zone_item_add()
Add item below active item

### `simulation_zone_item_move`

bpy.ops.node.simulation_zone_item_move(direction='UP')
Move active item

### `simulation_zone_item_remove`

bpy.ops.node.simulation_zone_item_remove()
Remove active item

### `translate_attach`

bpy.ops.node.translate_attach(TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False}, NODE_OT_attach={})
Move nodes and attach to frame

### `translate_attach_remove_on_cancel`

bpy.ops.node.translate_attach_remove_on_cancel(TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False}, NODE_OT_attach={})
Move nodes and attach to frame

### `tree_path_parent`

bpy.ops.node.tree_path_parent()
Go to parent node tree

### `view_all`

bpy.ops.node.view_all()
Resize view so you can see all nodes

### `view_selected`

bpy.ops.node.view_selected()
Resize view so you can see selected nodes

### `viewer_border`

bpy.ops.node.viewer_border(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True)
Set the boundaries for viewer operations

### `viewer_shortcut_get`

bpy.ops.node.viewer_shortcut_get(viewer_index=0)
Activate a specific compositor viewer node using 1,2,..,9 keys

### `viewer_shortcut_set`

bpy.ops.node.viewer_shortcut_set(viewer_index=0)
Create a compositor viewer shortcut for the selected node by pressing ctrl+1,2,..9
