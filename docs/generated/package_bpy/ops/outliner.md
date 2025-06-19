# outliner

Part of `bpy.ops`
Module: `bpy.ops.outliner`

## Operators (71)

### `action_set`

bpy.ops.outliner.action_set(action='<UNKNOWN ENUM>')
Change the active action used

### `animdata_operation`

bpy.ops.outliner.animdata_operation(type='CLEAR_ANIMDATA')
(undocumented operator)

### `clear_filter`

bpy.ops.outliner.clear_filter()
Clear the search filter

### `collection_color_tag_set`

bpy.ops.outliner.collection_color_tag_set(color='NONE')
Set a color tag for the selected collections

### `collection_disable`

bpy.ops.outliner.collection_disable()
Disable viewport display in the view layers

### `collection_disable_render`

bpy.ops.outliner.collection_disable_render()
Do not render this collection

### `collection_drop`

bpy.ops.outliner.collection_drop()
Drag to move to collection in Outliner

### `collection_duplicate`

bpy.ops.outliner.collection_duplicate()
Recursively duplicate the collection, all its children, objects and object data

### `collection_duplicate_linked`

bpy.ops.outliner.collection_duplicate_linked()
Recursively duplicate the collection, all its children and objects, with linked object data

### `collection_enable`

bpy.ops.outliner.collection_enable()
Enable viewport display in the view layers

### `collection_enable_render`

bpy.ops.outliner.collection_enable_render()
Render the collection

### `collection_exclude_clear`

bpy.ops.outliner.collection_exclude_clear()
Include collection in the active view layer

### `collection_exclude_set`

bpy.ops.outliner.collection_exclude_set()
Exclude collection from the active view layer

### `collection_hide`

bpy.ops.outliner.collection_hide()
Hide the collection in this view layer

### `collection_hide_inside`

bpy.ops.outliner.collection_hide_inside()
Hide all the objects and collections inside the collection

### `collection_hierarchy_delete`

bpy.ops.outliner.collection_hierarchy_delete()
Delete selected collection hierarchies

### `collection_holdout_clear`

bpy.ops.outliner.collection_holdout_clear()
Clear masking of collection in the active view layer

### `collection_holdout_set`

bpy.ops.outliner.collection_holdout_set()
Mask collection in the active view layer

### `collection_indirect_only_clear`

bpy.ops.outliner.collection_indirect_only_clear()
Clear collection contributing only indirectly in the view layer

### `collection_indirect_only_set`

bpy.ops.outliner.collection_indirect_only_set()
Set collection to only contribute indirectly (through shadows and reflections) in the view layer

### `collection_instance`

bpy.ops.outliner.collection_instance()
Instance selected collections to active scene

### `collection_isolate`

bpy.ops.outliner.collection_isolate(extend=False)
Hide all but this collection and its parents

### `collection_link`

bpy.ops.outliner.collection_link()
Link selected collections to active scene

### `collection_new`

bpy.ops.outliner.collection_new(nested=True)
Add a new collection inside selected collection

### `collection_objects_deselect`

bpy.ops.outliner.collection_objects_deselect()
Deselect objects in collection

### `collection_objects_select`

bpy.ops.outliner.collection_objects_select()
Select objects in collection

### `collection_show`

bpy.ops.outliner.collection_show()
Show the collection in this view layer

### `collection_show_inside`

bpy.ops.outliner.collection_show_inside()
Show all the objects and collections inside the collection

### `constraint_operation`

bpy.ops.outliner.constraint_operation(type='ENABLE')
(undocumented operator)

### `data_operation`

bpy.ops.outliner.data_operation(type='DEFAULT')
(undocumented operator)

### `datastack_drop`

bpy.ops.outliner.datastack_drop()
Copy or reorder modifiers, constraints, and effects

### `delete`

bpy.ops.outliner.delete(hierarchy=False)
Delete selected objects and collections

### `drivers_add_selected`

bpy.ops.outliner.drivers_add_selected()
Add drivers to selected items

### `drivers_delete_selected`

bpy.ops.outliner.drivers_delete_selected()
Delete drivers assigned to selected items

### `expanded_toggle`

bpy.ops.outliner.expanded_toggle()
Expand/Collapse all items

### `hide`

bpy.ops.outliner.hide()
Hide selected objects and collections

### `highlight_update`

bpy.ops.outliner.highlight_update()
Update the item highlight based on the current mouse position

### `id_copy`

bpy.ops.outliner.id_copy()
Copy the selected data-blocks to the internal clipboard

### `id_delete`

bpy.ops.outliner.id_delete()
Delete the ID under cursor

### `id_operation`

bpy.ops.outliner.id_operation(type='UNLINK')
General data-block management operations

### `id_paste`

bpy.ops.outliner.id_paste()
Paste data-blocks from the internal clipboard

### `id_remap`

bpy.ops.outliner.id_remap(id_type='OBJECT', old_id='<UNKNOWN ENUM>', new_id='<UNKNOWN ENUM>')
(undocumented operator)

### `item_activate`

bpy.ops.outliner.item_activate(extend=False, extend_range=False, deselect_all=False, recurse=False)
Handle mouse clicks to select and activate items

### `item_drag_drop`

bpy.ops.outliner.item_drag_drop()
Drag and drop element to another place

### `item_openclose`

bpy.ops.outliner.item_openclose(all=False)
Toggle whether item under cursor is enabled or closed

### `item_rename`

bpy.ops.outliner.item_rename(use_active=False)
Rename the active element

### `keyingset_add_selected`

bpy.ops.outliner.keyingset_add_selected()
Add selected items (blue-gray rows) to active Keying Set

### `keyingset_remove_selected`

bpy.ops.outliner.keyingset_remove_selected()
Remove selected items (blue-gray rows) from active Keying Set

### `lib_operation`

bpy.ops.outliner.lib_operation(type='DELETE')
(undocumented operator)

### `lib_relocate`

bpy.ops.outliner.lib_relocate()
Relocate the library under cursor

### `liboverride_operation`

bpy.ops.outliner.liboverride_operation(type='OVERRIDE_LIBRARY_CREATE_HIERARCHY', selection_set='SELECTED')
Create, reset or clear library override hierarchies

### `liboverride_troubleshoot_operation`

bpy.ops.outliner.liboverride_troubleshoot_operation(type='OVERRIDE_LIBRARY_RESYNC_HIERARCHY', selection_set='SELECTED')
Advanced operations over library override to help fix broken hierarchies

### `material_drop`

bpy.ops.outliner.material_drop()
Drag material to object in Outliner

### `modifier_operation`

bpy.ops.outliner.modifier_operation(type='APPLY')
(undocumented operator)

### `object_operation`

bpy.ops.outliner.object_operation(type='SELECT')
(undocumented operator)

### `operation`

bpy.ops.outliner.operation()
Context menu for item operations

### `orphans_manage`

bpy.ops.outliner.orphans_manage()
Open a window to manage unused data

### `orphans_purge`

bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
Clear all orphaned data-blocks without any users from the file

### `parent_clear`

bpy.ops.outliner.parent_clear()
Drag to clear parent in Outliner

### `parent_drop`

bpy.ops.outliner.parent_drop()
Drag to parent in Outliner

### `scene_drop`

bpy.ops.outliner.scene_drop()
Drag object to scene in Outliner

### `scene_operation`

bpy.ops.outliner.scene_operation(type='DELETE')
Context menu for scene operations

### `scroll_page`

bpy.ops.outliner.scroll_page(up=False)
Scroll page up or down

### `select_all`

bpy.ops.outliner.select_all(action='TOGGLE')
Toggle the Outliner selection of items

### `select_box`

bpy.ops.outliner.select_box(tweak=False, xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True, mode='SET')
Use box selection to select tree elements

### `select_walk`

bpy.ops.outliner.select_walk(direction='UP', extend=False, toggle_all=False)
Use walk navigation to select tree elements

### `show_active`

bpy.ops.outliner.show_active()
Open up the tree and adjust the view so that the active object is shown centered

### `show_hierarchy`

bpy.ops.outliner.show_hierarchy()
Open all object entries and close all others

### `show_one_level`

bpy.ops.outliner.show_one_level(open=True)
Expand/collapse all entries by one level

### `start_filter`

bpy.ops.outliner.start_filter()
Start entering filter text

### `unhide_all`

bpy.ops.outliner.unhide_all()
Unhide all objects and collections
