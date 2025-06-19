# ui

Part of `bpy.ops`
Module: `bpy.ops.ui`

## Operators (34)

### `assign_default_button`

bpy.ops.ui.assign_default_button()
Set this property's current value as the new default

### `button_execute`

bpy.ops.ui.button_execute(skip_depressed=False)
Presses active button

### `button_string_clear`

bpy.ops.ui.button_string_clear()
Unsets the text of the active button

### `copy_as_driver_button`

bpy.ops.ui.copy_as_driver_button()
Create a new driver with this property as input, and copy it to the internal clipboard. Use Paste Driver to add it to the target property, or Paste Driver Variables to extend an existing driver

### `copy_data_path_button`

bpy.ops.ui.copy_data_path_button(full_path=False)
Copy the RNA data path for this property to the clipboard

### `copy_driver_to_selected_button`

bpy.ops.ui.copy_driver_to_selected_button(all=False)
Copy the property's driver from the active item to the same property of all selected items, if the same property exists

### `copy_python_command_button`

bpy.ops.ui.copy_python_command_button()
Copy the Python command matching this button

### `copy_to_selected_button`

bpy.ops.ui.copy_to_selected_button(all=True)
Copy the property's value from the active item to the same property of all selected items if the same property exists

### `drop_color`

bpy.ops.ui.drop_color(color=(0, 0, 0, 0), gamma=False, has_alpha=False)
Drop colors to buttons

### `drop_material`

bpy.ops.ui.drop_material(session_uid=0)
Drag material to Material slots in Properties

### `drop_name`

bpy.ops.ui.drop_name(string="")
Drop name to button

### `editsource`

bpy.ops.ui.editsource()
Edit UI source code of the active button

### `eyedropper_bone`

bpy.ops.ui.eyedropper_bone()
Sample a bone from the 3D View or the Outliner to store in a property

### `eyedropper_color`

bpy.ops.ui.eyedropper_color(prop_data_path="")
Sample a color from the Blender window to store in a property

### `eyedropper_colorramp`

bpy.ops.ui.eyedropper_colorramp()
Sample a color band

### `eyedropper_colorramp_point`

bpy.ops.ui.eyedropper_colorramp_point()
Point-sample a color band

### `eyedropper_depth`

bpy.ops.ui.eyedropper_depth(prop_data_path="")
Sample depth from the 3D view

### `eyedropper_driver`

bpy.ops.ui.eyedropper_driver(mapping_type='SINGLE_MANY')
Pick a property to use as a driver target

### `eyedropper_grease_pencil_color`

bpy.ops.ui.eyedropper_grease_pencil_color(mode='MATERIAL', material_mode='STROKE')
Sample a color from the Blender Window and create Grease Pencil material

### `eyedropper_id`

bpy.ops.ui.eyedropper_id()
Sample a data-block from the 3D View to store in a property

### `jump_to_target_button`

bpy.ops.ui.jump_to_target_button()
Switch to the target object or bone

### `list_start_filter`

bpy.ops.ui.list_start_filter()
Start entering filter text for the list in focus

### `override_idtemplate_clear`

bpy.ops.ui.override_idtemplate_clear()
Delete the selected local override and relink its usages to the linked data-block if possible, else reset it and mark it as non editable

### `override_idtemplate_make`

bpy.ops.ui.override_idtemplate_make()
Create a local override of the selected linked data-block, and its hierarchy of dependencies

### `override_idtemplate_reset`

bpy.ops.ui.override_idtemplate_reset()
Reset the selected local override to its linked reference values

### `override_remove_button`

bpy.ops.ui.override_remove_button(all=True)
Remove an override operation

### `override_type_set_button`

bpy.ops.ui.override_type_set_button(all=True, type='REPLACE')
Create an override operation, or set the type of an existing one

### `reloadtranslation`

bpy.ops.ui.reloadtranslation()
Force a full reload of UI translation

### `reset_default_button`

bpy.ops.ui.reset_default_button(all=True)
Reset this property's value to its default value

### `unset_property_button`

bpy.ops.ui.unset_property_button()
Clear the property and use default or generated value in operators

### `view_drop`

bpy.ops.ui.view_drop()
Drag and drop onto a data-set or item within the data-set

### `view_item_rename`

bpy.ops.ui.view_item_rename()
Rename the active item in the data-set view

### `view_scroll`

bpy.ops.ui.view_scroll()
(undocumented operator)

### `view_start_filter`

bpy.ops.ui.view_start_filter()
Start entering filter text for the data-set in focus
