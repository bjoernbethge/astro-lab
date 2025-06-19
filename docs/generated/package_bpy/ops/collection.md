# collection

Part of `bpy.ops`
Module: `bpy.ops.collection`

## Operators (9)

### `create`

bpy.ops.collection.create(name="Collection")
Create an object collection from selected objects

### `export_all`

bpy.ops.collection.export_all()
Invoke all configured exporters on this collection

### `exporter_add`

bpy.ops.collection.exporter_add(name="")
Add Exporter

### `exporter_export`

bpy.ops.collection.exporter_export(index=0)
Invoke the export operation

### `exporter_remove`

bpy.ops.collection.exporter_remove(index=0)
Remove Exporter

### `objects_add_active`

bpy.ops.collection.objects_add_active(collection='<UNKNOWN ENUM>')
Add selected objects to one of the collections the active-object is part of. Optionally add to "All Collections" to ensure selected objects are included in the same collections as the active object

### `objects_remove`

bpy.ops.collection.objects_remove(collection='<UNKNOWN ENUM>')
Remove selected objects from a collection

### `objects_remove_active`

bpy.ops.collection.objects_remove_active(collection='<UNKNOWN ENUM>')
Remove the object from an object collection that contains the active object

### `objects_remove_all`

bpy.ops.collection.objects_remove_all()
Remove selected objects from all collections
