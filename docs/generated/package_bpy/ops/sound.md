# sound

Part of `bpy.ops`
Module: `bpy.ops.sound`

## Operators (7)

### `bake_animation`

bpy.ops.sound.bake_animation()
Update the audio animation cache

### `mixdown`

bpy.ops.sound.mixdown(filepath="", check_existing=True, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=True, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, relative_path=True, display_type='DEFAULT', sort_method='DEFAULT', accuracy=1024, container='FLAC', codec='FLAC', channels='STEREO', format='S16', mixrate=48000, bitrate=192, split_channels=False)
Mix the scene's audio to a sound file

### `open`

bpy.ops.sound.open(filepath="", hide_props_region=True, check_existing=False, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=True, filter_python=False, filter_font=False, filter_sound=True, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, relative_path=True, show_multiview=False, use_multiview=False, display_type='DEFAULT', sort_method='DEFAULT', cache=False, mono=False)
Load a sound file

### `open_mono`

bpy.ops.sound.open_mono(filepath="", hide_props_region=True, check_existing=False, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=True, filter_python=False, filter_font=False, filter_sound=True, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, relative_path=True, show_multiview=False, use_multiview=False, display_type='DEFAULT', sort_method='DEFAULT', cache=False, mono=True)
Load a sound file as mono

### `pack`

bpy.ops.sound.pack()
Pack the sound into the current blend file

### `unpack`

bpy.ops.sound.unpack(method='USE_LOCAL', id="")
Unpack the sound to the samples filename

### `update_animation_flags`

bpy.ops.sound.update_animation_flags()
Update animation flags
