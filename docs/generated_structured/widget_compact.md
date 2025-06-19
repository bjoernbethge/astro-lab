# Widget Module

Auto-generated documentation for `astro_lab.widget`

## Classes

### BlenderImageWidget

AnyWidget zur Anzeige von Blender-Renderings mit interaktiven Kontrollen.

#### Methods

**`display_image_from_path(self, image_path: Union[str, pathlib.Path], render_time: float = 0.0, resolution: str = '', engine: str = 'BLENDER_EEVEE_NEXT', samples: int = 64)`**

Zeige ein Bild aus einer Datei an.

Args:
image_path: Pfad zur Bilddatei
render_time: Renderzeit in Sekunden
resolution: Auflösung als String (z.B. "1920x1080")
engine: Render-Engine Name
samples: Anzahl der Samples

**`display_image_from_data(self, image_data: bytes, render_time: float = 0.0, resolution: str = '', engine: str = 'BLENDER_EEVEE_NEXT', samples: int = 64)`**

Zeige ein Bild aus Binärdaten an.

Args:
image_data: Bilddaten als Bytes
render_time: Renderzeit in Sekunden
resolution: Auflösung als String
engine: Render-Engine Name
samples: Anzahl der Samples

**`clear_image(self)`**

Lösche das angezeigte Bild.

**`capture_live_viewport(self)`**

Erfasse Live-Daten aus dem Blender-Viewport.

Returns:
Viewport-Screenshot und -Daten

**`capture_live_render(self)`**

Erfasse Live-Render-Daten aus Blender.

Returns:
Render-Daten und -Informationen

### BlenderImageWidgetStub

Stub-Klasse wenn anywidget nicht verfügbar ist.

#### Methods

**`display_image_from_path(self, *args, **kwargs)`**

*No documentation available.*

**`display_image_from_data(self, *args, **kwargs)`**

*No documentation available.*

**`clear_image(self)`**

*No documentation available.*

## Constants

- **ANYWIDGET_AVAILABLE** (bool): `True`
