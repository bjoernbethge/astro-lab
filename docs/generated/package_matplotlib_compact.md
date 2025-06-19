# Matplotlib Package Documentation

Auto-generated documentation for installed package `matplotlib`

## Package Information

- **Version**: 3.10.3
- **Location**: D:\astro-lab\.venv\Lib\site-packages
- **Summary**: Python plotting package

## Submodules

### artist
Module: `matplotlib.artist`

#### Functions

- **`allow_rasterization(draw)`**
  Decorator for Artist.draw method. Provides routines

- **`cache(user_function, /)`**
  Simple lightweight unbounded cache.  Sometimes called "memoize".

- **`get(obj, property=None)`**
  Return the value of an `.Artist`'s *property*, or print all of them.

- **`getp(obj, property=None)`**
  Return the value of an `.Artist`'s *property*, or print all of them.

- **`kwdoc(artist)`**
  Inspect an `~matplotlib.artist.Artist` class (using `.ArtistInspector`) and

- **`namedtuple(typename, field_names, *, rename=False, defaults=None, module=None)`**
  Returns a new subclass of tuple with named fields.

- **`setp(obj, *args, file=None, **kwargs)`**
  Set one or more properties on an `.Artist`, or list allowed values.

- **`wraps(wrapped, assigned=('__module__', '__name__', '__qualname__', '__doc__', '__annotations__'), updated=('__dict__',))`**
  Decorator factory to apply update_wrapper() to a wrapper function

#### Classes

- **`Artist`**
  Abstract base class for objects that render into a FigureCanvas.
  Methods: remove, have_units, convert_xunits

- **`ArtistInspector`**
  A helper class to inspect an `~matplotlib.artist.Artist` and return
  Methods: get_aliases, get_valid_values, get_setters

- **`Bbox`**
  A mutable bounding box.
  Methods: frozen, unit, null

- **`BboxBase`**
  The base class of all bounding boxes.
  Methods: frozen, get_points, containsx

- **`IdentityTransform`**
  A special class that does one thing, the identity transform, in a
  Methods: frozen, get_matrix, transform

### backends
Module: `matplotlib.backends`

#### Classes

- **`BackendFilter`**
  Filter used with :meth:`~matplotlib.backends.registry.BackendRegistry.list_builtin`

#### Attributes

- **`backend_registry`** (BackendRegistry): `<matplotlib.backends.registry.BackendRegistry object at 0x000001359BF5A9D0>`

### bezier
Module: `matplotlib.bezier`

A module providing some utility functions regarding Bézier path manipulation.

#### Functions

- **`check_if_parallel(dx1, dy1, dx2, dy2, tolerance=1e-05)`**
  Check if two lines are parallel.

- **`find_bezier_t_intersecting_with_closedpath(bezier_point_at_t, inside_closedpath, t0=0.0, t1=1.0, tolerance=0.01)`**
  Find the intersection of the Bézier curve with a closed path.

- **`find_control_points(c1x, c1y, mmx, mmy, c2x, c2y)`**
  Find control points of the Bézier curve passing through (*c1x*, *c1y*),

- **`get_cos_sin(x0, y0, x1, y1)`**

- **`get_intersection(cx1, cy1, cos_t1, sin_t1, cx2, cy2, cos_t2, sin_t2)`**
  Return the intersection between the line through (*cx1*, *cy1*) at angle

- **`get_normal_points(cx, cy, cos_t, sin_t, length)`**
  For a line passing through (*cx*, *cy*) and having an angle *t*, return

- **`get_parallels(bezier2, width)`**
  Given the quadratic Bézier control points *bezier2*, returns

- **`inside_circle(cx, cy, r)`**
  Return a function that checks whether a point is in a circle with center

- **`lru_cache(maxsize=128, typed=False)`**
  Least-recently-used cache decorator.

- **`make_wedged_bezier2(bezier2, width, w1=1.0, wm=0.5, w2=0.0)`**
  Being similar to `get_parallels`, returns control points of two quadratic

#### Classes

- **`BezierSegment`**
  A d-dimensional Bézier segment.
  Methods: point_at_t, axis_aligned_extrema

- **`NonIntersectingPathException`**
  Inappropriate argument value (of correct type).

### cbook
Module: `matplotlib.cbook`

A collection of utility functions and classes.  Originally, many
(but not all) were from the Python Cookbook -- hence the name cbook.

#### Functions

- **`boxplot_stats(X, whis=1.5, bootstrap=None, labels=None, autorange=False)`**
  Return a list of dictionaries of statistics used to draw a series of box

- **`contiguous_regions(mask)`**
  Return a list of (ind0, ind1) such that ``mask[ind0:ind1].all()`` is

- **`delete_masked_points(*args)`**
  Find all masked and/or non-finite points in a set of arguments,

- **`file_requires_unicode(x)`**
  Return whether the given writable file-like object requires Unicode to be

- **`flatten(seq, scalarp=<function is_scalar_or_string at 0x00000135C97C6980>)`**
  Return a generator of flattened nested containers.

- **`get_sample_data(fname, asfileobj=True)`**
  Return a sample data file.  *fname* is a path relative to the

- **`index_of(y)`**
  A helper function to create reasonable x values for the given *y*.

- **`is_math_text(s)`**
  Return whether the string *s* contains math expressions.

- **`is_scalar_or_string(val)`**
  Return whether the given object is a scalar or string like.

- **`is_writable_file_like(obj)`**
  Return whether *obj* looks like a file object with a *write* method.

#### Classes

- **`CallbackRegistry`**
  Handle registering, processing, blocking, and disconnecting
  Methods: connect, disconnect, process

- **`Grouper`**
  A disjoint-set data structure.
  Methods: join, joined, remove

- **`GrouperView`**
  Immutable view over a `.Grouper`.
  Methods: joined, get_siblings

- **`Path`**
  PurePath subclass that can make system calls.
  Methods: samefile, iterdir, glob

- **`VisibleDeprecationWarning`**
  Visible deprecation warning.

#### Attributes

- **`STEP_LOOKUP_MAP`** (dict): `{'default': <function <lambda> at 0x00000135C97D04A0>, 'steps': <function pts_to_prestep at 0x000001...`
- **`ls_mapper`** (dict): `{'-': 'solid', '--': 'dashed', '-.': 'dashdot', ':': 'dotted'}`
- **`ls_mapper_r`** (dict): `{'solid': '-', 'dashed': '--', 'dashdot': '-.', 'dotted': ':'}`

### cm
Module: `matplotlib.cm`

Builtin colormaps, colormap handling utilities, and the `ScalarMappable` mixin.

.. seealso::

  :doc:`/gallery/color/colormap_reference` for a list of builtin colormaps.

  :ref:`colormap-manipulation` for examples of how to make
  colormaps.

  :ref:`colormaps` an in-depth discussion of choosing
  colormaps.

  :ref:`colormapnorms` for more details about data normalization.

#### Functions

- **`get_cmap(name=None, lut=None)`**
  [*Deprecated*] Get a colormap instance, defaulting to rc values if *name* is None.

#### Classes

- **`ColormapRegistry`**
  Container for colormaps that are known to Matplotlib by name.
  Methods: register, unregister, get_cmap

- **`Mapping`**
  A Mapping is a generic container for associating key/value
  Methods: get, keys, items

- **`ScalarMappable`**
  A mixin class to map one or multiple sets of scalar data to RGBA.
  Methods: set_array, get_array, changed

#### Attributes

- **`bivar_cmaps`** (dict): `{'BiPeak': <matplotlib.colors.SegmentedBivarColormap object at 0x00000135C9D011D0>, 'BiOrangeBlue': ...`
- **`cmaps_listed`** (dict): `{'magma': <matplotlib.colors.ListedColormap object at 0x00000135C9D94750>, 'inferno': <matplotlib.co...`
- **`datad`** (dict): `{'Blues': ((0.9686274509803922, 0.984313725490196, 1.0), (0.8705882352941177, 0.9215686274509803, 0....`
- **`multivar_cmaps`** (dict): `{'2VarAddA': <matplotlib.colors.MultivarColormap object at 0x00000135C9D00FD0>, '2VarSubA': <matplot...`

### colorizer
Module: `matplotlib.colorizer`

The Colorizer class which handles the data to color pipeline via a
normalization and a colormap.

.. admonition:: Provisional status of colorizer

    The ``colorizer`` module and classes in this file are considered
    provisional and may change at any time without a deprecation period.

.. seealso::

  :doc:`/gallery/color/colormap_reference` for a list of builtin colormaps.

  :ref:`colormap-manipulation` for examples of how to make colormaps.

  :ref:`colormaps` for an in-depth discussion of choosing colormaps.

  :ref:`colormapnorms` for more details about data normalization.

#### Classes

- **`Colorizer`**
  Data to color pipeline.
  Methods: to_rgba, autoscale, autoscale_None

- **`ColorizingArtist`**
  Base class for artists that make map data to color using a `.colorizer.Colorizer`.
  Methods: draw, set

### colors
Module: `matplotlib.colors`

A module for converting numbers or color arguments to *RGB* or *RGBA*.

*RGB* and *RGBA* are sequences of, respectively, 3 or 4 floats in the
range 0-1.

This module includes functions and classes for color specification conversions,
and for mapping numbers to colors in a 1-D array of colors called a colormap.

Mapping data onto colors using a colormap typically involves two steps: a data
array is first mapped onto the range 0-1 using a subclass of `Normalize`,
then this number is mapped to a color using a subclass of `Colormap`.  Two
subclasses of `Colormap` provided here:  `LinearSegmentedColormap`, which uses
piecewise-linear interpolation to define colormaps, and `ListedColormap`, which
makes a colormap from a list of colors.

.. seealso::

  :ref:`colormap-manipulation` for examples of how to
  make colormaps and

  :ref:`colormaps` for a list of built-in colormaps.

  :ref:`colormapnorms` for more details about data
  normalization

  More colormaps are available at palettable_.

The module also provides functions for checking whether an object can be
interpreted as a color (`is_color_like`), for converting such an object
to an RGBA tuple (`to_rgba`) or to an HTML-like hex string in the
"#rrggbb" format (`to_hex`), and a sequence of colors to an (n, 4)
RGBA array (`to_rgba_array`).  Caching is used for efficiency.

Colors that Matplotlib recognizes are listed at
:ref:`colors_def`.

.. _palettable: https://jiffyclub.github.io/palettable/
.. _xkcd color survey: https://xkcd.com/color/rgb/

#### Functions

- **`from_levels_and_colors(levels, colors, extend='neither')`**
  A helper routine to generate a cmap and a norm instance which

- **`get_named_colors_mapping()`**
  Return the global mapping of names to named colors.

- **`hex2color(c)`**
  Convert *c* to an RGB color, silently dropping the alpha channel.

- **`hsv_to_rgb(hsv)`**
  Convert HSV values to RGB.

- **`is_color_like(c)`**
  Return whether *c* can be interpreted as an RGB(A) color.

- **`make_norm_from_scale(scale_cls, base_norm_cls=None, *, init=None)`**
  Decorator for building a `.Normalize` subclass from a `~.scale.ScaleBase`

- **`rgb2hex(c, keep_alpha=False)`**
  Convert *c* to a hex color.

- **`rgb_to_hsv(arr)`**
  Convert an array of float RGB values (in the range [0, 1]) to HSV values.

- **`same_color(c1, c2)`**
  Return whether the colors *c1* and *c2* are the same.

- **`to_hex(c, keep_alpha=False)`**
  Convert *c* to a hex color.

#### Classes

- **`AsinhNorm`**
  The inverse hyperbolic sine scale is approximately linear near
  Methods: inverse, autoscale_None

- **`BivarColormap`**
  Base class for all bivariate to RGBA mappings.
  Methods: get_bad, get_outside, resampled

- **`BivarColormapFromImage`**
  BivarColormap object generated by supersampling a regular grid.

- **`BoundaryNorm`**
  Generate a colormap index based on discrete intervals.
  Methods: inverse

- **`CenteredNorm`**
  A class which, when called, maps values within the interval
  Methods: autoscale, autoscale_None

#### Attributes

- **`BASE_COLORS`** (dict): `{'b': (0, 0, 1), 'g': (0, 0.5, 0), 'r': (1, 0, 0), 'c': (0, 0.75, 0.75), 'm': (0.75, 0, 0.75), 'y': ...`
- **`CSS4_COLORS`** (dict): `{'aliceblue': '#F0F8FF', 'antiquewhite': '#FAEBD7', 'aqua': '#00FFFF', 'aquamarine': '#7FFFD4', 'azu...`
- **`TABLEAU_COLORS`** (dict): `{'tab:blue': '#1f77b4', 'tab:orange': '#ff7f0e', 'tab:green': '#2ca02c', 'tab:red': '#d62728', 'tab:...`
- **`XKCD_COLORS`** (dict): `{'xkcd:cloudy blue': '#acc2d9', 'xkcd:dark pastel green': '#56ae57', 'xkcd:dust': '#b2996e', 'xkcd:e...`
- **`cnames`** (dict): `{'aliceblue': '#F0F8FF', 'antiquewhite': '#FAEBD7', 'aqua': '#00FFFF', 'aquamarine': '#7FFFD4', 'azu...`

### ft2font
Module: `matplotlib.ft2font`

#### Classes

- **`FT2Font`**
  An object representing a single font face.
  Methods: clear, set_size, set_charmap

- **`FT2Image`**
  An image buffer for drawing glyphs.
  Methods: draw_rect_filled

- **`FaceFlags`**
  Flags returned by `FT2Font.face_flags`.

- **`Glyph`**
  Information about a single glyph.

- **`Kerning`**
  Kerning modes for `.FT2Font.get_kerning`.

### path
Module: `matplotlib.path`

A module for dealing with the polylines used throughout Matplotlib.

The primary class for polyline handling in Matplotlib is `Path`.  Almost all
vector drawing makes use of `Path`\s somewhere in the drawing pipeline.

Whilst a `Path` instance itself cannot be drawn, some `.Artist` subclasses,
such as `.PathPatch` and `.PathCollection`, can be used for convenient `Path`
visualisation.

#### Functions

- **`get_path_collection_extents(master_transform, paths, transforms, offsets, offset_transform)`**
  Get bounding box of a `.PathCollection`\s internal objects.

- **`lru_cache(maxsize=128, typed=False)`**
  Least-recently-used cache decorator.

- **`simple_linear_interpolation(a, steps)`**
  Resample an array with ``steps - 1`` points between original point pairs.

#### Classes

- **`BezierSegment`**
  A d-dimensional Bézier segment.
  Methods: point_at_t, axis_aligned_extrema

- **`Path`**
  A series of possibly disconnected, possibly closed, line and curve
  Methods: code_type, copy, deepcopy

- **`WeakValueDictionary`**
  Mapping class that references values weakly.
  Methods: copy, get, items

### rcsetup
Module: `matplotlib.rcsetup`

The rcsetup module contains the validation code for customization using
Matplotlib's rc settings.

Each rc setting is assigned a function used to validate any attempted changes
to that setting.  The validation functions are defined in the rcsetup module,
and are used to construct the rcParams global object which stores the settings
and is referenced throughout Matplotlib.

The default values of the rc settings are set in the default matplotlibrc file.
Any additions or deletions to the parameter set listed here should also be
propagated to the :file:`lib/matplotlib/mpl-data/matplotlibrc` in Matplotlib's
root source directory.

#### Functions

- **`ccycler(*args, **kwargs)`**
  Create a new `Cycler` object from a single positional argument,

- **`cycler(*args, **kwargs)`**
  Create a `~cycler.Cycler` object much like :func:`cycler.cycler`,

- **`is_color_like(c)`**
  Return whether *c* can be interpreted as an RGB(A) color.

- **`lru_cache(maxsize=128, typed=False)`**
  Least-recently-used cache decorator.

- **`validate_any(s)`**

- **`validate_anylist(s)`**

- **`validate_aspect(s)`**

- **`validate_axisbelow(s)`**

- **`validate_backend(s)`**

- **`validate_bbox(s)`**

#### Classes

- **`BackendFilter`**
  Filter used with :meth:`~matplotlib.backends.registry.BackendRegistry.list_builtin`

- **`CapStyle`**
  Define how the two endpoints (caps) of an unclosed line are drawn.
  Methods: demo

- **`Colormap`**
  Baseclass for all scalar to RGBA mappings.
  Methods: get_bad, set_bad, get_under

- **`Cycler`**
  Composable cycles.
  Methods: change_key, by_key, simplify

- **`JoinStyle`**
  Define how the connection between two line segments is drawn.
  Methods: demo

#### Attributes

- **`backend_registry`** (BackendRegistry): `<matplotlib.backends.registry.BackendRegistry object at 0x000001359BF5A9D0>`
- **`defaultParams`** (dict): `{'backend': [<object object at 0x000001359BF47040>, <function validate_backend at 0x00000135C9C43CE0...`
- **`ls_mapper`** (dict): `{'-': 'solid', '--': 'dashed', '-.': 'dashdot', ':': 'dotted'}`

### scale
Module: `matplotlib.scale`

Scales define the distribution of data values on an axis, e.g. a log scaling.

The mapping is implemented through `.Transform` subclasses.

The following scales are built-in:

.. _builtin_scales:

============= ===================== ================================ =================================
Name          Class                 Transform                        Inverted transform
============= ===================== ================================ =================================
"asinh"       `AsinhScale`          `AsinhTransform`                 `InvertedAsinhTransform`
"function"    `FuncScale`           `FuncTransform`                  `FuncTransform`
"functionlog" `FuncScaleLog`        `FuncTransform` + `LogTransform` `InvertedLogTransform` + `FuncTransform`
"linear"      `LinearScale`         `.IdentityTransform`             `.IdentityTransform`
"log"         `LogScale`            `LogTransform`                   `InvertedLogTransform`
"logit"       `LogitScale`          `LogitTransform`                 `LogisticTransform`
"symlog"      `SymmetricalLogScale` `SymmetricalLogTransform`        `InvertedSymmetricalLogTransform`
============= ===================== ================================ =================================

A user will often only use the scale name, e.g. when setting the scale through
`~.Axes.set_xscale`: ``ax.set_xscale("log")``.

See also the :ref:`scales examples <sphx_glr_gallery_scales>` in the documentation.

Custom scaling can be achieved through `FuncScale`, or by creating your own
`ScaleBase` subclass and corresponding transforms (see :doc:`/gallery/scales/custom_scale`).
Third parties can register their scales by name through `register_scale`.

#### Functions

- **`get_scale_names()`**
  Return the names of the available scales.

- **`register_scale(scale_class)`**
  Register a new kind of scale.

- **`scale_factory(scale, axis, **kwargs)`**
  Return a scale class by name.

#### Classes

- **`AsinhLocator`**
  Place ticks spaced evenly on an inverse-sinh scale.
  Methods: set_params, tick_values

- **`AsinhScale`**
  A quasi-logarithmic scale based on the inverse hyperbolic sine (asinh)
  Methods: get_transform, set_default_locators_and_formatters

- **`AsinhTransform`**
  Inverse hyperbolic-sine transformation used by `.AsinhScale`
  Methods: transform_non_affine, inverted

- **`AutoLocator`**
  Place evenly spaced ticks, with the step size and maximum number of ticks chosen

- **`AutoMinorLocator`**
  Place evenly spaced minor ticks, with the step size and maximum number of ticks
  Methods: tick_values

### ticker
Module: `matplotlib.ticker`

Tick locating and formatting
============================

This module contains classes for configuring tick locating and formatting.
Generic tick locators and formatters are provided, as well as domain specific
custom ones.

Although the locators know nothing about major or minor ticks, they are used
by the Axis class to support major and minor tick locating and formatting.

.. _tick_locating:
.. _locators:

Tick locating
-------------

The Locator class is the base class for all tick locators. The locators
handle autoscaling of the view limits based on the data limits, and the
choosing of tick locations. A useful semi-automatic tick locator is
`MultipleLocator`. It is initialized with a base, e.g., 10, and it picks
axis limits and ticks that are multiples of that base.

The Locator subclasses defined here are:

======================= =======================================================
`AutoLocator`           `MaxNLocator` with simple defaults. This is the default
                        tick locator for most plotting.
`MaxNLocator`           Finds up to a max number of intervals with ticks at
                        nice locations.
`LinearLocator`         Space ticks evenly from min to max.
`LogLocator`            Space ticks logarithmically from min to max.
`MultipleLocator`       Ticks and range are a multiple of base; either integer
                        or float.
`FixedLocator`          Tick locations are fixed.
`IndexLocator`          Locator for index plots (e.g., where
                        ``x = range(len(y))``).
`NullLocator`           No ticks.
`SymmetricalLogLocator` Locator for use with the symlog norm; works like
                        `LogLocator` for the part outside of the threshold and
                        adds 0 if inside the limits.
`AsinhLocator`          Locator for use with the asinh norm, attempting to
                        space ticks approximately uniformly.
`LogitLocator`          Locator for logit scaling.
`AutoMinorLocator`      Locator for minor ticks when the axis is linear and the
                        major ticks are uniformly spaced. Subdivides the major
                        tick interval into a specified number of minor
                        intervals, defaulting to 4 or 5 depending on the major
                        interval.
======================= =======================================================

There are a number of locators specialized for date locations - see
the :mod:`.dates` module.

You can define your own locator by deriving from Locator. You must
override the ``__call__`` method, which returns a sequence of locations,
and you will probably want to override the autoscale method to set the
view limits from the data limits.

If you want to override the default locator, use one of the above or a custom
locator and pass it to the x- or y-axis instance. The relevant methods are::

  ax.xaxis.set_major_locator(xmajor_locator)
  ax.xaxis.set_minor_locator(xminor_locator)
  ax.yaxis.set_major_locator(ymajor_locator)
  ax.yaxis.set_minor_locator(yminor_locator)

The default minor locator is `NullLocator`, i.e., no minor ticks on by default.

.. note::
    `Locator` instances should not be used with more than one
    `~matplotlib.axis.Axis` or `~matplotlib.axes.Axes`. So instead of::

        locator = MultipleLocator(5)
        ax.xaxis.set_major_locator(locator)
        ax2.xaxis.set_major_locator(locator)

    do the following instead::

        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax2.xaxis.set_major_locator(MultipleLocator(5))

.. _formatters:

Tick formatting
---------------

Tick formatting is controlled by classes derived from Formatter. The formatter
operates on a single tick value and returns a string to the axis.

========================= =====================================================
`NullFormatter`           No labels on the ticks.
`FixedFormatter`          Set the strings manually for the labels.
`FuncFormatter`           User defined function sets the labels.
`StrMethodFormatter`      Use string `format` method.
`FormatStrFormatter`      Use an old-style sprintf format string.
`ScalarFormatter`         Default formatter for scalars: autopick the format
                          string.
`LogFormatter`            Formatter for log axes.
`LogFormatterExponent`    Format values for log axis using
                          ``exponent = log_base(value)``.
`LogFormatterMathtext`    Format values for log axis using
                          ``exponent = log_base(value)`` using Math text.
`LogFormatterSciNotation` Format values for log axis using scientific notation.
`LogitFormatter`          Probability formatter.
`EngFormatter`            Format labels in engineering notation.
`PercentFormatter`        Format labels as a percentage.
========================= =====================================================

You can derive your own formatter from the Formatter base class by
simply overriding the ``__call__`` method. The formatter class has
access to the axis view and data limits.

To control the major and minor tick label formats, use one of the
following methods::

  ax.xaxis.set_major_formatter(xmajor_formatter)
  ax.xaxis.set_minor_formatter(xminor_formatter)
  ax.yaxis.set_major_formatter(ymajor_formatter)
  ax.yaxis.set_minor_formatter(yminor_formatter)

In addition to a `.Formatter` instance, `~.Axis.set_major_formatter` and
`~.Axis.set_minor_formatter` also accept a ``str`` or function.  ``str`` input
will be internally replaced with an autogenerated `.StrMethodFormatter` with
the input ``str``. For function input, a `.FuncFormatter` with the input
function will be generated and used.

See :doc:`/gallery/ticks/major_minor_demo` for an example of setting major
and minor ticks. See the :mod:`matplotlib.dates` module for more information
and examples of using date locators and formatters.

#### Functions

- **`scale_range(vmin, vmax, n=1, threshold=100)`**

#### Classes

- **`AsinhLocator`**
  Place ticks spaced evenly on an inverse-sinh scale.
  Methods: set_params, tick_values

- **`AutoLocator`**
  Place evenly spaced ticks, with the step size and maximum number of ticks chosen

- **`AutoMinorLocator`**
  Place evenly spaced minor ticks, with the step size and maximum number of ticks
  Methods: tick_values

- **`EngFormatter`**
  Format axis values using engineering prefixes to represent powers
  Methods: set_locs, get_offset, format_eng

- **`FixedFormatter`**
  Return fixed strings for tick labels based only on position, not value.
  Methods: get_offset, set_offset_string

### transforms
Module: `matplotlib.transforms`

Matplotlib includes a framework for arbitrary geometric transformations that is used to
determine the final position of all elements drawn on the canvas.

Transforms are composed into trees of `TransformNode` objects
whose actual value depends on their children.  When the contents of
children change, their parents are automatically invalidated.  The
next time an invalidated transform is accessed, it is recomputed to
reflect those changes.  This invalidation/caching approach prevents
unnecessary recomputations of transforms, and contributes to better
interactive performance.

For example, here is a graph of the transform tree used to plot data to the figure:

.. graphviz:: /api/transforms.dot
    :alt: Diagram of transform tree from data to figure coordinates.

The framework can be used for both affine and non-affine
transformations.  However, for speed, we want to use the backend
renderers to perform affine transformations whenever possible.
Therefore, it is possible to perform just the affine or non-affine
part of a transformation on a set of data.  The affine is always
assumed to occur after the non-affine.  For any transform::

  full transform == non-affine part + affine part

The backends are not expected to handle non-affine transformations
themselves.

See the tutorial :ref:`transforms_tutorial` for examples
of how to use transforms.

#### Functions

- **`blended_transform_factory(x_transform, y_transform)`**
  Create a new "blended" transform using *x_transform* to transform

- **`composite_transform_factory(a, b)`**
  Create a new composite transform that is the result of applying

- **`interval_contains(interval, val)`**
  Check, inclusively, whether an interval includes a given value.

- **`interval_contains_open(interval, val)`**
  Check, excluding endpoints, whether an interval includes a given value.

- **`nonsingular(vmin, vmax, expander=0.001, tiny=1e-15, increasing=True)`**
  Modify the endpoints of a range as needed to avoid singularities.

- **`offset_copy(trans, fig=None, x=0.0, y=0.0, units='inches')`**
  Return a new transform with an added offset.

#### Classes

- **`Affine2D`**
  A mutable 2D affine transformation.
  Methods: from_values, get_matrix, set_matrix

- **`Affine2DBase`**
  The base class of all 2D affine transformations.
  Methods: frozen, to_values, transform_affine

- **`AffineBase`**
  The base class of all affine transformations of any number of dimensions.
  Methods: transform, transform_affine, transform_non_affine

- **`AffineDeltaTransform`**
  A transform wrapper for transforming displacements between pairs of points.
  Methods: get_matrix

- **`Bbox`**
  A mutable bounding box.
  Methods: frozen, unit, null

#### Attributes

- **`DEBUG`** (bool): `False`

## Functions

### cycler(*args, **kwargs)
Module: `matplotlib.rcsetup`

Create a `~cycler.Cycler` object much like :func:`cycler.cycler`,
but includes input validation.

Call signatures::

  cycler(cycler)
  cycler(label=values, label2=values2, ...)
  cycler(label, values)

Form 1 copies a given `~cycler.Cycler` object.

Form 2 creates a `~cycler.Cycler` which cycles over one or more
properties simultaneously. If multiple properties are given, their
value lists must have the same length.

Form 3 creates a `~cycler.Cycler` for a single property. This form
exists for compatibility with the original cycler. Its use is
discouraged in favor of the kwarg form, i.e. ``cycler(label=values)``.

Parameters
----------
cycler : Cycler
    Copy constructor for Cycler.

label : str
    The property key. Must be a valid `.Artist` property.
    For example, 'color' or 'linestyle'. Aliases are allowed,
    such as 'c' for 'color' and 'lw' for 'linewidth'.

values : iterable
    Finite-length iterable of the property values. These values
    are validated and will raise a ValueError if invalid.

Returns
-------
Cycler
    A new :class:`~cycler.Cycler` for the given properties.

Examples
--------
Creating a cycler for a single property:

>>> c = cycler(color=['red', 'green', 'blue'])

Creating a cycler for simultaneously cycling over multiple properties
(e.g. red circle, green plus, blue cross):

>>> c = cycler(color=['red', 'green', 'blue'],
...            marker=['o', '+', 'x'])

### get_backend(*, auto_select=True)
Module: `matplotlib`

Return the name of the current backend.

Parameters
----------
auto_select : bool, default: True
    Whether to trigger backend resolution if no backend has been
    selected so far. If True, this ensures that a valid backend
    is returned. If False, this returns None if no backend has been
    selected so far.

    .. versionadded:: 3.10

    .. admonition:: Provisional

       The *auto_select* flag is provisional. It may be changed or removed
       without prior warning.

See Also
--------
matplotlib.use

### get_cachedir()
Module: `matplotlib`

Return the string path of the cache directory.

The procedure used to find the directory is the same as for
`get_configdir`, except using ``$XDG_CACHE_HOME``/``$HOME/.cache`` instead.

### get_configdir()
Module: `matplotlib`

Return the string path of the configuration directory.

The directory is chosen as follows:

1. If the MPLCONFIGDIR environment variable is supplied, choose that.
2. On Linux, follow the XDG specification and look first in
   ``$XDG_CONFIG_HOME``, if defined, or ``$HOME/.config``.  On other
   platforms, choose ``$HOME/.matplotlib``.
3. If the chosen directory exists and is writable, use that as the
   configuration directory.
4. Else, create a temporary directory, and use it as the configuration
   directory.

### get_data_path()
Module: `matplotlib`

Return the path to Matplotlib data.

### interactive(b)
Module: `matplotlib`

Set whether to redraw after every plotting command (e.g. `.pyplot.xlabel`).

### is_interactive()
Module: `matplotlib`

Return whether to redraw after every plotting command.

.. note::

    This function is only intended for use in backends. End users should
    use `.pyplot.isinteractive` instead.

### matplotlib_fname()
Module: `matplotlib`

Get the location of the config file.

The file location is determined in the following order

- ``$PWD/matplotlibrc``
- ``$MATPLOTLIBRC`` if it is not a directory
- ``$MATPLOTLIBRC/matplotlibrc``
- ``$MPLCONFIGDIR/matplotlibrc``
- On Linux,
    - ``$XDG_CONFIG_HOME/matplotlib/matplotlibrc`` (if ``$XDG_CONFIG_HOME``
      is defined)
    - or ``$HOME/.config/matplotlib/matplotlibrc`` (if ``$XDG_CONFIG_HOME``
      is not defined)
- On other platforms,
  - ``$HOME/.matplotlib/matplotlibrc`` if ``$HOME`` is defined
- Lastly, it looks in ``$MATPLOTLIBDATA/matplotlibrc``, which should always
  exist.

### rc(group, **kwargs)
Module: `matplotlib`

Set the current `.rcParams`.  *group* is the grouping for the rc, e.g.,
for ``lines.linewidth`` the group is ``lines``, for
``axes.facecolor``, the group is ``axes``, and so on.  Group may
also be a list or tuple of group names, e.g., (*xtick*, *ytick*).
*kwargs* is a dictionary attribute name/value pairs, e.g.,::

  rc('lines', linewidth=2, color='r')

sets the current `.rcParams` and is equivalent to::

  rcParams['lines.linewidth'] = 2
  rcParams['lines.color'] = 'r'

The following aliases are available to save typing for interactive users:

=====   =================
Alias   Property
=====   =================
'lw'    'linewidth'
'ls'    'linestyle'
'c'     'color'
'fc'    'facecolor'
'ec'    'edgecolor'
'mew'   'markeredgewidth'
'aa'    'antialiased'
=====   =================

Thus you could abbreviate the above call as::

      rc('lines', lw=2, c='r')

Note you can use python's kwargs dictionary facility to store
dictionaries of default parameters.  e.g., you can customize the
font rc as follows::

  font = {'family' : 'monospace',
          'weight' : 'bold',
          'size'   : 'larger'}
  rc('font', **font)  # pass in the font dict as kwargs

This enables you to easily switch between several configurations.  Use
``matplotlib.style.use('default')`` or :func:`~matplotlib.rcdefaults` to
restore the default `.rcParams` after changes.

Notes
-----
Similar functionality is available by using the normal dict interface, i.e.
``rcParams.update({"lines.linewidth": 2, ...})`` (but ``rcParams.update``
does not support abbreviations or grouping).

### rc_context(rc=None, fname=None)
Module: `matplotlib`

Return a context manager for temporarily changing rcParams.

The :rc:`backend` will not be reset by the context manager.

rcParams changed both through the context manager invocation and
in the body of the context will be reset on context exit.

Parameters
----------
rc : dict
    The rcParams to temporarily set.
fname : str or path-like
    A file with Matplotlib rc settings. If both *fname* and *rc* are given,
    settings from *rc* take precedence.

See Also
--------
:ref:`customizing-with-matplotlibrc-files`

Examples
--------
Passing explicit values via a dict::

    with mpl.rc_context({'interactive': False}):
        fig, ax = plt.subplots()
        ax.plot(range(3), range(3))
        fig.savefig('example.png')
        plt.close(fig)

Loading settings from a file::

     with mpl.rc_context(fname='print.rc'):
         plt.plot(x, y)  # uses 'print.rc'

Setting in the context body::

    with mpl.rc_context():
        # will be reset
        mpl.rcParams['lines.linewidth'] = 5
        plt.plot(x, y)

### rc_file(fname, *, use_default_template=True)
Module: `matplotlib`

Update `.rcParams` from file.

Style-blacklisted `.rcParams` (defined in
``matplotlib.style.core.STYLE_BLACKLIST``) are not updated.

Parameters
----------
fname : str or path-like
    A file with Matplotlib rc settings.

use_default_template : bool
    If True, initialize with default parameters before updating with those
    in the given file. If False, the current configuration persists
    and only the parameters specified in the file are updated.

### rc_file_defaults()
Module: `matplotlib`

Restore the `.rcParams` from the original rc file loaded by Matplotlib.

Style-blacklisted `.rcParams` (defined in
``matplotlib.style.core.STYLE_BLACKLIST``) are not updated.

### rc_params(fail_on_error=False)
Module: `matplotlib`

Construct a `RcParams` instance from the default Matplotlib rc file.

### rc_params_from_file(fname, fail_on_error=False, use_default_template=True)
Module: `matplotlib`

Construct a `RcParams` from file *fname*.

Parameters
----------
fname : str or path-like
    A file with Matplotlib rc settings.
fail_on_error : bool
    If True, raise an error when the parser fails to convert a parameter.
use_default_template : bool
    If True, initialize with default parameters before updating with those
    in the given file. If False, the configuration class only contains the
    parameters specified in the file. (Useful for updating dicts.)

### rcdefaults()
Module: `matplotlib`

Restore the `.rcParams` from Matplotlib's internal default style.

Style-blacklisted `.rcParams` (defined in
``matplotlib.style.core.STYLE_BLACKLIST``) are not updated.

See Also
--------
matplotlib.rc_file_defaults
    Restore the `.rcParams` from the rc file originally loaded by
    Matplotlib.
matplotlib.style.use
    Use a specific style file.  Call ``style.use('default')`` to restore
    the default style.

### sanitize_sequence(data)
Module: `matplotlib`

[*Deprecated*] 

Notes
-----
.. deprecated:: 3.10
   Use matplotlib.cbook.sanitize_sequence instead.\ 

### set_loglevel(level)
Module: `matplotlib`

Configure Matplotlib's logging levels.

Matplotlib uses the standard library `logging` framework under the root
logger 'matplotlib'.  This is a helper function to:

- set Matplotlib's root logger level
- set the root logger handler's level, creating the handler
  if it does not exist yet

Typically, one should call ``set_loglevel("info")`` or
``set_loglevel("debug")`` to get additional debugging information.

Users or applications that are installing their own logging handlers
may want to directly manipulate ``logging.getLogger('matplotlib')`` rather
than use this function.

Parameters
----------
level : {"notset", "debug", "info", "warning", "error", "critical"}
    The log level of the handler.

Notes
-----
The first time this function is called, an additional handler is attached
to Matplotlib's root handler; this handler is reused every time and this
function simply manipulates the logger and handler's level.

### use(backend, *, force=True)
Module: `matplotlib`

Select the backend used for rendering and GUI integration.

If pyplot is already imported, `~matplotlib.pyplot.switch_backend` is used
and if the new backend is different than the current backend, all Figures
will be closed.

Parameters
----------
backend : str
    The backend to switch to.  This can either be one of the standard
    backend names, which are case-insensitive:

    - interactive backends:
      GTK3Agg, GTK3Cairo, GTK4Agg, GTK4Cairo, MacOSX, nbAgg, notebook, QtAgg,
      QtCairo, TkAgg, TkCairo, WebAgg, WX, WXAgg, WXCairo, Qt5Agg, Qt5Cairo

    - non-interactive backends:
      agg, cairo, pdf, pgf, ps, svg, template

    or a string of the form: ``module://my.module.name``.

    notebook is a synonym for nbAgg.

    Switching to an interactive backend is not possible if an unrelated
    event loop has already been started (e.g., switching to GTK3Agg if a
    TkAgg window has already been opened).  Switching to a non-interactive
    backend is always possible.

force : bool, default: True
    If True (the default), raise an `ImportError` if the backend cannot be
    set up (either because it fails to import, or because an incompatible
    GUI interactive framework is already running); if False, silently
    ignore the failure.

See Also
--------
:ref:`backends`
matplotlib.get_backend
matplotlib.pyplot.switch_backend

### validate_backend(s)
Module: `matplotlib`

[*Deprecated*] 

Notes
-----
.. deprecated:: 3.10
   Use matplotlib.rcsetup.validate_backend instead.\ 

## Classes

### ExecutableNotFoundError
Module: `matplotlib`

Error raised when an executable that Matplotlib optionally
depends on can't be found.

### MatplotlibDeprecationWarning
Module: `matplotlib._api.deprecation`

A class for issuing deprecation warnings for Matplotlib users.

### RcParams
Module: `matplotlib`

A dict-like key-value store for config parameters, including validation.

Validating functions are defined and associated with rc parameters in
:mod:`matplotlib.rcsetup`.

The list of rcParams is:

- _internal.classic_mode
- agg.path.chunksize
- animation.bitrate
- animation.codec
- animation.convert_args
- animation.convert_path
- animation.embed_limit
- animation.ffmpeg_args
- animation.ffmpeg_path
- animation.frame_format
- animation.html
- animation.writer
- axes.autolimit_mode
- axes.axisbelow
- axes.edgecolor
- axes.facecolor
- axes.formatter.limits
- axes.formatter.min_exponent
- axes.formatter.offset_threshold
- axes.formatter.use_locale
- axes.formatter.use_mathtext
- axes.formatter.useoffset
- axes.grid
- axes.grid.axis
- axes.grid.which
- axes.labelcolor
- axes.labelpad
- axes.labelsize
- axes.labelweight
- axes.linewidth
- axes.prop_cycle
- axes.spines.bottom
- axes.spines.left
- axes.spines.right
- axes.spines.top
- axes.titlecolor
- axes.titlelocation
- axes.titlepad
- axes.titlesize
- axes.titleweight
- axes.titley
- axes.unicode_minus
- axes.xmargin
- axes.ymargin
- axes.zmargin
- axes3d.automargin
- axes3d.grid
- axes3d.mouserotationstyle
- axes3d.trackballborder
- axes3d.trackballsize
- axes3d.xaxis.panecolor
- axes3d.yaxis.panecolor
- axes3d.zaxis.panecolor
- backend
- backend_fallback
- boxplot.bootstrap
- boxplot.boxprops.color
- boxplot.boxprops.linestyle
- boxplot.boxprops.linewidth
- boxplot.capprops.color
- boxplot.capprops.linestyle
- boxplot.capprops.linewidth
- boxplot.flierprops.color
- boxplot.flierprops.linestyle
- boxplot.flierprops.linewidth
- boxplot.flierprops.marker
- boxplot.flierprops.markeredgecolor
- boxplot.flierprops.markeredgewidth
- boxplot.flierprops.markerfacecolor
- boxplot.flierprops.markersize
- boxplot.meanline
- boxplot.meanprops.color
- boxplot.meanprops.linestyle
- boxplot.meanprops.linewidth
- boxplot.meanprops.marker
- boxplot.meanprops.markeredgecolor
- boxplot.meanprops.markerfacecolor
- boxplot.meanprops.markersize
- boxplot.medianprops.color
- boxplot.medianprops.linestyle
- boxplot.medianprops.linewidth
- boxplot.notch
- boxplot.patchartist
- boxplot.showbox
- boxplot.showcaps
- boxplot.showfliers
- boxplot.showmeans
- boxplot.vertical
- boxplot.whiskerprops.color
- boxplot.whiskerprops.linestyle
- boxplot.whiskerprops.linewidth
- boxplot.whiskers
- contour.algorithm
- contour.corner_mask
- contour.linewidth
- contour.negative_linestyle
- date.autoformatter.day
- date.autoformatter.hour
- date.autoformatter.microsecond
- date.autoformatter.minute
- date.autoformatter.month
- date.autoformatter.second
- date.autoformatter.year
- date.converter
- date.epoch
- date.interval_multiples
- docstring.hardcopy
- errorbar.capsize
- figure.autolayout
- figure.constrained_layout.h_pad
- figure.constrained_layout.hspace
- figure.constrained_layout.use
- figure.constrained_layout.w_pad
- figure.constrained_layout.wspace
- figure.dpi
- figure.edgecolor
- figure.facecolor
- figure.figsize
- figure.frameon
- figure.hooks
- figure.labelsize
- figure.labelweight
- figure.max_open_warning
- figure.raise_window
- figure.subplot.bottom
- figure.subplot.hspace
- figure.subplot.left
- figure.subplot.right
- figure.subplot.top
- figure.subplot.wspace
- figure.titlesize
- figure.titleweight
- font.cursive
- font.family
- font.fantasy
- font.monospace
- font.sans-serif
- font.serif
- font.size
- font.stretch
- font.style
- font.variant
- font.weight
- grid.alpha
- grid.color
- grid.linestyle
- grid.linewidth
- hatch.color
- hatch.linewidth
- hist.bins
- image.aspect
- image.cmap
- image.composite_image
- image.interpolation
- image.interpolation_stage
- image.lut
- image.origin
- image.resample
- interactive
- keymap.back
- keymap.copy
- keymap.forward
- keymap.fullscreen
- keymap.grid
- keymap.grid_minor
- keymap.help
- keymap.home
- keymap.pan
- keymap.quit
- keymap.quit_all
- keymap.save
- keymap.xscale
- keymap.yscale
- keymap.zoom
- legend.borderaxespad
- legend.borderpad
- legend.columnspacing
- legend.edgecolor
- legend.facecolor
- legend.fancybox
- legend.fontsize
- legend.framealpha
- legend.frameon
- legend.handleheight
- legend.handlelength
- legend.handletextpad
- legend.labelcolor
- legend.labelspacing
- legend.loc
- legend.markerscale
- legend.numpoints
- legend.scatterpoints
- legend.shadow
- legend.title_fontsize
- lines.antialiased
- lines.color
- lines.dash_capstyle
- lines.dash_joinstyle
- lines.dashdot_pattern
- lines.dashed_pattern
- lines.dotted_pattern
- lines.linestyle
- lines.linewidth
- lines.marker
- lines.markeredgecolor
- lines.markeredgewidth
- lines.markerfacecolor
- lines.markersize
- lines.scale_dashes
- lines.solid_capstyle
- lines.solid_joinstyle
- macosx.window_mode
- markers.fillstyle
- mathtext.bf
- mathtext.bfit
- mathtext.cal
- mathtext.default
- mathtext.fallback
- mathtext.fontset
- mathtext.it
- mathtext.rm
- mathtext.sf
- mathtext.tt
- patch.antialiased
- patch.edgecolor
- patch.facecolor
- patch.force_edgecolor
- patch.linewidth
- path.effects
- path.simplify
- path.simplify_threshold
- path.sketch
- path.snap
- pcolor.shading
- pcolormesh.snap
- pdf.compression
- pdf.fonttype
- pdf.inheritcolor
- pdf.use14corefonts
- pgf.preamble
- pgf.rcfonts
- pgf.texsystem
- polaraxes.grid
- ps.distiller.res
- ps.fonttype
- ps.papersize
- ps.useafm
- ps.usedistiller
- savefig.bbox
- savefig.directory
- savefig.dpi
- savefig.edgecolor
- savefig.facecolor
- savefig.format
- savefig.orientation
- savefig.pad_inches
- savefig.transparent
- scatter.edgecolors
- scatter.marker
- svg.fonttype
- svg.hashsalt
- svg.id
- svg.image_inline
- text.antialiased
- text.color
- text.hinting
- text.hinting_factor
- text.kerning_factor
- text.latex.preamble
- text.parse_math
- text.usetex
- timezone
- tk.window_focus
- toolbar
- webagg.address
- webagg.open_in_browser
- webagg.port
- webagg.port_retries
- xaxis.labellocation
- xtick.alignment
- xtick.bottom
- xtick.color
- xtick.direction
- xtick.labelbottom
- xtick.labelcolor
- xtick.labelsize
- xtick.labeltop
- xtick.major.bottom
- xtick.major.pad
- xtick.major.size
- xtick.major.top
- xtick.major.width
- xtick.minor.bottom
- xtick.minor.ndivs
- xtick.minor.pad
- xtick.minor.size
- xtick.minor.top
- xtick.minor.visible
- xtick.minor.width
- xtick.top
- yaxis.labellocation
- ytick.alignment
- ytick.color
- ytick.direction
- ytick.labelcolor
- ytick.labelleft
- ytick.labelright
- ytick.labelsize
- ytick.left
- ytick.major.left
- ytick.major.pad
- ytick.major.right
- ytick.major.size
- ytick.major.width
- ytick.minor.left
- ytick.minor.ndivs
- ytick.minor.pad
- ytick.minor.right
- ytick.minor.size
- ytick.minor.visible
- ytick.minor.width
- ytick.right

See Also
--------
:ref:`customizing-with-matplotlibrc-files`

#### Methods

**`find_all(self, pattern)`**

Return the subset of this RcParams dictionary whose keys match,
using :func:`re.search`, the given ``pattern``.

.. note::

Changes to the returned dictionary are *not* propagated to
the parent RcParams dictionary.

**`copy(self)`**

Copy this RcParams instance.

## Attributes

### bivar_colormaps
Type: `ColormapRegistry`
Value: `ColormapRegistry; available colormaps:
'BiPeak', 'BiOrangeBlue', 'BiCone'`

Container for colormaps that are known to Matplotlib by name.

The universal registry instance is `matplotlib.colormaps`. There should be
no need for users to instantiate `.ColormapRegistry` themselves.

Read access uses a dict-like interface mapping names to `.Colormap`\s::

    import matplotlib as mpl
    cmap = mpl.colormaps['viridis']

Returned `.Colormap`\s are copies, so that their modification does not
change the global definition of the colormap.

Additional colormaps can be added via `.ColormapRegistry.register`::

    mpl.colormaps.register(my_colormap)

To get a list of all registered colormaps, you can do::

    from matplotlib import colormaps
    list(colormaps)

### color_sequences
Type: `ColorSequenceRegistry`
Value: `ColorSequenceRegistry; available colormaps:
'tab10', 'tab20', 'tab20b', 'tab20c', 'Pastel1', 'Pastel...`

Container for sequences of colors that are known to Matplotlib by name.

The universal registry instance is `matplotlib.color_sequences`. There
should be no need for users to instantiate `.ColorSequenceRegistry`
themselves.

Read access uses a dict-like interface mapping names to lists of colors::

    import matplotlib as mpl
    colors = mpl.color_sequences['tab10']

For a list of built in color sequences, see :doc:`/gallery/color/color_sequences`.
The returned lists are copies, so that their modification does not change
the global definition of the color sequence.

Additional color sequences can be added via
`.ColorSequenceRegistry.register`::

    mpl.color_sequences.register('rgb', ['r', 'g', 'b'])

### colormaps
Type: `ColormapRegistry`
Value: `ColormapRegistry; available colormaps:
'magma', 'inferno', 'plasma', 'viridis', 'cividis', 'twilight...`

Container for colormaps that are known to Matplotlib by name.

The universal registry instance is `matplotlib.colormaps`. There should be
no need for users to instantiate `.ColormapRegistry` themselves.

Read access uses a dict-like interface mapping names to `.Colormap`\s::

    import matplotlib as mpl
    cmap = mpl.colormaps['viridis']

Returned `.Colormap`\s are copies, so that their modification does not
change the global definition of the colormap.

Additional colormaps can be added via `.ColormapRegistry.register`::

    mpl.colormaps.register(my_colormap)

To get a list of all registered colormaps, you can do::

    from matplotlib import colormaps
    list(colormaps)

### defaultParams
Type: `dict`
Value: `{'backend': [<object object at 0x000001359BF47040>, <function validate_backend at 0x00000135C9C43CE0...`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### multivar_colormaps
Type: `ColormapRegistry`
Value: `ColormapRegistry; available colormaps:
'2VarAddA', '2VarSubA', '3VarAddA'`

Container for colormaps that are known to Matplotlib by name.

The universal registry instance is `matplotlib.colormaps`. There should be
no need for users to instantiate `.ColormapRegistry` themselves.

Read access uses a dict-like interface mapping names to `.Colormap`\s::

    import matplotlib as mpl
    cmap = mpl.colormaps['viridis']

Returned `.Colormap`\s are copies, so that their modification does not
change the global definition of the colormap.

Additional colormaps can be added via `.ColormapRegistry.register`::

    mpl.colormaps.register(my_colormap)

To get a list of all registered colormaps, you can do::

    from matplotlib import colormaps
    list(colormaps)

### rcParams
Type: `RcParams`
Value: `_internal.classic_mode: False
agg.path.chunksize: 0
animation.bitrate: -1
animation.codec: h264
anim...`

A dict-like key-value store for config parameters, including validation.

Validating functions are defined and associated with rc parameters in
:mod:`matplotlib.rcsetup`.

The list of rcParams is:

- _internal.classic_mode
- agg.path.chunksize
- animation.bitrate
- animation.codec
- animation.convert_args
- animation.convert_path
- animation.embed_limit
- animation.ffmpeg_args
- animation.ffmpeg_path
- animation.frame_format
- animation.html
- animation.writer
- axes.autolimit_mode
- axes.axisbelow
- axes.edgecolor
- axes.facecolor
- axes.formatter.limits
- axes.formatter.min_exponent
- axes.formatter.offset_threshold
- axes.formatter.use_locale
- axes.formatter.use_mathtext
- axes.formatter.useoffset
- axes.grid
- axes.grid.axis
- axes.grid.which
- axes.labelcolor
- axes.labelpad
- axes.labelsize
- axes.labelweight
- axes.linewidth
- axes.prop_cycle
- axes.spines.bottom
- axes.spines.left
- axes.spines.right
- axes.spines.top
- axes.titlecolor
- axes.titlelocation
- axes.titlepad
- axes.titlesize
- axes.titleweight
- axes.titley
- axes.unicode_minus
- axes.xmargin
- axes.ymargin
- axes.zmargin
- axes3d.automargin
- axes3d.grid
- axes3d.mouserotationstyle
- axes3d.trackballborder
- axes3d.trackballsize
- axes3d.xaxis.panecolor
- axes3d.yaxis.panecolor
- axes3d.zaxis.panecolor
- backend
- backend_fallback
- boxplot.bootstrap
- boxplot.boxprops.color
- boxplot.boxprops.linestyle
- boxplot.boxprops.linewidth
- boxplot.capprops.color
- boxplot.capprops.linestyle
- boxplot.capprops.linewidth
- boxplot.flierprops.color
- boxplot.flierprops.linestyle
- boxplot.flierprops.linewidth
- boxplot.flierprops.marker
- boxplot.flierprops.markeredgecolor
- boxplot.flierprops.markeredgewidth
- boxplot.flierprops.markerfacecolor
- boxplot.flierprops.markersize
- boxplot.meanline
- boxplot.meanprops.color
- boxplot.meanprops.linestyle
- boxplot.meanprops.linewidth
- boxplot.meanprops.marker
- boxplot.meanprops.markeredgecolor
- boxplot.meanprops.markerfacecolor
- boxplot.meanprops.markersize
- boxplot.medianprops.color
- boxplot.medianprops.linestyle
- boxplot.medianprops.linewidth
- boxplot.notch
- boxplot.patchartist
- boxplot.showbox
- boxplot.showcaps
- boxplot.showfliers
- boxplot.showmeans
- boxplot.vertical
- boxplot.whiskerprops.color
- boxplot.whiskerprops.linestyle
- boxplot.whiskerprops.linewidth
- boxplot.whiskers
- contour.algorithm
- contour.corner_mask
- contour.linewidth
- contour.negative_linestyle
- date.autoformatter.day
- date.autoformatter.hour
- date.autoformatter.microsecond
- date.autoformatter.minute
- date.autoformatter.month
- date.autoformatter.second
- date.autoformatter.year
- date.converter
- date.epoch
- date.interval_multiples
- docstring.hardcopy
- errorbar.capsize
- figure.autolayout
- figure.constrained_layout.h_pad
- figure.constrained_layout.hspace
- figure.constrained_layout.use
- figure.constrained_layout.w_pad
- figure.constrained_layout.wspace
- figure.dpi
- figure.edgecolor
- figure.facecolor
- figure.figsize
- figure.frameon
- figure.hooks
- figure.labelsize
- figure.labelweight
- figure.max_open_warning
- figure.raise_window
- figure.subplot.bottom
- figure.subplot.hspace
- figure.subplot.left
- figure.subplot.right
- figure.subplot.top
- figure.subplot.wspace
- figure.titlesize
- figure.titleweight
- font.cursive
- font.family
- font.fantasy
- font.monospace
- font.sans-serif
- font.serif
- font.size
- font.stretch
- font.style
- font.variant
- font.weight
- grid.alpha
- grid.color
- grid.linestyle
- grid.linewidth
- hatch.color
- hatch.linewidth
- hist.bins
- image.aspect
- image.cmap
- image.composite_image
- image.interpolation
- image.interpolation_stage
- image.lut
- image.origin
- image.resample
- interactive
- keymap.back
- keymap.copy
- keymap.forward
- keymap.fullscreen
- keymap.grid
- keymap.grid_minor
- keymap.help
- keymap.home
- keymap.pan
- keymap.quit
- keymap.quit_all
- keymap.save
- keymap.xscale
- keymap.yscale
- keymap.zoom
- legend.borderaxespad
- legend.borderpad
- legend.columnspacing
- legend.edgecolor
- legend.facecolor
- legend.fancybox
- legend.fontsize
- legend.framealpha
- legend.frameon
- legend.handleheight
- legend.handlelength
- legend.handletextpad
- legend.labelcolor
- legend.labelspacing
- legend.loc
- legend.markerscale
- legend.numpoints
- legend.scatterpoints
- legend.shadow
- legend.title_fontsize
- lines.antialiased
- lines.color
- lines.dash_capstyle
- lines.dash_joinstyle
- lines.dashdot_pattern
- lines.dashed_pattern
- lines.dotted_pattern
- lines.linestyle
- lines.linewidth
- lines.marker
- lines.markeredgecolor
- lines.markeredgewidth
- lines.markerfacecolor
- lines.markersize
- lines.scale_dashes
- lines.solid_capstyle
- lines.solid_joinstyle
- macosx.window_mode
- markers.fillstyle
- mathtext.bf
- mathtext.bfit
- mathtext.cal
- mathtext.default
- mathtext.fallback
- mathtext.fontset
- mathtext.it
- mathtext.rm
- mathtext.sf
- mathtext.tt
- patch.antialiased
- patch.edgecolor
- patch.facecolor
- patch.force_edgecolor
- patch.linewidth
- path.effects
- path.simplify
- path.simplify_threshold
- path.sketch
- path.snap
- pcolor.shading
- pcolormesh.snap
- pdf.compression
- pdf.fonttype
- pdf.inheritcolor
- pdf.use14corefonts
- pgf.preamble
- pgf.rcfonts
- pgf.texsystem
- polaraxes.grid
- ps.distiller.res
- ps.fonttype
- ps.papersize
- ps.useafm
- ps.usedistiller
- savefig.bbox
- savefig.directory
- savefig.dpi
- savefig.edgecolor
- savefig.facecolor
- savefig.format
- savefig.orientation
- savefig.pad_inches
- savefig.transparent
- scatter.edgecolors
- scatter.marker
- svg.fonttype
- svg.hashsalt
- svg.id
- svg.image_inline
- text.antialiased
- text.color
- text.hinting
- text.hinting_factor
- text.kerning_factor
- text.latex.preamble
- text.parse_math
- text.usetex
- timezone
- tk.window_focus
- toolbar
- webagg.address
- webagg.open_in_browser
- webagg.port
- webagg.port_retries
- xaxis.labellocation
- xtick.alignment
- xtick.bottom
- xtick.color
- xtick.direction
- xtick.labelbottom
- xtick.labelcolor
- xtick.labelsize
- xtick.labeltop
- xtick.major.bottom
- xtick.major.pad
- xtick.major.size
- xtick.major.top
- xtick.major.width
- xtick.minor.bottom
- xtick.minor.ndivs
- xtick.minor.pad
- xtick.minor.size
- xtick.minor.top
- xtick.minor.visible
- xtick.minor.width
- xtick.top
- yaxis.labellocation
- ytick.alignment
- ytick.color
- ytick.direction
- ytick.labelcolor
- ytick.labelleft
- ytick.labelright
- ytick.labelsize
- ytick.left
- ytick.major.left
- ytick.major.pad
- ytick.major.right
- ytick.major.size
- ytick.major.width
- ytick.minor.left
- ytick.minor.ndivs
- ytick.minor.pad
- ytick.minor.right
- ytick.minor.size
- ytick.minor.visible
- ytick.minor.width
- ytick.right

See Also
--------
:ref:`customizing-with-matplotlibrc-files`

### rcParamsDefault
Type: `RcParams`
Value: `_internal.classic_mode: False
agg.path.chunksize: 0
animation.bitrate: -1
animation.codec: h264
anim...`

A dict-like key-value store for config parameters, including validation.

Validating functions are defined and associated with rc parameters in
:mod:`matplotlib.rcsetup`.

The list of rcParams is:

- _internal.classic_mode
- agg.path.chunksize
- animation.bitrate
- animation.codec
- animation.convert_args
- animation.convert_path
- animation.embed_limit
- animation.ffmpeg_args
- animation.ffmpeg_path
- animation.frame_format
- animation.html
- animation.writer
- axes.autolimit_mode
- axes.axisbelow
- axes.edgecolor
- axes.facecolor
- axes.formatter.limits
- axes.formatter.min_exponent
- axes.formatter.offset_threshold
- axes.formatter.use_locale
- axes.formatter.use_mathtext
- axes.formatter.useoffset
- axes.grid
- axes.grid.axis
- axes.grid.which
- axes.labelcolor
- axes.labelpad
- axes.labelsize
- axes.labelweight
- axes.linewidth
- axes.prop_cycle
- axes.spines.bottom
- axes.spines.left
- axes.spines.right
- axes.spines.top
- axes.titlecolor
- axes.titlelocation
- axes.titlepad
- axes.titlesize
- axes.titleweight
- axes.titley
- axes.unicode_minus
- axes.xmargin
- axes.ymargin
- axes.zmargin
- axes3d.automargin
- axes3d.grid
- axes3d.mouserotationstyle
- axes3d.trackballborder
- axes3d.trackballsize
- axes3d.xaxis.panecolor
- axes3d.yaxis.panecolor
- axes3d.zaxis.panecolor
- backend
- backend_fallback
- boxplot.bootstrap
- boxplot.boxprops.color
- boxplot.boxprops.linestyle
- boxplot.boxprops.linewidth
- boxplot.capprops.color
- boxplot.capprops.linestyle
- boxplot.capprops.linewidth
- boxplot.flierprops.color
- boxplot.flierprops.linestyle
- boxplot.flierprops.linewidth
- boxplot.flierprops.marker
- boxplot.flierprops.markeredgecolor
- boxplot.flierprops.markeredgewidth
- boxplot.flierprops.markerfacecolor
- boxplot.flierprops.markersize
- boxplot.meanline
- boxplot.meanprops.color
- boxplot.meanprops.linestyle
- boxplot.meanprops.linewidth
- boxplot.meanprops.marker
- boxplot.meanprops.markeredgecolor
- boxplot.meanprops.markerfacecolor
- boxplot.meanprops.markersize
- boxplot.medianprops.color
- boxplot.medianprops.linestyle
- boxplot.medianprops.linewidth
- boxplot.notch
- boxplot.patchartist
- boxplot.showbox
- boxplot.showcaps
- boxplot.showfliers
- boxplot.showmeans
- boxplot.vertical
- boxplot.whiskerprops.color
- boxplot.whiskerprops.linestyle
- boxplot.whiskerprops.linewidth
- boxplot.whiskers
- contour.algorithm
- contour.corner_mask
- contour.linewidth
- contour.negative_linestyle
- date.autoformatter.day
- date.autoformatter.hour
- date.autoformatter.microsecond
- date.autoformatter.minute
- date.autoformatter.month
- date.autoformatter.second
- date.autoformatter.year
- date.converter
- date.epoch
- date.interval_multiples
- docstring.hardcopy
- errorbar.capsize
- figure.autolayout
- figure.constrained_layout.h_pad
- figure.constrained_layout.hspace
- figure.constrained_layout.use
- figure.constrained_layout.w_pad
- figure.constrained_layout.wspace
- figure.dpi
- figure.edgecolor
- figure.facecolor
- figure.figsize
- figure.frameon
- figure.hooks
- figure.labelsize
- figure.labelweight
- figure.max_open_warning
- figure.raise_window
- figure.subplot.bottom
- figure.subplot.hspace
- figure.subplot.left
- figure.subplot.right
- figure.subplot.top
- figure.subplot.wspace
- figure.titlesize
- figure.titleweight
- font.cursive
- font.family
- font.fantasy
- font.monospace
- font.sans-serif
- font.serif
- font.size
- font.stretch
- font.style
- font.variant
- font.weight
- grid.alpha
- grid.color
- grid.linestyle
- grid.linewidth
- hatch.color
- hatch.linewidth
- hist.bins
- image.aspect
- image.cmap
- image.composite_image
- image.interpolation
- image.interpolation_stage
- image.lut
- image.origin
- image.resample
- interactive
- keymap.back
- keymap.copy
- keymap.forward
- keymap.fullscreen
- keymap.grid
- keymap.grid_minor
- keymap.help
- keymap.home
- keymap.pan
- keymap.quit
- keymap.quit_all
- keymap.save
- keymap.xscale
- keymap.yscale
- keymap.zoom
- legend.borderaxespad
- legend.borderpad
- legend.columnspacing
- legend.edgecolor
- legend.facecolor
- legend.fancybox
- legend.fontsize
- legend.framealpha
- legend.frameon
- legend.handleheight
- legend.handlelength
- legend.handletextpad
- legend.labelcolor
- legend.labelspacing
- legend.loc
- legend.markerscale
- legend.numpoints
- legend.scatterpoints
- legend.shadow
- legend.title_fontsize
- lines.antialiased
- lines.color
- lines.dash_capstyle
- lines.dash_joinstyle
- lines.dashdot_pattern
- lines.dashed_pattern
- lines.dotted_pattern
- lines.linestyle
- lines.linewidth
- lines.marker
- lines.markeredgecolor
- lines.markeredgewidth
- lines.markerfacecolor
- lines.markersize
- lines.scale_dashes
- lines.solid_capstyle
- lines.solid_joinstyle
- macosx.window_mode
- markers.fillstyle
- mathtext.bf
- mathtext.bfit
- mathtext.cal
- mathtext.default
- mathtext.fallback
- mathtext.fontset
- mathtext.it
- mathtext.rm
- mathtext.sf
- mathtext.tt
- patch.antialiased
- patch.edgecolor
- patch.facecolor
- patch.force_edgecolor
- patch.linewidth
- path.effects
- path.simplify
- path.simplify_threshold
- path.sketch
- path.snap
- pcolor.shading
- pcolormesh.snap
- pdf.compression
- pdf.fonttype
- pdf.inheritcolor
- pdf.use14corefonts
- pgf.preamble
- pgf.rcfonts
- pgf.texsystem
- polaraxes.grid
- ps.distiller.res
- ps.fonttype
- ps.papersize
- ps.useafm
- ps.usedistiller
- savefig.bbox
- savefig.directory
- savefig.dpi
- savefig.edgecolor
- savefig.facecolor
- savefig.format
- savefig.orientation
- savefig.pad_inches
- savefig.transparent
- scatter.edgecolors
- scatter.marker
- svg.fonttype
- svg.hashsalt
- svg.id
- svg.image_inline
- text.antialiased
- text.color
- text.hinting
- text.hinting_factor
- text.kerning_factor
- text.latex.preamble
- text.parse_math
- text.usetex
- timezone
- tk.window_focus
- toolbar
- webagg.address
- webagg.open_in_browser
- webagg.port
- webagg.port_retries
- xaxis.labellocation
- xtick.alignment
- xtick.bottom
- xtick.color
- xtick.direction
- xtick.labelbottom
- xtick.labelcolor
- xtick.labelsize
- xtick.labeltop
- xtick.major.bottom
- xtick.major.pad
- xtick.major.size
- xtick.major.top
- xtick.major.width
- xtick.minor.bottom
- xtick.minor.ndivs
- xtick.minor.pad
- xtick.minor.size
- xtick.minor.top
- xtick.minor.visible
- xtick.minor.width
- xtick.top
- yaxis.labellocation
- ytick.alignment
- ytick.color
- ytick.direction
- ytick.labelcolor
- ytick.labelleft
- ytick.labelright
- ytick.labelsize
- ytick.left
- ytick.major.left
- ytick.major.pad
- ytick.major.right
- ytick.major.size
- ytick.major.width
- ytick.minor.left
- ytick.minor.ndivs
- ytick.minor.pad
- ytick.minor.right
- ytick.minor.size
- ytick.minor.visible
- ytick.minor.width
- ytick.right

See Also
--------
:ref:`customizing-with-matplotlibrc-files`

### rcParamsOrig
Type: `RcParams`
Value: `_internal.classic_mode: False
agg.path.chunksize: 0
animation.bitrate: -1
animation.codec: h264
anim...`

A dict-like key-value store for config parameters, including validation.

Validating functions are defined and associated with rc parameters in
:mod:`matplotlib.rcsetup`.

The list of rcParams is:

- _internal.classic_mode
- agg.path.chunksize
- animation.bitrate
- animation.codec
- animation.convert_args
- animation.convert_path
- animation.embed_limit
- animation.ffmpeg_args
- animation.ffmpeg_path
- animation.frame_format
- animation.html
- animation.writer
- axes.autolimit_mode
- axes.axisbelow
- axes.edgecolor
- axes.facecolor
- axes.formatter.limits
- axes.formatter.min_exponent
- axes.formatter.offset_threshold
- axes.formatter.use_locale
- axes.formatter.use_mathtext
- axes.formatter.useoffset
- axes.grid
- axes.grid.axis
- axes.grid.which
- axes.labelcolor
- axes.labelpad
- axes.labelsize
- axes.labelweight
- axes.linewidth
- axes.prop_cycle
- axes.spines.bottom
- axes.spines.left
- axes.spines.right
- axes.spines.top
- axes.titlecolor
- axes.titlelocation
- axes.titlepad
- axes.titlesize
- axes.titleweight
- axes.titley
- axes.unicode_minus
- axes.xmargin
- axes.ymargin
- axes.zmargin
- axes3d.automargin
- axes3d.grid
- axes3d.mouserotationstyle
- axes3d.trackballborder
- axes3d.trackballsize
- axes3d.xaxis.panecolor
- axes3d.yaxis.panecolor
- axes3d.zaxis.panecolor
- backend
- backend_fallback
- boxplot.bootstrap
- boxplot.boxprops.color
- boxplot.boxprops.linestyle
- boxplot.boxprops.linewidth
- boxplot.capprops.color
- boxplot.capprops.linestyle
- boxplot.capprops.linewidth
- boxplot.flierprops.color
- boxplot.flierprops.linestyle
- boxplot.flierprops.linewidth
- boxplot.flierprops.marker
- boxplot.flierprops.markeredgecolor
- boxplot.flierprops.markeredgewidth
- boxplot.flierprops.markerfacecolor
- boxplot.flierprops.markersize
- boxplot.meanline
- boxplot.meanprops.color
- boxplot.meanprops.linestyle
- boxplot.meanprops.linewidth
- boxplot.meanprops.marker
- boxplot.meanprops.markeredgecolor
- boxplot.meanprops.markerfacecolor
- boxplot.meanprops.markersize
- boxplot.medianprops.color
- boxplot.medianprops.linestyle
- boxplot.medianprops.linewidth
- boxplot.notch
- boxplot.patchartist
- boxplot.showbox
- boxplot.showcaps
- boxplot.showfliers
- boxplot.showmeans
- boxplot.vertical
- boxplot.whiskerprops.color
- boxplot.whiskerprops.linestyle
- boxplot.whiskerprops.linewidth
- boxplot.whiskers
- contour.algorithm
- contour.corner_mask
- contour.linewidth
- contour.negative_linestyle
- date.autoformatter.day
- date.autoformatter.hour
- date.autoformatter.microsecond
- date.autoformatter.minute
- date.autoformatter.month
- date.autoformatter.second
- date.autoformatter.year
- date.converter
- date.epoch
- date.interval_multiples
- docstring.hardcopy
- errorbar.capsize
- figure.autolayout
- figure.constrained_layout.h_pad
- figure.constrained_layout.hspace
- figure.constrained_layout.use
- figure.constrained_layout.w_pad
- figure.constrained_layout.wspace
- figure.dpi
- figure.edgecolor
- figure.facecolor
- figure.figsize
- figure.frameon
- figure.hooks
- figure.labelsize
- figure.labelweight
- figure.max_open_warning
- figure.raise_window
- figure.subplot.bottom
- figure.subplot.hspace
- figure.subplot.left
- figure.subplot.right
- figure.subplot.top
- figure.subplot.wspace
- figure.titlesize
- figure.titleweight
- font.cursive
- font.family
- font.fantasy
- font.monospace
- font.sans-serif
- font.serif
- font.size
- font.stretch
- font.style
- font.variant
- font.weight
- grid.alpha
- grid.color
- grid.linestyle
- grid.linewidth
- hatch.color
- hatch.linewidth
- hist.bins
- image.aspect
- image.cmap
- image.composite_image
- image.interpolation
- image.interpolation_stage
- image.lut
- image.origin
- image.resample
- interactive
- keymap.back
- keymap.copy
- keymap.forward
- keymap.fullscreen
- keymap.grid
- keymap.grid_minor
- keymap.help
- keymap.home
- keymap.pan
- keymap.quit
- keymap.quit_all
- keymap.save
- keymap.xscale
- keymap.yscale
- keymap.zoom
- legend.borderaxespad
- legend.borderpad
- legend.columnspacing
- legend.edgecolor
- legend.facecolor
- legend.fancybox
- legend.fontsize
- legend.framealpha
- legend.frameon
- legend.handleheight
- legend.handlelength
- legend.handletextpad
- legend.labelcolor
- legend.labelspacing
- legend.loc
- legend.markerscale
- legend.numpoints
- legend.scatterpoints
- legend.shadow
- legend.title_fontsize
- lines.antialiased
- lines.color
- lines.dash_capstyle
- lines.dash_joinstyle
- lines.dashdot_pattern
- lines.dashed_pattern
- lines.dotted_pattern
- lines.linestyle
- lines.linewidth
- lines.marker
- lines.markeredgecolor
- lines.markeredgewidth
- lines.markerfacecolor
- lines.markersize
- lines.scale_dashes
- lines.solid_capstyle
- lines.solid_joinstyle
- macosx.window_mode
- markers.fillstyle
- mathtext.bf
- mathtext.bfit
- mathtext.cal
- mathtext.default
- mathtext.fallback
- mathtext.fontset
- mathtext.it
- mathtext.rm
- mathtext.sf
- mathtext.tt
- patch.antialiased
- patch.edgecolor
- patch.facecolor
- patch.force_edgecolor
- patch.linewidth
- path.effects
- path.simplify
- path.simplify_threshold
- path.sketch
- path.snap
- pcolor.shading
- pcolormesh.snap
- pdf.compression
- pdf.fonttype
- pdf.inheritcolor
- pdf.use14corefonts
- pgf.preamble
- pgf.rcfonts
- pgf.texsystem
- polaraxes.grid
- ps.distiller.res
- ps.fonttype
- ps.papersize
- ps.useafm
- ps.usedistiller
- savefig.bbox
- savefig.directory
- savefig.dpi
- savefig.edgecolor
- savefig.facecolor
- savefig.format
- savefig.orientation
- savefig.pad_inches
- savefig.transparent
- scatter.edgecolors
- scatter.marker
- svg.fonttype
- svg.hashsalt
- svg.id
- svg.image_inline
- text.antialiased
- text.color
- text.hinting
- text.hinting_factor
- text.kerning_factor
- text.latex.preamble
- text.parse_math
- text.usetex
- timezone
- tk.window_focus
- toolbar
- webagg.address
- webagg.open_in_browser
- webagg.port
- webagg.port_retries
- xaxis.labellocation
- xtick.alignment
- xtick.bottom
- xtick.color
- xtick.direction
- xtick.labelbottom
- xtick.labelcolor
- xtick.labelsize
- xtick.labeltop
- xtick.major.bottom
- xtick.major.pad
- xtick.major.size
- xtick.major.top
- xtick.major.width
- xtick.minor.bottom
- xtick.minor.ndivs
- xtick.minor.pad
- xtick.minor.size
- xtick.minor.top
- xtick.minor.visible
- xtick.minor.width
- xtick.top
- yaxis.labellocation
- ytick.alignment
- ytick.color
- ytick.direction
- ytick.labelcolor
- ytick.labelleft
- ytick.labelright
- ytick.labelsize
- ytick.left
- ytick.major.left
- ytick.major.pad
- ytick.major.right
- ytick.major.size
- ytick.major.width
- ytick.minor.left
- ytick.minor.ndivs
- ytick.minor.pad
- ytick.minor.right
- ytick.minor.size
- ytick.minor.visible
- ytick.minor.width
- ytick.right

See Also
--------
:ref:`customizing-with-matplotlibrc-files`
