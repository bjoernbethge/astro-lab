# template

Part of `torch_geometric.torch_geometric`
Module: `torch_geometric.template`

## Functions (1)

### `module_from_template(module_name: str, template_path: str, tmp_dirname: str, **kwargs: Any) -> Any`

## Classes (3)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `Environment`

The core component of Jinja is the `Environment`.  It contains
important shared variables like configuration, filters, tests,
globals and others.  Instances of this class may be modified if
they are not shared and if no template was loaded so far.
Modifications on environments after the first template was loaded
will lead to surprising effects and undefined behavior.

Here are the possible initialization parameters:

    `block_start_string`
        The string marking the beginning of a block.  Defaults to ``'{%'``.

    `block_end_string`
        The string marking the end of a block.  Defaults to ``'%}'``.

    `variable_start_string`
        The string marking the beginning of a print statement.
        Defaults to ``'{{'``.

    `variable_end_string`
        The string marking the end of a print statement.  Defaults to
        ``'}}'``.

    `comment_start_string`
        The string marking the beginning of a comment.  Defaults to ``'{#'``.

    `comment_end_string`
        The string marking the end of a comment.  Defaults to ``'#}'``.

    `line_statement_prefix`
        If given and a string, this will be used as prefix for line based
        statements.  See also :ref:`line-statements`.

    `line_comment_prefix`
        If given and a string, this will be used as prefix for line based
        comments.  See also :ref:`line-statements`.

        .. versionadded:: 2.2

    `trim_blocks`
        If this is set to ``True`` the first newline after a block is
        removed (block, not variable tag!).  Defaults to `False`.

    `lstrip_blocks`
        If this is set to ``True`` leading spaces and tabs are stripped
        from the start of a line to a block.  Defaults to `False`.

    `newline_sequence`
        The sequence that starts a newline.  Must be one of ``'\r'``,
        ``'\n'`` or ``'\r\n'``.  The default is ``'\n'`` which is a
        useful default for Linux and OS X systems as well as web
        applications.

    `keep_trailing_newline`
        Preserve the trailing newline when rendering templates.
        The default is ``False``, which causes a single newline,
        if present, to be stripped from the end of the template.

        .. versionadded:: 2.7

    `extensions`
        List of Jinja extensions to use.  This can either be import paths
        as strings or extension classes.  For more information have a
        look at :ref:`the extensions documentation <jinja-extensions>`.

    `optimized`
        should the optimizer be enabled?  Default is ``True``.

    `undefined`
        :class:`Undefined` or a subclass of it that is used to represent
        undefined values in the template.

    `finalize`
        A callable that can be used to process the result of a variable
        expression before it is output.  For example one can convert
        ``None`` implicitly into an empty string here.

    `autoescape`
        If set to ``True`` the XML/HTML autoescaping feature is enabled by
        default.  For more details about autoescaping see
        :class:`~markupsafe.Markup`.  As of Jinja 2.4 this can also
        be a callable that is passed the template name and has to
        return ``True`` or ``False`` depending on autoescape should be
        enabled by default.

        .. versionchanged:: 2.4
           `autoescape` can now be a function

    `loader`
        The template loader for this environment.

    `cache_size`
        The size of the cache.  Per default this is ``400`` which means
        that if more than 400 templates are loaded the loader will clean
        out the least recently used template.  If the cache size is set to
        ``0`` templates are recompiled all the time, if the cache size is
        ``-1`` the cache will not be cleaned.

        .. versionchanged:: 2.8
           The cache size was increased to 400 from a low 50.

    `auto_reload`
        Some loaders load templates from locations where the template
        sources may change (ie: file system or database).  If
        ``auto_reload`` is set to ``True`` (default) every time a template is
        requested the loader checks if the source changed and if yes, it
        will reload the template.  For higher performance it's possible to
        disable that.

    `bytecode_cache`
        If set to a bytecode cache object, this object will provide a
        cache for the internal Jinja bytecode so that templates don't
        have to be parsed if they were not changed.

        See :ref:`bytecode-cache` for more information.

    `enable_async`
        If set to true this enables async template execution which
        allows using async functions and generators.

#### Methods

- **`code_generator_class(environment: 'Environment', name: Optional[str], filename: Optional[str], stream: Optional[TextIO] = None, defer_init: bool = False, optimized: bool = True) -> None`**
  Walks the abstract syntax tree and call visitor functions for every

- **`concat(iterable, /)`**
  Concatenate any number of strings.

- **`context_class(environment: 'Environment', parent: Dict[str, Any], name: Optional[str], blocks: Dict[str, Callable[[ForwardRef('Context')], Iterator[str]]], globals: Optional[MutableMapping[str, Any]] = None)`**
  The template context holds the variables of a template.  It stores the

### `FileSystemLoader`

Load templates from a directory in the file system.

The path can be relative or absolute. Relative paths are relative to
the current working directory.

.. code-block:: python

    loader = FileSystemLoader("templates")

A list of paths can be given. The directories will be searched in
order, stopping at the first matching template.

.. code-block:: python

    loader = FileSystemLoader(["/override/templates", "/default/templates"])

:param searchpath: A path, or list of paths, to the directory that
    contains the templates.
:param encoding: Use this encoding to read the text from template
    files.
:param followlinks: Follow symbolic links in the path.

.. versionchanged:: 2.8
    Added the ``followlinks`` parameter.

#### Methods

- **`get_source(self, environment: 'Environment', template: str) -> Tuple[str, str, Callable[[], bool]]`**
  Get the template source, filename and reload helper for a template.

- **`list_templates(self) -> List[str]`**
  Iterates over all templates.  If the loader does not support that
