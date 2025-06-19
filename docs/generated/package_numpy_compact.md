# Numpy Package Documentation

Auto-generated documentation for installed package `numpy`

## Package Information

- **Version**: 1.26.4
- **Location**: D:\astro-lab\.venv\Lib\site-packages
- **Summary**: Fundamental package for array computing in Python

## Submodules

### char
Module: `numpy.core.defchararray`

This module contains a set of functions for vectorized string
operations and methods.

.. note::
   The `chararray` class exists for backwards compatibility with
   Numarray, it is not recommended for new development. Starting from numpy
   1.4, if one needs arrays of strings, it is recommended to use arrays of
   `dtype` `object_`, `bytes_` or `str_`, and use the free functions
   in the `numpy.char` module for fast vectorized string operations.

Some methods will only be available if the corresponding string method is
available in your version of Python.

The preferred alias for `defchararray` is `numpy.char`.

### compat
Module: `numpy.compat`

Compatibility module.

This module contains duplicated code from Python itself or 3rd party
extensions, which may be included for the following reasons:

  * compatibility
  * we may only need a small subset of the copied library/module

### ctypeslib
Module: `numpy.ctypeslib`

============================
``ctypes`` Utility Functions
============================

See Also
--------
load_library : Load a C library.
ndpointer : Array restype/argtype with verification.
as_ctypes : Create a ctypes array from an ndarray.
as_array : Create an ndarray from a ctypes array.

References
----------
.. [1] "SciPy Cookbook: ctypes", https://scipy-cookbook.readthedocs.io/items/Ctypes.html

Examples
--------
Load the C library:

>>> _lib = np.ctypeslib.load_library('libmystuff', '.')     #doctest: +SKIP

Our result type, an ndarray that must be of type double, be 1-dimensional
and is C-contiguous in memory:

>>> array_1d_double = np.ctypeslib.ndpointer(
...                          dtype=np.double,
...                          ndim=1, flags='CONTIGUOUS')    #doctest: +SKIP

Our C-function typically takes an array and updates its values
in-place.  For example::

    void foo_func(double* x, int length)
    {
        int i;
        for (i = 0; i < length; i++) {
            x[i] = i*i;
        }
    }

We wrap it using:

>>> _lib.foo_func.restype = None                      #doctest: +SKIP
>>> _lib.foo_func.argtypes = [array_1d_double, c_int] #doctest: +SKIP

Then, we're ready to call ``foo_func``:

>>> out = np.empty(15, dtype=np.double)
>>> _lib.foo_func(out, len(out))                #doctest: +SKIP

### dtypes
Module: `numpy.dtypes`

DType classes and utility (:mod:`numpy.dtypes`)
===============================================

This module is home to specific dtypes related functionality and their classes.
For more general information about dtypes, also see `numpy.dtype` and
:ref:`arrays.dtypes`.

Similar to the builtin ``types`` module, this submodule defines types (classes)
that are not widely used directly.

.. versionadded:: NumPy 1.25

    The dtypes module is new in NumPy 1.25.  Previously DType classes were
    only accessible indirectly.


DType classes
-------------

The following are the classes of the corresponding NumPy dtype instances and
NumPy scalar types.  The classes can be used in ``isinstance`` checks and can
also be instantiated or used directly.  Direct use of these classes is not
typical, since their scalar counterparts (e.g. ``np.float64``) or strings
like ``"float64"`` can be used.

.. list-table::
    :header-rows: 1

    * - Group
      - DType class

    * - Boolean
      - ``BoolDType``

    * - Bit-sized integers
      - ``Int8DType``, ``UInt8DType``, ``Int16DType``, ``UInt16DType``,
        ``Int32DType``, ``UInt32DType``, ``Int64DType``, ``UInt64DType``

    * - C-named integers (may be aliases)
      - ``ByteDType``, ``UByteDType``, ``ShortDType``, ``UShortDType``,
        ``IntDType``, ``UIntDType``, ``LongDType``, ``ULongDType``,
        ``LongLongDType``, ``ULongLongDType``

    * - Floating point
      - ``Float16DType``, ``Float32DType``, ``Float64DType``,
        ``LongDoubleDType``

    * - Complex
      - ``Complex64DType``, ``Complex128DType``, ``CLongDoubleDType``

    * - Strings
      - ``BytesDType``, ``BytesDType``

    * - Times
      - ``DateTime64DType``, ``TimeDelta64DType``

    * - Others
      - ``ObjectDType``, ``VoidDType``

### emath
Module: `numpy.lib.scimath`

Wrapper functions to more user-friendly calling of certain math functions
whose output data-type is different than the input data-type in certain
domains of the input.

For example, for functions like `log` with branch cuts, the versions in this
module provide the mathematically valid answers in the complex plane::

  >>> import math
  >>> np.emath.log(-math.exp(1)) == (1+1j*math.pi)
  True

Similarly, `sqrt`, other base logarithms, `power` and trig functions are
correctly handled.  See their respective docstrings for specific examples.

Functions
---------

.. autosummary::
   :toctree: generated/

   sqrt
   log
   log2
   logn
   log10
   power
   arccos
   arcsin
   arctanh

### exceptions
Module: `numpy.exceptions`

Exceptions and Warnings (:mod:`numpy.exceptions`)
=================================================

General exceptions used by NumPy.  Note that some exceptions may be module
specific, such as linear algebra errors.

.. versionadded:: NumPy 1.25

    The exceptions module is new in NumPy 1.25.  Older exceptions remain
    available through the main NumPy namespace for compatibility.

.. currentmodule:: numpy.exceptions

Warnings
--------
.. autosummary::
   :toctree: generated/

   ComplexWarning             Given when converting complex to real.
   VisibleDeprecationWarning  Same as a DeprecationWarning, but more visible.

Exceptions
----------
.. autosummary::
   :toctree: generated/

    AxisError          Given when an axis was invalid.
    DTypePromotionError   Given when no common dtype could be found.
    TooHardError       Error specific to `numpy.shares_memory`.

### fft
Module: `numpy.fft`

Discrete Fourier Transform (:mod:`numpy.fft`)
=============================================

.. currentmodule:: numpy.fft

The SciPy module `scipy.fft` is a more comprehensive superset
of ``numpy.fft``, which includes only a basic set of routines.

Standard FFTs
-------------

.. autosummary::
   :toctree: generated/

   fft       Discrete Fourier transform.
   ifft      Inverse discrete Fourier transform.
   fft2      Discrete Fourier transform in two dimensions.
   ifft2     Inverse discrete Fourier transform in two dimensions.
   fftn      Discrete Fourier transform in N-dimensions.
   ifftn     Inverse discrete Fourier transform in N dimensions.

Real FFTs
---------

.. autosummary::
   :toctree: generated/

   rfft      Real discrete Fourier transform.
   irfft     Inverse real discrete Fourier transform.
   rfft2     Real discrete Fourier transform in two dimensions.
   irfft2    Inverse real discrete Fourier transform in two dimensions.
   rfftn     Real discrete Fourier transform in N dimensions.
   irfftn    Inverse real discrete Fourier transform in N dimensions.

Hermitian FFTs
--------------

.. autosummary::
   :toctree: generated/

   hfft      Hermitian discrete Fourier transform.
   ihfft     Inverse Hermitian discrete Fourier transform.

Helper routines
---------------

.. autosummary::
   :toctree: generated/

   fftfreq   Discrete Fourier Transform sample frequencies.
   rfftfreq  DFT sample frequencies (for usage with rfft, irfft).
   fftshift  Shift zero-frequency component to center of spectrum.
   ifftshift Inverse of fftshift.


Background information
----------------------

Fourier analysis is fundamentally a method for expressing a function as a
sum of periodic components, and for recovering the function from those
components.  When both the function and its Fourier transform are
replaced with discretized counterparts, it is called the discrete Fourier
transform (DFT).  The DFT has become a mainstay of numerical computing in
part because of a very fast algorithm for computing it, called the Fast
Fourier Transform (FFT), which was known to Gauss (1805) and was brought
to light in its current form by Cooley and Tukey [CT]_.  Press et al. [NR]_
provide an accessible introduction to Fourier analysis and its
applications.

Because the discrete Fourier transform separates its input into
components that contribute at discrete frequencies, it has a great number
of applications in digital signal processing, e.g., for filtering, and in
this context the discretized input to the transform is customarily
referred to as a *signal*, which exists in the *time domain*.  The output
is called a *spectrum* or *transform* and exists in the *frequency
domain*.

Implementation details
----------------------

There are many ways to define the DFT, varying in the sign of the
exponent, normalization, etc.  In this implementation, the DFT is defined
as

.. math::
   A_k =  \sum_{m=0}^{n-1} a_m \exp\left\{-2\pi i{mk \over n}\right\}
   \qquad k = 0,\ldots,n-1.

The DFT is in general defined for complex inputs and outputs, and a
single-frequency component at linear frequency :math:`f` is
represented by a complex exponential
:math:`a_m = \exp\{2\pi i\,f m\Delta t\}`, where :math:`\Delta t`
is the sampling interval.

The values in the result follow so-called "standard" order: If ``A =
fft(a, n)``, then ``A[0]`` contains the zero-frequency term (the sum of
the signal), which is always purely real for real inputs. Then ``A[1:n/2]``
contains the positive-frequency terms, and ``A[n/2+1:]`` contains the
negative-frequency terms, in order of decreasingly negative frequency.
For an even number of input points, ``A[n/2]`` represents both positive and
negative Nyquist frequency, and is also purely real for real input.  For
an odd number of input points, ``A[(n-1)/2]`` contains the largest positive
frequency, while ``A[(n+1)/2]`` contains the largest negative frequency.
The routine ``np.fft.fftfreq(n)`` returns an array giving the frequencies
of corresponding elements in the output.  The routine
``np.fft.fftshift(A)`` shifts transforms and their frequencies to put the
zero-frequency components in the middle, and ``np.fft.ifftshift(A)`` undoes
that shift.

When the input `a` is a time-domain signal and ``A = fft(a)``, ``np.abs(A)``
is its amplitude spectrum and ``np.abs(A)**2`` is its power spectrum.
The phase spectrum is obtained by ``np.angle(A)``.

The inverse DFT is defined as

.. math::
   a_m = \frac{1}{n}\sum_{k=0}^{n-1}A_k\exp\left\{2\pi i{mk\over n}\right\}
   \qquad m = 0,\ldots,n-1.

It differs from the forward transform by the sign of the exponential
argument and the default normalization by :math:`1/n`.

Type Promotion
--------------

`numpy.fft` promotes ``float32`` and ``complex64`` arrays to ``float64`` and
``complex128`` arrays respectively. For an FFT implementation that does not
promote input arrays, see `scipy.fftpack`.

Normalization
-------------

The argument ``norm`` indicates which direction of the pair of direct/inverse
transforms is scaled and with what normalization factor.
The default normalization (``"backward"``) has the direct (forward) transforms
unscaled and the inverse (backward) transforms scaled by :math:`1/n`. It is
possible to obtain unitary transforms by setting the keyword argument ``norm``
to ``"ortho"`` so that both direct and inverse transforms are scaled by
:math:`1/\sqrt{n}`. Finally, setting the keyword argument ``norm`` to
``"forward"`` has the direct transforms scaled by :math:`1/n` and the inverse
transforms unscaled (i.e. exactly opposite to the default ``"backward"``).
`None` is an alias of the default option ``"backward"`` for backward
compatibility.

Real and Hermitian transforms
-----------------------------

When the input is purely real, its transform is Hermitian, i.e., the
component at frequency :math:`f_k` is the complex conjugate of the
component at frequency :math:`-f_k`, which means that for real
inputs there is no information in the negative frequency components that
is not already available from the positive frequency components.
The family of `rfft` functions is
designed to operate on real inputs, and exploits this symmetry by
computing only the positive frequency components, up to and including the
Nyquist frequency.  Thus, ``n`` input points produce ``n/2+1`` complex
output points.  The inverses of this family assumes the same symmetry of
its input, and for an output of ``n`` points uses ``n/2+1`` input points.

Correspondingly, when the spectrum is purely real, the signal is
Hermitian.  The `hfft` family of functions exploits this symmetry by
using ``n/2+1`` complex points in the input (time) domain for ``n`` real
points in the frequency domain.

In higher dimensions, FFTs are used, e.g., for image analysis and
filtering.  The computational efficiency of the FFT means that it can
also be a faster way to compute large convolutions, using the property
that a convolution in the time domain is equivalent to a point-by-point
multiplication in the frequency domain.

Higher dimensions
-----------------

In two dimensions, the DFT is defined as

.. math::
   A_{kl} =  \sum_{m=0}^{M-1} \sum_{n=0}^{N-1}
   a_{mn}\exp\left\{-2\pi i \left({mk\over M}+{nl\over N}\right)\right\}
   \qquad k = 0, \ldots, M-1;\quad l = 0, \ldots, N-1,

which extends in the obvious way to higher dimensions, and the inverses
in higher dimensions also extend in the same way.

References
----------

.. [CT] Cooley, James W., and John W. Tukey, 1965, "An algorithm for the
        machine calculation of complex Fourier series," *Math. Comput.*
        19: 297-301.

.. [NR] Press, W., Teukolsky, S., Vetterline, W.T., and Flannery, B.P.,
        2007, *Numerical Recipes: The Art of Scientific Computing*, ch.
        12-13.  Cambridge Univ. Press, Cambridge, UK.

Examples
--------

For examples, see the various functions.

### lib
Module: `numpy.lib`

**Note:** almost all functions in the ``numpy.lib`` namespace
are also present in the main ``numpy`` namespace.  Please use the
functions as ``np.<funcname>`` where possible.

``numpy.lib`` is mostly a space for implementing functions that don't
belong in core or in another NumPy submodule with a clear purpose
(e.g. ``random``, ``fft``, ``linalg``, ``ma``).

Most contains basic functions that are used by several submodules and are
useful to have in the main name-space.

### linalg
Module: `numpy.linalg`

``numpy.linalg``
================

The NumPy linear algebra functions rely on BLAS and LAPACK to provide efficient
low level implementations of standard linear algebra algorithms. Those
libraries may be provided by NumPy itself using C versions of a subset of their
reference implementations but, when possible, highly optimized libraries that
take advantage of specialized processor functionality are preferred. Examples
of such libraries are OpenBLAS, MKL (TM), and ATLAS. Because those libraries
are multithreaded and processor dependent, environmental variables and external
packages such as threadpoolctl may be needed to control the number of threads
or specify the processor architecture.

- OpenBLAS: https://www.openblas.net/
- threadpoolctl: https://github.com/joblib/threadpoolctl

Please note that the most-used linear algebra functions in NumPy are present in
the main ``numpy`` namespace rather than in ``numpy.linalg``.  There are:
``dot``, ``vdot``, ``inner``, ``outer``, ``matmul``, ``tensordot``, ``einsum``,
``einsum_path`` and ``kron``.

Functions present in numpy.linalg are listed below.


Matrix and vector products
--------------------------

   multi_dot
   matrix_power

Decompositions
--------------

   cholesky
   qr
   svd

Matrix eigenvalues
------------------

   eig
   eigh
   eigvals
   eigvalsh

Norms and other numbers
-----------------------

   norm
   cond
   det
   matrix_rank
   slogdet

Solving equations and inverting matrices
----------------------------------------

   solve
   tensorsolve
   lstsq
   inv
   pinv
   tensorinv

Exceptions
----------

   LinAlgError

### ma
Module: `numpy.ma`

=============
Masked Arrays
=============

Arrays sometimes contain invalid or missing data.  When doing operations
on such arrays, we wish to suppress invalid values, which is the purpose masked
arrays fulfill (an example of typical use is given below).

For example, examine the following array:

>>> x = np.array([2, 1, 3, np.nan, 5, 2, 3, np.nan])

When we try to calculate the mean of the data, the result is undetermined:

>>> np.mean(x)
nan

The mean is calculated using roughly ``np.sum(x)/len(x)``, but since
any number added to ``NaN`` [1]_ produces ``NaN``, this doesn't work.  Enter
masked arrays:

>>> m = np.ma.masked_array(x, np.isnan(x))
>>> m
masked_array(data = [2.0 1.0 3.0 -- 5.0 2.0 3.0 --],
      mask = [False False False  True False False False  True],
      fill_value=1e+20)

Here, we construct a masked array that suppress all ``NaN`` values.  We
may now proceed to calculate the mean of the other values:

>>> np.mean(m)
2.6666666666666665

.. [1] Not-a-Number, a floating point value that is the result of an
       invalid operation.

.. moduleauthor:: Pierre Gerard-Marchant
.. moduleauthor:: Jarrod Millman

### polynomial
Module: `numpy.polynomial`

A sub-package for efficiently dealing with polynomials.

Within the documentation for this sub-package, a "finite power series,"
i.e., a polynomial (also referred to simply as a "series") is represented
by a 1-D numpy array of the polynomial's coefficients, ordered from lowest
order term to highest.  For example, array([1,2,3]) represents
``P_0 + 2*P_1 + 3*P_2``, where P_n is the n-th order basis polynomial
applicable to the specific module in question, e.g., `polynomial` (which
"wraps" the "standard" basis) or `chebyshev`.  For optimal performance,
all operations on polynomials, including evaluation at an argument, are
implemented as operations on the coefficients.  Additional (module-specific)
information can be found in the docstring for the module of interest.

This package provides *convenience classes* for each of six different kinds
of polynomials:

         ========================    ================
         **Name**                    **Provides**
         ========================    ================
         `~polynomial.Polynomial`    Power series
         `~chebyshev.Chebyshev`      Chebyshev series
         `~legendre.Legendre`        Legendre series
         `~laguerre.Laguerre`        Laguerre series
         `~hermite.Hermite`          Hermite series
         `~hermite_e.HermiteE`       HermiteE series
         ========================    ================

These *convenience classes* provide a consistent interface for creating,
manipulating, and fitting data with polynomials of different bases.
The convenience classes are the preferred interface for the `~numpy.polynomial`
package, and are available from the ``numpy.polynomial`` namespace.
This eliminates the need to navigate to the corresponding submodules, e.g.
``np.polynomial.Polynomial`` or ``np.polynomial.Chebyshev`` instead of
``np.polynomial.polynomial.Polynomial`` or
``np.polynomial.chebyshev.Chebyshev``, respectively.
The classes provide a more consistent and concise interface than the
type-specific functions defined in the submodules for each type of polynomial.
For example, to fit a Chebyshev polynomial with degree ``1`` to data given
by arrays ``xdata`` and ``ydata``, the
`~chebyshev.Chebyshev.fit` class method::

    >>> from numpy.polynomial import Chebyshev
    >>> c = Chebyshev.fit(xdata, ydata, deg=1)

is preferred over the `chebyshev.chebfit` function from the
``np.polynomial.chebyshev`` module::

    >>> from numpy.polynomial.chebyshev import chebfit
    >>> c = chebfit(xdata, ydata, deg=1)

See :doc:`routines.polynomials.classes` for more details.

Convenience Classes
===================

The following lists the various constants and methods common to all of
the classes representing the various kinds of polynomials. In the following,
the term ``Poly`` represents any one of the convenience classes (e.g.
`~polynomial.Polynomial`, `~chebyshev.Chebyshev`, `~hermite.Hermite`, etc.)
while the lowercase ``p`` represents an **instance** of a polynomial class.

Constants
---------

- ``Poly.domain``     -- Default domain
- ``Poly.window``     -- Default window
- ``Poly.basis_name`` -- String used to represent the basis
- ``Poly.maxpower``   -- Maximum value ``n`` such that ``p**n`` is allowed
- ``Poly.nickname``   -- String used in printing

Creation
--------

Methods for creating polynomial instances.

- ``Poly.basis(degree)``    -- Basis polynomial of given degree
- ``Poly.identity()``       -- ``p`` where ``p(x) = x`` for all ``x``
- ``Poly.fit(x, y, deg)``   -- ``p`` of degree ``deg`` with coefficients
  determined by the least-squares fit to the data ``x``, ``y``
- ``Poly.fromroots(roots)`` -- ``p`` with specified roots
- ``p.copy()``              -- Create a copy of ``p``

Conversion
----------

Methods for converting a polynomial instance of one kind to another.

- ``p.cast(Poly)``    -- Convert ``p`` to instance of kind ``Poly``
- ``p.convert(Poly)`` -- Convert ``p`` to instance of kind ``Poly`` or map
  between ``domain`` and ``window``

Calculus
--------
- ``p.deriv()`` -- Take the derivative of ``p``
- ``p.integ()`` -- Integrate ``p``

Validation
----------
- ``Poly.has_samecoef(p1, p2)``   -- Check if coefficients match
- ``Poly.has_samedomain(p1, p2)`` -- Check if domains match
- ``Poly.has_sametype(p1, p2)``   -- Check if types match
- ``Poly.has_samewindow(p1, p2)`` -- Check if windows match

Misc
----
- ``p.linspace()`` -- Return ``x, p(x)`` at equally-spaced points in ``domain``
- ``p.mapparms()`` -- Return the parameters for the linear mapping between
  ``domain`` and ``window``.
- ``p.roots()``    -- Return the roots of `p`.
- ``p.trim()``     -- Remove trailing coefficients.
- ``p.cutdeg(degree)`` -- Truncate p to given degree
- ``p.truncate(size)`` -- Truncate p to given size

### random
Module: `numpy.random`

========================
Random Number Generation
========================

Use ``default_rng()`` to create a `Generator` and call its methods.

=============== =========================================================
Generator
--------------- ---------------------------------------------------------
Generator       Class implementing all of the random number distributions
default_rng     Default constructor for ``Generator``
=============== =========================================================

============================================= ===
BitGenerator Streams that work with Generator
--------------------------------------------- ---
MT19937
PCG64
PCG64DXSM
Philox
SFC64
============================================= ===

============================================= ===
Getting entropy to initialize a BitGenerator
--------------------------------------------- ---
SeedSequence
============================================= ===


Legacy
------

For backwards compatibility with previous versions of numpy before 1.17, the
various aliases to the global `RandomState` methods are left alone and do not
use the new `Generator` API.

==================== =========================================================
Utility functions
-------------------- ---------------------------------------------------------
random               Uniformly distributed floats over ``[0, 1)``
bytes                Uniformly distributed random bytes.
permutation          Randomly permute a sequence / generate a random sequence.
shuffle              Randomly permute a sequence in place.
choice               Random sample from 1-D array.
==================== =========================================================

==================== =========================================================
Compatibility
functions - removed
in the new API
-------------------- ---------------------------------------------------------
rand                 Uniformly distributed values.
randn                Normally distributed values.
ranf                 Uniformly distributed floating point numbers.
random_integers      Uniformly distributed integers in a given range.
                     (deprecated, use ``integers(..., closed=True)`` instead)
random_sample        Alias for `random_sample`
randint              Uniformly distributed integers in a given range
seed                 Seed the legacy random number generator.
==================== =========================================================

==================== =========================================================
Univariate
distributions
-------------------- ---------------------------------------------------------
beta                 Beta distribution over ``[0, 1]``.
binomial             Binomial distribution.
chisquare            :math:`\chi^2` distribution.
exponential          Exponential distribution.
f                    F (Fisher-Snedecor) distribution.
gamma                Gamma distribution.
geometric            Geometric distribution.
gumbel               Gumbel distribution.
hypergeometric       Hypergeometric distribution.
laplace              Laplace distribution.
logistic             Logistic distribution.
lognormal            Log-normal distribution.
logseries            Logarithmic series distribution.
negative_binomial    Negative binomial distribution.
noncentral_chisquare Non-central chi-square distribution.
noncentral_f         Non-central F distribution.
normal               Normal / Gaussian distribution.
pareto               Pareto distribution.
poisson              Poisson distribution.
power                Power distribution.
rayleigh             Rayleigh distribution.
triangular           Triangular distribution.
uniform              Uniform distribution.
vonmises             Von Mises circular distribution.
wald                 Wald (inverse Gaussian) distribution.
weibull              Weibull distribution.
zipf                 Zipf's distribution over ranked data.
==================== =========================================================

==================== ==========================================================
Multivariate
distributions
-------------------- ----------------------------------------------------------
dirichlet            Multivariate generalization of Beta distribution.
multinomial          Multivariate generalization of the binomial distribution.
multivariate_normal  Multivariate generalization of the normal distribution.
==================== ==========================================================

==================== =========================================================
Standard
distributions
-------------------- ---------------------------------------------------------
standard_cauchy      Standard Cauchy-Lorentz distribution.
standard_exponential Standard exponential distribution.
standard_gamma       Standard Gamma distribution.
standard_normal      Standard normal distribution.
standard_t           Standard Student's t-distribution.
==================== =========================================================

==================== =========================================================
Internal functions
-------------------- ---------------------------------------------------------
get_state            Get tuple representing internal state of generator.
set_state            Set state of generator.
==================== =========================================================

### rec
Module: `numpy.core.records`

Record Arrays
=============
Record arrays expose the fields of structured arrays as properties.

Most commonly, ndarrays contain elements of a single type, e.g. floats,
integers, bools etc.  However, it is possible for elements to be combinations
of these using structured types, such as::

  >>> a = np.array([(1, 2.0), (1, 2.0)], dtype=[('x', np.int64), ('y', np.float64)])
  >>> a
  array([(1, 2.), (1, 2.)], dtype=[('x', '<i8'), ('y', '<f8')])

Here, each element consists of two fields: x (and int), and y (a float).
This is known as a structured array.  The different fields are analogous
to columns in a spread-sheet.  The different fields can be accessed as
one would a dictionary::

  >>> a['x']
  array([1, 1])

  >>> a['y']
  array([2., 2.])

Record arrays allow us to access fields as properties::

  >>> ar = np.rec.array(a)

  >>> ar.x
  array([1, 1])

  >>> ar.y
  array([2., 2.])

### testing
Module: `numpy.testing`

Common test support for all numpy test scripts.

This single module should provide all the common functionality for numpy tests
in a single location, so that test scripts can just import it and work right
away.

### version
Module: `numpy.version`

## Functions

### add_newdoc(place, obj, doc, warn_on_python=True)
Module: `numpy.core.function_base`

Add documentation to an existing object, typically one defined in C

The purpose is to allow easier editing of the docstrings without requiring
a re-compile. This exists primarily for internal use within numpy itself.

Parameters
----------
place : str
    The absolute name of the module to import from
obj : str
    The name of the object to add documentation to, typically a class or
    function name
doc : {str, Tuple[str, str], List[Tuple[str, str]]}
    If a string, the documentation to apply to `obj`

    If a tuple, then the first element is interpreted as an attribute of
    `obj` and the second as the docstring to apply - ``(method, docstring)``

    If a list, then each element of the list should be a tuple of length
    two - ``[(method1, docstring1), (method2, docstring2), ...]``
warn_on_python : bool
    If True, the default, emit `UserWarning` if this is used to attach
    documentation to a pure-python object.

Notes
-----
This routine never raises an error if the docstring can't be written, but
will raise an error if the object being documented does not exist.

This routine cannot modify read-only docstrings, as appear
in new-style classes or built-in functions. Because this
routine never raises an error the caller must check manually
that the docstrings were changed.

Since this function grabs the ``char *`` from a c-level str object and puts
it into the ``tp_doc`` slot of the type of `obj`, it violates a number of
C-API best-practices, by:

- modifying a `PyTypeObject` after calling `PyType_Ready`
- calling `Py_INCREF` on the str and losing the reference, so the str
  will never be released

If possible it should be avoided.

### asarray_chkfinite(a, dtype=None, order=None)
Module: `numpy`

Convert the input to an array, checking for NaNs or Infs.

Parameters
----------
a : array_like
    Input data, in any form that can be converted to an array.  This
    includes lists, lists of tuples, tuples, tuples of tuples, tuples
    of lists and ndarrays.  Success requires no NaNs or Infs.
dtype : data-type, optional
    By default, the data-type is inferred from the input data.
order : {'C', 'F', 'A', 'K'}, optional
    Memory layout.  'A' and 'K' depend on the order of input array a.
    'C' row-major (C-style),
    'F' column-major (Fortran-style) memory representation.
    'A' (any) means 'F' if `a` is Fortran contiguous, 'C' otherwise
    'K' (keep) preserve input order
    Defaults to 'C'.

Returns
-------
out : ndarray
    Array interpretation of `a`.  No copy is performed if the input
    is already an ndarray.  If `a` is a subclass of ndarray, a base
    class ndarray is returned.

Raises
------
ValueError
    Raises ValueError if `a` contains NaN (Not a Number) or Inf (Infinity).

See Also
--------
asarray : Create and array.
asanyarray : Similar function which passes through subclasses.
ascontiguousarray : Convert input to a contiguous array.
asfarray : Convert input to a floating point ndarray.
asfortranarray : Convert input to an ndarray with column-major
                 memory order.
fromiter : Create an array from an iterator.
fromfunction : Construct an array by executing a function on grid
               positions.

Examples
--------
Convert a list into an array.  If all elements are finite
``asarray_chkfinite`` is identical to ``asarray``.

>>> a = [1, 2]
>>> np.asarray_chkfinite(a, dtype=float)
array([1., 2.])

Raises ValueError if array_like contains Nans or Infs.

>>> a = [1, 2, np.inf]
>>> try:
...     np.asarray_chkfinite(a)
... except ValueError:
...     print('ValueError')
...
ValueError

### asmatrix(data, dtype=None)
Module: `numpy`

Interpret the input as a matrix.

Unlike `matrix`, `asmatrix` does not make a copy if the input is already
a matrix or an ndarray.  Equivalent to ``matrix(data, copy=False)``.

Parameters
----------
data : array_like
    Input data.
dtype : data-type
   Data-type of the output matrix.

Returns
-------
mat : matrix
    `data` interpreted as a matrix.

Examples
--------
>>> x = np.array([[1, 2], [3, 4]])

>>> m = np.asmatrix(x)

>>> x[0,0] = 5

>>> m
matrix([[5, 2],
        [3, 4]])

### bartlett(M)
Module: `numpy`

Return the Bartlett window.

The Bartlett window is very similar to a triangular window, except
that the end points are at zero.  It is often used in signal
processing for tapering a signal, without generating too much
ripple in the frequency domain.

Parameters
----------
M : int
    Number of points in the output window. If zero or less, an
    empty array is returned.

Returns
-------
out : array
    The triangular window, with the maximum value normalized to one
    (the value one appears only if the number of samples is odd), with
    the first and last samples equal to zero.

See Also
--------
blackman, hamming, hanning, kaiser

Notes
-----
The Bartlett window is defined as

.. math:: w(n) = \frac{2}{M-1} \left(
          \frac{M-1}{2} - \left|n - \frac{M-1}{2}\right|
          \right)

Most references to the Bartlett window come from the signal processing
literature, where it is used as one of many windowing functions for
smoothing values.  Note that convolution with this window produces linear
interpolation.  It is also known as an apodization (which means "removing
the foot", i.e. smoothing discontinuities at the beginning and end of the
sampled signal) or tapering function. The Fourier transform of the
Bartlett window is the product of two sinc functions. Note the excellent
discussion in Kanasewich [2]_.

References
----------
.. [1] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
       Biometrika 37, 1-16, 1950.
.. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
       The University of Alberta Press, 1975, pp. 109-110.
.. [3] A.V. Oppenheim and R.W. Schafer, "Discrete-Time Signal
       Processing", Prentice-Hall, 1999, pp. 468-471.
.. [4] Wikipedia, "Window function",
       https://en.wikipedia.org/wiki/Window_function
.. [5] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
       "Numerical Recipes", Cambridge University Press, 1986, page 429.

Examples
--------
>>> import matplotlib.pyplot as plt
>>> np.bartlett(12)
array([ 0.        ,  0.18181818,  0.36363636,  0.54545455,  0.72727273, # may vary
        0.90909091,  0.90909091,  0.72727273,  0.54545455,  0.36363636,
        0.18181818,  0.        ])

Plot the window and its frequency response (requires SciPy and matplotlib):

>>> from numpy.fft import fft, fftshift
>>> window = np.bartlett(51)
>>> plt.plot(window)
[<matplotlib.lines.Line2D object at 0x...>]
>>> plt.title("Bartlett window")
Text(0.5, 1.0, 'Bartlett window')
>>> plt.ylabel("Amplitude")
Text(0, 0.5, 'Amplitude')
>>> plt.xlabel("Sample")
Text(0.5, 0, 'Sample')
>>> plt.show()

>>> plt.figure()
<Figure size 640x480 with 0 Axes>
>>> A = fft(window, 2048) / 25.5
>>> mag = np.abs(fftshift(A))
>>> freq = np.linspace(-0.5, 0.5, len(A))
>>> with np.errstate(divide='ignore', invalid='ignore'):
...     response = 20 * np.log10(mag)
...
>>> response = np.clip(response, -100, 100)
>>> plt.plot(freq, response)
[<matplotlib.lines.Line2D object at 0x...>]
>>> plt.title("Frequency response of Bartlett window")
Text(0.5, 1.0, 'Frequency response of Bartlett window')
>>> plt.ylabel("Magnitude [dB]")
Text(0, 0.5, 'Magnitude [dB]')
>>> plt.xlabel("Normalized frequency [cycles per sample]")
Text(0.5, 0, 'Normalized frequency [cycles per sample]')
>>> _ = plt.axis('tight')
>>> plt.show()

### base_repr(number, base=2, padding=0)
Module: `numpy`

Return a string representation of a number in the given base system.

Parameters
----------
number : int
    The value to convert. Positive and negative values are handled.
base : int, optional
    Convert `number` to the `base` number system. The valid range is 2-36,
    the default value is 2.
padding : int, optional
    Number of zeros padded on the left. Default is 0 (no padding).

Returns
-------
out : str
    String representation of `number` in `base` system.

See Also
--------
binary_repr : Faster version of `base_repr` for base 2.

Examples
--------
>>> np.base_repr(5)
'101'
>>> np.base_repr(6, 5)
'11'
>>> np.base_repr(7, base=5, padding=3)
'00012'

>>> np.base_repr(10, base=16)
'A'
>>> np.base_repr(32, base=16)
'20'

### binary_repr(num, width=None)
Module: `numpy`

Return the binary representation of the input number as a string.

For negative numbers, if width is not given, a minus sign is added to the
front. If width is given, the two's complement of the number is
returned, with respect to that width.

In a two's-complement system negative numbers are represented by the two's
complement of the absolute value. This is the most common method of
representing signed integers on computers [1]_. A N-bit two's-complement
system can represent every integer in the range
:math:`-2^{N-1}` to :math:`+2^{N-1}-1`.

Parameters
----------
num : int
    Only an integer decimal number can be used.
width : int, optional
    The length of the returned string if `num` is positive, or the length
    of the two's complement if `num` is negative, provided that `width` is
    at least a sufficient number of bits for `num` to be represented in the
    designated form.

    If the `width` value is insufficient, it will be ignored, and `num` will
    be returned in binary (`num` > 0) or two's complement (`num` < 0) form
    with its width equal to the minimum number of bits needed to represent
    the number in the designated form. This behavior is deprecated and will
    later raise an error.

    .. deprecated:: 1.12.0

Returns
-------
bin : str
    Binary representation of `num` or two's complement of `num`.

See Also
--------
base_repr: Return a string representation of a number in the given base
           system.
bin: Python's built-in binary representation generator of an integer.

Notes
-----
`binary_repr` is equivalent to using `base_repr` with base 2, but about 25x
faster.

References
----------
.. [1] Wikipedia, "Two's complement",
    https://en.wikipedia.org/wiki/Two's_complement

Examples
--------
>>> np.binary_repr(3)
'11'
>>> np.binary_repr(-3)
'-11'
>>> np.binary_repr(3, width=4)
'0011'

The two's complement is returned when the input number is negative and
width is specified:

>>> np.binary_repr(-3, width=3)
'101'
>>> np.binary_repr(-3, width=5)
'11101'

### blackman(M)
Module: `numpy`

Return the Blackman window.

The Blackman window is a taper formed by using the first three
terms of a summation of cosines. It was designed to have close to the
minimal leakage possible.  It is close to optimal, only slightly worse
than a Kaiser window.

Parameters
----------
M : int
    Number of points in the output window. If zero or less, an empty
    array is returned.

Returns
-------
out : ndarray
    The window, with the maximum value normalized to one (the value one
    appears only if the number of samples is odd).

See Also
--------
bartlett, hamming, hanning, kaiser

Notes
-----
The Blackman window is defined as

.. math::  w(n) = 0.42 - 0.5 \cos(2\pi n/M) + 0.08 \cos(4\pi n/M)

Most references to the Blackman window come from the signal processing
literature, where it is used as one of many windowing functions for
smoothing values.  It is also known as an apodization (which means
"removing the foot", i.e. smoothing discontinuities at the beginning
and end of the sampled signal) or tapering function. It is known as a
"near optimal" tapering function, almost as good (by some measures)
as the kaiser window.

References
----------
Blackman, R.B. and Tukey, J.W., (1958) The measurement of power spectra,
Dover Publications, New York.

Oppenheim, A.V., and R.W. Schafer. Discrete-Time Signal Processing.
Upper Saddle River, NJ: Prentice-Hall, 1999, pp. 468-471.

Examples
--------
>>> import matplotlib.pyplot as plt
>>> np.blackman(12)
array([-1.38777878e-17,   3.26064346e-02,   1.59903635e-01, # may vary
        4.14397981e-01,   7.36045180e-01,   9.67046769e-01,
        9.67046769e-01,   7.36045180e-01,   4.14397981e-01,
        1.59903635e-01,   3.26064346e-02,  -1.38777878e-17])

Plot the window and the frequency response:

>>> from numpy.fft import fft, fftshift
>>> window = np.blackman(51)
>>> plt.plot(window)
[<matplotlib.lines.Line2D object at 0x...>]
>>> plt.title("Blackman window")
Text(0.5, 1.0, 'Blackman window')
>>> plt.ylabel("Amplitude")
Text(0, 0.5, 'Amplitude')
>>> plt.xlabel("Sample")
Text(0.5, 0, 'Sample')
>>> plt.show()

>>> plt.figure()
<Figure size 640x480 with 0 Axes>
>>> A = fft(window, 2048) / 25.5
>>> mag = np.abs(fftshift(A))
>>> freq = np.linspace(-0.5, 0.5, len(A))
>>> with np.errstate(divide='ignore', invalid='ignore'):
...     response = 20 * np.log10(mag)
...
>>> response = np.clip(response, -100, 100)
>>> plt.plot(freq, response)
[<matplotlib.lines.Line2D object at 0x...>]
>>> plt.title("Frequency response of Blackman window")
Text(0.5, 1.0, 'Frequency response of Blackman window')
>>> plt.ylabel("Magnitude [dB]")
Text(0, 0.5, 'Magnitude [dB]')
>>> plt.xlabel("Normalized frequency [cycles per sample]")
Text(0.5, 0, 'Normalized frequency [cycles per sample]')
>>> _ = plt.axis('tight')
>>> plt.show()

### bmat(obj, ldict=None, gdict=None)
Module: `numpy`

Build a matrix object from a string, nested sequence, or array.

Parameters
----------
obj : str or array_like
    Input data. If a string, variables in the current scope may be
    referenced by name.
ldict : dict, optional
    A dictionary that replaces local operands in current frame.
    Ignored if `obj` is not a string or `gdict` is None.
gdict : dict, optional
    A dictionary that replaces global operands in current frame.
    Ignored if `obj` is not a string.

Returns
-------
out : matrix
    Returns a matrix object, which is a specialized 2-D array.

See Also
--------
block :
    A generalization of this function for N-d arrays, that returns normal
    ndarrays.

Examples
--------
>>> A = np.mat('1 1; 1 1')
>>> B = np.mat('2 2; 2 2')
>>> C = np.mat('3 4; 5 6')
>>> D = np.mat('7 8; 9 0')

All the following expressions construct the same block matrix:

>>> np.bmat([[A, B], [C, D]])
matrix([[1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 4, 7, 8],
        [5, 6, 9, 0]])
>>> np.bmat(np.r_[np.c_[A, B], np.c_[C, D]])
matrix([[1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 4, 7, 8],
        [5, 6, 9, 0]])
>>> np.bmat('A,B; C,D')
matrix([[1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 4, 7, 8],
        [5, 6, 9, 0]])

### broadcast_shapes(*args)
Module: `numpy`

Broadcast the input shapes into a single shape.

:ref:`Learn more about broadcasting here <basics.broadcasting>`.

.. versionadded:: 1.20.0

Parameters
----------
`*args` : tuples of ints, or ints
    The shapes to be broadcast against each other.

Returns
-------
tuple
    Broadcasted shape.

Raises
------
ValueError
    If the shapes are not compatible and cannot be broadcast according
    to NumPy's broadcasting rules.

See Also
--------
broadcast
broadcast_arrays
broadcast_to

Examples
--------
>>> np.broadcast_shapes((1, 2), (3, 1), (3, 2))
(3, 2)

>>> np.broadcast_shapes((6, 7), (5, 6, 1), (7,), (5, 1, 7))
(5, 6, 7)

### byte_bounds(a)
Module: `numpy.lib.utils`

Returns pointers to the end-points of an array.

Parameters
----------
a : ndarray
    Input array. It must conform to the Python-side of the array
    interface.

Returns
-------
(low, high) : tuple of 2 integers
    The first integer is the first byte of the array, the second
    integer is just past the last byte of the array.  If `a` is not
    contiguous it will not use every byte between the (`low`, `high`)
    values.

Examples
--------
>>> I = np.eye(2, dtype='f'); I.dtype
dtype('float32')
>>> low, high = np.byte_bounds(I)
>>> high - low == I.size*I.itemsize
True
>>> I = np.eye(2); I.dtype
dtype('float64')
>>> low, high = np.byte_bounds(I)
>>> high - low == I.size*I.itemsize
True

### deprecate(*args, **kwargs)
Module: `numpy.lib.utils`

Issues a DeprecationWarning, adds warning to `old_name`'s
docstring, rebinds ``old_name.__name__`` and returns the new
function object.

This function may also be used as a decorator.

Parameters
----------
func : function
    The function to be deprecated.
old_name : str, optional
    The name of the function to be deprecated. Default is None, in
    which case the name of `func` is used.
new_name : str, optional
    The new name for the function. Default is None, in which case the
    deprecation message is that `old_name` is deprecated. If given, the
    deprecation message is that `old_name` is deprecated and `new_name`
    should be used instead.
message : str, optional
    Additional explanation of the deprecation.  Displayed in the
    docstring after the warning.

Returns
-------
old_func : function
    The deprecated function.

Examples
--------
Note that ``olduint`` returns a value after printing Deprecation
Warning:

>>> olduint = np.deprecate(np.uint)
DeprecationWarning: `uint64` is deprecated! # may vary
>>> olduint(6)
6

### deprecate_with_doc(msg)
Module: `numpy.lib.utils`

Deprecates a function and includes the deprecation in its docstring.

This function is used as a decorator. It returns an object that can be
used to issue a DeprecationWarning, by passing the to-be decorated
function as argument, this adds warning to the to-be decorated function's
docstring and returns the new function object.

See Also
--------
deprecate : Decorate a function such that it issues a `DeprecationWarning`

Parameters
----------
msg : str
    Additional explanation of the deprecation. Displayed in the
    docstring after the warning.

Returns
-------
obj : object

### diag_indices(n, ndim=2)
Module: `numpy`

Return the indices to access the main diagonal of an array.

This returns a tuple of indices that can be used to access the main
diagonal of an array `a` with ``a.ndim >= 2`` dimensions and shape
(n, n, ..., n). For ``a.ndim = 2`` this is the usual diagonal, for
``a.ndim > 2`` this is the set of indices to access ``a[i, i, ..., i]``
for ``i = [0..n-1]``.

Parameters
----------
n : int
  The size, along each dimension, of the arrays for which the returned
  indices can be used.

ndim : int, optional
  The number of dimensions.

See Also
--------
diag_indices_from

Notes
-----
.. versionadded:: 1.4.0

Examples
--------
Create a set of indices to access the diagonal of a (4, 4) array:

>>> di = np.diag_indices(4)
>>> di
(array([0, 1, 2, 3]), array([0, 1, 2, 3]))
>>> a = np.arange(16).reshape(4, 4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])
>>> a[di] = 100
>>> a
array([[100,   1,   2,   3],
       [  4, 100,   6,   7],
       [  8,   9, 100,  11],
       [ 12,  13,  14, 100]])

Now, we create indices to manipulate a 3-D array:

>>> d3 = np.diag_indices(2, 3)
>>> d3
(array([0, 1]), array([0, 1]), array([0, 1]))

And use it to set the diagonal of an array of zeros to 1:

>>> a = np.zeros((2, 2, 2), dtype=int)
>>> a[d3] = 1
>>> a
array([[[1, 0],
        [0, 0]],
       [[0, 0],
        [0, 1]]])

### disp(mesg, device=None, linefeed=True)
Module: `numpy.lib.function_base`

Display a message on a device.

Parameters
----------
mesg : str
    Message to display.
device : object
    Device to write message. If None, defaults to ``sys.stdout`` which is
    very similar to ``print``. `device` needs to have ``write()`` and
    ``flush()`` methods.
linefeed : bool, optional
    Option whether to print a line feed or not. Defaults to True.

Raises
------
AttributeError
    If `device` does not have a ``write()`` or ``flush()`` method.

Examples
--------
Besides ``sys.stdout``, a file-like object can also be used as it has
both required methods:

>>> from io import StringIO
>>> buf = StringIO()
>>> np.disp(u'"Display" in a file', device=buf)
>>> buf.getvalue()
'"Display" in a file\n'

### eye(N, M=None, k=0, dtype=<class 'float'>, order='C', *, like=None)
Module: `numpy`

Return a 2-D array with ones on the diagonal and zeros elsewhere.

Parameters
----------
N : int
  Number of rows in the output.
M : int, optional
  Number of columns in the output. If None, defaults to `N`.
k : int, optional
  Index of the diagonal: 0 (the default) refers to the main diagonal,
  a positive value refers to an upper diagonal, and a negative value
  to a lower diagonal.
dtype : data-type, optional
  Data-type of the returned array.
order : {'C', 'F'}, optional
    Whether the output should be stored in row-major (C-style) or
    column-major (Fortran-style) order in memory.

    .. versionadded:: 1.14.0
like : array_like, optional
    Reference object to allow the creation of arrays which are not
    NumPy arrays. If an array-like passed in as ``like`` supports
    the ``__array_function__`` protocol, the result will be defined
    by it. In this case, it ensures the creation of an array object
    compatible with that passed in via this argument.

    .. versionadded:: 1.20.0

Returns
-------
I : ndarray of shape (N,M)
  An array where all elements are equal to zero, except for the `k`-th
  diagonal, whose values are equal to one.

See Also
--------
identity : (almost) equivalent function
diag : diagonal 2-D array from a 1-D array specified by the user.

Examples
--------
>>> np.eye(2, dtype=int)
array([[1, 0],
       [0, 1]])
>>> np.eye(3, k=1)
array([[0.,  1.,  0.],
       [0.,  0.,  1.],
       [0.,  0.,  0.]])

### find_common_type(array_types, scalar_types)
Module: `numpy`

Determine common type following standard coercion rules.

.. deprecated:: NumPy 1.25

    This function is deprecated, use `numpy.promote_types` or
    `numpy.result_type` instead.  To achieve semantics for the
    `scalar_types` argument, use `numpy.result_type` and pass the Python
    values `0`, `0.0`, or `0j`.
    This will give the same results in almost all cases.
    More information and rare exception can be found in the
    `NumPy 1.25 release notes <https://numpy.org/devdocs/release/1.25.0-notes.html>`_.

Parameters
----------
array_types : sequence
    A list of dtypes or dtype convertible objects representing arrays.
scalar_types : sequence
    A list of dtypes or dtype convertible objects representing scalars.

Returns
-------
datatype : dtype
    The common data type, which is the maximum of `array_types` ignoring
    `scalar_types`, unless the maximum of `scalar_types` is of a
    different kind (`dtype.kind`). If the kind is not understood, then
    None is returned.

See Also
--------
dtype, common_type, can_cast, mintypecode

Examples
--------
>>> np.find_common_type([], [np.int64, np.float32, complex])
dtype('complex128')
>>> np.find_common_type([np.int64, np.float32], [])
dtype('float64')

The standard casting rules ensure that a scalar cannot up-cast an
array unless the scalar is of a fundamentally different kind of data
(i.e. under a different hierarchy in the data type hierarchy) then
the array:

>>> np.find_common_type([np.float32], [np.int64, np.float64])
dtype('float32')

Complex is of a different type, so it up-casts the float in the
`array_types` argument:

>>> np.find_common_type([np.float32], [complex])
dtype('complex128')

Type specifier strings are convertible to dtypes and can therefore
be used instead of dtypes:

>>> np.find_common_type(['f4', 'f4', 'i4'], ['c8'])
dtype('complex128')

### format_float_positional(x, precision=None, unique=True, fractional=True, trim='k', sign=False, pad_left=None, pad_right=None, min_digits=None)
Module: `numpy`

Format a floating-point scalar as a decimal string in positional notation.

Provides control over rounding, trimming and padding. Uses and assumes
IEEE unbiased rounding. Uses the "Dragon4" algorithm.

Parameters
----------
x : python float or numpy floating scalar
    Value to format.
precision : non-negative integer or None, optional
    Maximum number of digits to print. May be None if `unique` is
    `True`, but must be an integer if unique is `False`.
unique : boolean, optional
    If `True`, use a digit-generation strategy which gives the shortest
    representation which uniquely identifies the floating-point number from
    other values of the same type, by judicious rounding. If `precision`
    is given fewer digits than necessary can be printed, or if `min_digits`
    is given more can be printed, in which cases the last digit is rounded
    with unbiased rounding.
    If `False`, digits are generated as if printing an infinite-precision
    value and stopping after `precision` digits, rounding the remaining
    value with unbiased rounding
fractional : boolean, optional
    If `True`, the cutoffs of `precision` and `min_digits` refer to the
    total number of digits after the decimal point, including leading
    zeros.
    If `False`, `precision` and `min_digits` refer to the total number of
    significant digits, before or after the decimal point, ignoring leading
    zeros.
trim : one of 'k', '.', '0', '-', optional
    Controls post-processing trimming of trailing digits, as follows:

    * 'k' : keep trailing zeros, keep decimal point (no trimming)
    * '.' : trim all trailing zeros, leave decimal point
    * '0' : trim all but the zero before the decimal point. Insert the
      zero if it is missing.
    * '-' : trim trailing zeros and any trailing decimal point
sign : boolean, optional
    Whether to show the sign for positive values.
pad_left : non-negative integer, optional
    Pad the left side of the string with whitespace until at least that
    many characters are to the left of the decimal point.
pad_right : non-negative integer, optional
    Pad the right side of the string with whitespace until at least that
    many characters are to the right of the decimal point.
min_digits : non-negative integer or None, optional
    Minimum number of digits to print. Only has an effect if `unique=True`
    in which case additional digits past those necessary to uniquely
    identify the value may be printed, rounding the last additional digit.

    -- versionadded:: 1.21.0

Returns
-------
rep : string
    The string representation of the floating point value

See Also
--------
format_float_scientific

Examples
--------
>>> np.format_float_positional(np.float32(np.pi))
'3.1415927'
>>> np.format_float_positional(np.float16(np.pi))
'3.14'
>>> np.format_float_positional(np.float16(0.3))
'0.3'
>>> np.format_float_positional(np.float16(0.3), unique=False, precision=10)
'0.3000488281'

### format_float_scientific(x, precision=None, unique=True, trim='k', sign=False, pad_left=None, exp_digits=None, min_digits=None)
Module: `numpy`

Format a floating-point scalar as a decimal string in scientific notation.

Provides control over rounding, trimming and padding. Uses and assumes
IEEE unbiased rounding. Uses the "Dragon4" algorithm.

Parameters
----------
x : python float or numpy floating scalar
    Value to format.
precision : non-negative integer or None, optional
    Maximum number of digits to print. May be None if `unique` is
    `True`, but must be an integer if unique is `False`.
unique : boolean, optional
    If `True`, use a digit-generation strategy which gives the shortest
    representation which uniquely identifies the floating-point number from
    other values of the same type, by judicious rounding. If `precision`
    is given fewer digits than necessary can be printed. If `min_digits`
    is given more can be printed, in which cases the last digit is rounded
    with unbiased rounding.
    If `False`, digits are generated as if printing an infinite-precision
    value and stopping after `precision` digits, rounding the remaining
    value with unbiased rounding
trim : one of 'k', '.', '0', '-', optional
    Controls post-processing trimming of trailing digits, as follows:

    * 'k' : keep trailing zeros, keep decimal point (no trimming)
    * '.' : trim all trailing zeros, leave decimal point
    * '0' : trim all but the zero before the decimal point. Insert the
      zero if it is missing.
    * '-' : trim trailing zeros and any trailing decimal point
sign : boolean, optional
    Whether to show the sign for positive values.
pad_left : non-negative integer, optional
    Pad the left side of the string with whitespace until at least that
    many characters are to the left of the decimal point.
exp_digits : non-negative integer, optional
    Pad the exponent with zeros until it contains at least this many digits.
    If omitted, the exponent will be at least 2 digits.
min_digits : non-negative integer or None, optional
    Minimum number of digits to print. This only has an effect for
    `unique=True`. In that case more digits than necessary to uniquely
    identify the value may be printed and rounded unbiased.

    -- versionadded:: 1.21.0

Returns
-------
rep : string
    The string representation of the floating point value

See Also
--------
format_float_positional

Examples
--------
>>> np.format_float_scientific(np.float32(np.pi))
'3.1415927e+00'
>>> s = np.float32(1.23e24)
>>> np.format_float_scientific(s, unique=False, precision=15)
'1.230000071797338e+24'
>>> np.format_float_scientific(s, exp_digits=4)
'1.23e+0024'

### fromfunction(function, shape, *, dtype=<class 'float'>, like=None, **kwargs)
Module: `numpy`

Construct an array by executing a function over each coordinate.

The resulting array therefore has a value ``fn(x, y, z)`` at
coordinate ``(x, y, z)``.

Parameters
----------
function : callable
    The function is called with N parameters, where N is the rank of
    `shape`.  Each parameter represents the coordinates of the array
    varying along a specific axis.  For example, if `shape`
    were ``(2, 2)``, then the parameters would be
    ``array([[0, 0], [1, 1]])`` and ``array([[0, 1], [0, 1]])``
shape : (N,) tuple of ints
    Shape of the output array, which also determines the shape of
    the coordinate arrays passed to `function`.
dtype : data-type, optional
    Data-type of the coordinate arrays passed to `function`.
    By default, `dtype` is float.
like : array_like, optional
    Reference object to allow the creation of arrays which are not
    NumPy arrays. If an array-like passed in as ``like`` supports
    the ``__array_function__`` protocol, the result will be defined
    by it. In this case, it ensures the creation of an array object
    compatible with that passed in via this argument.

    .. versionadded:: 1.20.0

Returns
-------
fromfunction : any
    The result of the call to `function` is passed back directly.
    Therefore the shape of `fromfunction` is completely determined by
    `function`.  If `function` returns a scalar value, the shape of
    `fromfunction` would not match the `shape` parameter.

See Also
--------
indices, meshgrid

Notes
-----
Keywords other than `dtype` and `like` are passed to `function`.

Examples
--------
>>> np.fromfunction(lambda i, j: i, (2, 2), dtype=float)
array([[0., 0.],
       [1., 1.]])

>>> np.fromfunction(lambda i, j: j, (2, 2), dtype=float)
array([[0., 1.],
       [0., 1.]])

>>> np.fromfunction(lambda i, j: i == j, (3, 3), dtype=int)
array([[ True, False, False],
       [False,  True, False],
       [False, False,  True]])

>>> np.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)
array([[0, 1, 2],
       [1, 2, 3],
       [2, 3, 4]])

### fromregex(file, regexp, dtype, encoding=None)
Module: `numpy`

Construct an array from a text file, using regular expression parsing.

The returned array is always a structured array, and is constructed from
all matches of the regular expression in the file. Groups in the regular
expression are converted to fields of the structured array.

Parameters
----------
file : path or file
    Filename or file object to read.

    .. versionchanged:: 1.22.0
        Now accepts `os.PathLike` implementations.
regexp : str or regexp
    Regular expression used to parse the file.
    Groups in the regular expression correspond to fields in the dtype.
dtype : dtype or list of dtypes
    Dtype for the structured array; must be a structured datatype.
encoding : str, optional
    Encoding used to decode the inputfile. Does not apply to input streams.

    .. versionadded:: 1.14.0

Returns
-------
output : ndarray
    The output array, containing the part of the content of `file` that
    was matched by `regexp`. `output` is always a structured array.

Raises
------
TypeError
    When `dtype` is not a valid dtype for a structured array.

See Also
--------
fromstring, loadtxt

Notes
-----
Dtypes for structured arrays can be specified in several forms, but all
forms specify at least the data type and field name. For details see
`basics.rec`.

Examples
--------
>>> from io import StringIO
>>> text = StringIO("1312 foo\n1534  bar\n444   qux")

>>> regexp = r"(\d+)\s+(...)"  # match [digits, whitespace, anything]
>>> output = np.fromregex(text, regexp,
...                       [('num', np.int64), ('key', 'S3')])
>>> output
array([(1312, b'foo'), (1534, b'bar'), ( 444, b'qux')],
      dtype=[('num', '<i8'), ('key', 'S3')])
>>> output['num']
array([1312, 1534,  444])

### full(shape, fill_value, dtype=None, order='C', *, like=None)
Module: `numpy`

Return a new array of given shape and type, filled with `fill_value`.

Parameters
----------
shape : int or sequence of ints
    Shape of the new array, e.g., ``(2, 3)`` or ``2``.
fill_value : scalar or array_like
    Fill value.
dtype : data-type, optional
    The desired data-type for the array  The default, None, means
     ``np.array(fill_value).dtype``.
order : {'C', 'F'}, optional
    Whether to store multidimensional data in C- or Fortran-contiguous
    (row- or column-wise) order in memory.
like : array_like, optional
    Reference object to allow the creation of arrays which are not
    NumPy arrays. If an array-like passed in as ``like`` supports
    the ``__array_function__`` protocol, the result will be defined
    by it. In this case, it ensures the creation of an array object
    compatible with that passed in via this argument.

    .. versionadded:: 1.20.0

Returns
-------
out : ndarray
    Array of `fill_value` with the given shape, dtype, and order.

See Also
--------
full_like : Return a new array with shape of input filled with value.
empty : Return a new uninitialized array.
ones : Return a new array setting values to one.
zeros : Return a new array setting values to zero.

Examples
--------
>>> np.full((2, 2), np.inf)
array([[inf, inf],
       [inf, inf]])
>>> np.full((2, 2), 10)
array([[10, 10],
       [10, 10]])

>>> np.full((2, 2), [1, 2])
array([[1, 2],
       [1, 2]])

### genfromtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, skip_header=0, skip_footer=0, converters=None, missing_values=None, filling_values=None, usecols=None, names=None, excludelist=None, deletechars=" !#$%&'()*+,-./:;<=>?@[\\]^{|}~", replace_space='_', autostrip=False, case_sensitive=True, defaultfmt='f%i', unpack=None, usemask=False, loose=True, invalid_raise=True, max_rows=None, encoding='bytes', *, ndmin=0, like=None)
Module: `numpy`

Load data from a text file, with missing values handled as specified.

Each line past the first `skip_header` lines is split at the `delimiter`
character, and characters following the `comments` character are discarded.

Parameters
----------
fname : file, str, pathlib.Path, list of str, generator
    File, filename, list, or generator to read.  If the filename
    extension is ``.gz`` or ``.bz2``, the file is first decompressed. Note
    that generators must return bytes or strings. The strings
    in a list or produced by a generator are treated as lines.
dtype : dtype, optional
    Data type of the resulting array.
    If None, the dtypes will be determined by the contents of each
    column, individually.
comments : str, optional
    The character used to indicate the start of a comment.
    All the characters occurring on a line after a comment are discarded.
delimiter : str, int, or sequence, optional
    The string used to separate values.  By default, any consecutive
    whitespaces act as delimiter.  An integer or sequence of integers
    can also be provided as width(s) of each field.
skiprows : int, optional
    `skiprows` was removed in numpy 1.10. Please use `skip_header` instead.
skip_header : int, optional
    The number of lines to skip at the beginning of the file.
skip_footer : int, optional
    The number of lines to skip at the end of the file.
converters : variable, optional
    The set of functions that convert the data of a column to a value.
    The converters can also be used to provide a default value
    for missing data: ``converters = {3: lambda s: float(s or 0)}``.
missing : variable, optional
    `missing` was removed in numpy 1.10. Please use `missing_values`
    instead.
missing_values : variable, optional
    The set of strings corresponding to missing data.
filling_values : variable, optional
    The set of values to be used as default when the data are missing.
usecols : sequence, optional
    Which columns to read, with 0 being the first.  For example,
    ``usecols = (1, 4, 5)`` will extract the 2nd, 5th and 6th columns.
names : {None, True, str, sequence}, optional
    If `names` is True, the field names are read from the first line after
    the first `skip_header` lines. This line can optionally be preceded
    by a comment delimiter. If `names` is a sequence or a single-string of
    comma-separated names, the names will be used to define the field names
    in a structured dtype. If `names` is None, the names of the dtype
    fields will be used, if any.
excludelist : sequence, optional
    A list of names to exclude. This list is appended to the default list
    ['return','file','print']. Excluded names are appended with an
    underscore: for example, `file` would become `file_`.
deletechars : str, optional
    A string combining invalid characters that must be deleted from the
    names.
defaultfmt : str, optional
    A format used to define default field names, such as "f%i" or "f_%02i".
autostrip : bool, optional
    Whether to automatically strip white spaces from the variables.
replace_space : char, optional
    Character(s) used in replacement of white spaces in the variable
    names. By default, use a '_'.
case_sensitive : {True, False, 'upper', 'lower'}, optional
    If True, field names are case sensitive.
    If False or 'upper', field names are converted to upper case.
    If 'lower', field names are converted to lower case.
unpack : bool, optional
    If True, the returned array is transposed, so that arguments may be
    unpacked using ``x, y, z = genfromtxt(...)``.  When used with a
    structured data-type, arrays are returned for each field.
    Default is False.
usemask : bool, optional
    If True, return a masked array.
    If False, return a regular array.
loose : bool, optional
    If True, do not raise errors for invalid values.
invalid_raise : bool, optional
    If True, an exception is raised if an inconsistency is detected in the
    number of columns.
    If False, a warning is emitted and the offending lines are skipped.
max_rows : int,  optional
    The maximum number of rows to read. Must not be used with skip_footer
    at the same time.  If given, the value must be at least 1. Default is
    to read the entire file.

    .. versionadded:: 1.10.0
encoding : str, optional
    Encoding used to decode the inputfile. Does not apply when `fname` is
    a file object.  The special value 'bytes' enables backward compatibility
    workarounds that ensure that you receive byte arrays when possible
    and passes latin1 encoded strings to converters. Override this value to
    receive unicode arrays and pass strings as input to converters.  If set
    to None the system default is used. The default value is 'bytes'.

    .. versionadded:: 1.14.0
ndmin : int, optional
    Same parameter as `loadtxt`

    .. versionadded:: 1.23.0
like : array_like, optional
    Reference object to allow the creation of arrays which are not
    NumPy arrays. If an array-like passed in as ``like`` supports
    the ``__array_function__`` protocol, the result will be defined
    by it. In this case, it ensures the creation of an array object
    compatible with that passed in via this argument.

    .. versionadded:: 1.20.0

Returns
-------
out : ndarray
    Data read from the text file. If `usemask` is True, this is a
    masked array.

See Also
--------
numpy.loadtxt : equivalent function when no data is missing.

Notes
-----
* When spaces are used as delimiters, or when no delimiter has been given
  as input, there should not be any missing data between two fields.
* When the variables are named (either by a flexible dtype or with `names`),
  there must not be any header in the file (else a ValueError
  exception is raised).
* Individual values are not stripped of spaces by default.
  When using a custom converter, make sure the function does remove spaces.

References
----------
.. [1] NumPy User Guide, section `I/O with NumPy
       <https://docs.scipy.org/doc/numpy/user/basics.io.genfromtxt.html>`_.

Examples
--------
>>> from io import StringIO
>>> import numpy as np

Comma delimited file with mixed dtype

>>> s = StringIO(u"1,1.3,abcde")
>>> data = np.genfromtxt(s, dtype=[('myint','i8'),('myfloat','f8'),
... ('mystring','S5')], delimiter=",")
>>> data
array((1, 1.3, b'abcde'),
      dtype=[('myint', '<i8'), ('myfloat', '<f8'), ('mystring', 'S5')])

Using dtype = None

>>> _ = s.seek(0) # needed for StringIO example only
>>> data = np.genfromtxt(s, dtype=None,
... names = ['myint','myfloat','mystring'], delimiter=",")
>>> data
array((1, 1.3, b'abcde'),
      dtype=[('myint', '<i8'), ('myfloat', '<f8'), ('mystring', 'S5')])

Specifying dtype and names

>>> _ = s.seek(0)
>>> data = np.genfromtxt(s, dtype="i8,f8,S5",
... names=['myint','myfloat','mystring'], delimiter=",")
>>> data
array((1, 1.3, b'abcde'),
      dtype=[('myint', '<i8'), ('myfloat', '<f8'), ('mystring', 'S5')])

An example with fixed-width columns

>>> s = StringIO(u"11.3abcde")
>>> data = np.genfromtxt(s, dtype=None, names=['intvar','fltvar','strvar'],
...     delimiter=[1,3,5])
>>> data
array((1, 1.3, b'abcde'),
      dtype=[('intvar', '<i8'), ('fltvar', '<f8'), ('strvar', 'S5')])

An example to show comments

>>> f = StringIO('''
... text,# of chars
... hello world,11
... numpy,5''')
>>> np.genfromtxt(f, dtype='S12,S12', delimiter=',')
array([(b'text', b''), (b'hello world', b'11'), (b'numpy', b'5')],
  dtype=[('f0', 'S12'), ('f1', 'S12')])

### get_array_wrap(*args)
Module: `numpy.lib.shape_base`

Find the wrapper for the array with the highest priority.

In case of ties, leftmost wins. If no wrapper is found, return None

### get_include()
Module: `numpy.lib.utils`

Return the directory that contains the NumPy \*.h header files.

Extension modules that need to compile against NumPy should use this
function to locate the appropriate include directory.

Notes
-----
When using ``distutils``, for example in ``setup.py``::

    import numpy as np
    ...
    Extension('extension_name', ...
            include_dirs=[np.get_include()])
    ...

### get_printoptions()
Module: `numpy`

Return the current print options.

Returns
-------
print_opts : dict
    Dictionary of current print options with keys

      - precision : int
      - threshold : int
      - edgeitems : int
      - linewidth : int
      - suppress : bool
      - nanstr : str
      - infstr : str
      - formatter : dict of callables
      - sign : str

    For a full description of these options, see `set_printoptions`.

See Also
--------
set_printoptions, printoptions, set_string_function

### getbufsize()
Module: `numpy`

Return the size of the buffer used in ufuncs.

Returns
-------
getbufsize : int
    Size of ufunc buffer in bytes.

### geterr()
Module: `numpy`

Get the current way of handling floating-point errors.

Returns
-------
res : dict
    A dictionary with keys "divide", "over", "under", and "invalid",
    whose values are from the strings "ignore", "print", "log", "warn",
    "raise", and "call". The keys represent possible floating-point
    exceptions, and the values define how these exceptions are handled.

See Also
--------
geterrcall, seterr, seterrcall

Notes
-----
For complete documentation of the types of floating-point exceptions and
treatment options, see `seterr`.

Examples
--------
>>> np.geterr()
{'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
>>> np.arange(3.) / np.arange(3.)
array([nan,  1.,  1.])

>>> oldsettings = np.seterr(all='warn', over='raise')
>>> np.geterr()
{'divide': 'warn', 'over': 'raise', 'under': 'warn', 'invalid': 'warn'}
>>> np.arange(3.) / np.arange(3.)
array([nan,  1.,  1.])

### geterrcall()
Module: `numpy`

Return the current callback function used on floating-point errors.

When the error handling for a floating-point error (one of "divide",
"over", "under", or "invalid") is set to 'call' or 'log', the function
that is called or the log instance that is written to is returned by
`geterrcall`. This function or log instance has been set with
`seterrcall`.

Returns
-------
errobj : callable, log instance or None
    The current error handler. If no handler was set through `seterrcall`,
    ``None`` is returned.

See Also
--------
seterrcall, seterr, geterr

Notes
-----
For complete documentation of the types of floating-point exceptions and
treatment options, see `seterr`.

Examples
--------
>>> np.geterrcall()  # we did not yet set a handler, returns None

>>> oldsettings = np.seterr(all='call')
>>> def err_handler(type, flag):
...     print("Floating point error (%s), with flag %s" % (type, flag))
>>> oldhandler = np.seterrcall(err_handler)
>>> np.array([1, 2, 3]) / 0.0
Floating point error (divide by zero), with flag 1
array([inf, inf, inf])

>>> cur_handler = np.geterrcall()
>>> cur_handler is err_handler
True

### hamming(M)
Module: `numpy`

Return the Hamming window.

The Hamming window is a taper formed by using a weighted cosine.

Parameters
----------
M : int
    Number of points in the output window. If zero or less, an
    empty array is returned.

Returns
-------
out : ndarray
    The window, with the maximum value normalized to one (the value
    one appears only if the number of samples is odd).

See Also
--------
bartlett, blackman, hanning, kaiser

Notes
-----
The Hamming window is defined as

.. math::  w(n) = 0.54 - 0.46\cos\left(\frac{2\pi{n}}{M-1}\right)
           \qquad 0 \leq n \leq M-1

The Hamming was named for R. W. Hamming, an associate of J. W. Tukey
and is described in Blackman and Tukey. It was recommended for
smoothing the truncated autocovariance function in the time domain.
Most references to the Hamming window come from the signal processing
literature, where it is used as one of many windowing functions for
smoothing values.  It is also known as an apodization (which means
"removing the foot", i.e. smoothing discontinuities at the beginning
and end of the sampled signal) or tapering function.

References
----------
.. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
       spectra, Dover Publications, New York.
.. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
       University of Alberta Press, 1975, pp. 109-110.
.. [3] Wikipedia, "Window function",
       https://en.wikipedia.org/wiki/Window_function
.. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
       "Numerical Recipes", Cambridge University Press, 1986, page 425.

Examples
--------
>>> np.hamming(12)
array([ 0.08      ,  0.15302337,  0.34890909,  0.60546483,  0.84123594, # may vary
        0.98136677,  0.98136677,  0.84123594,  0.60546483,  0.34890909,
        0.15302337,  0.08      ])

Plot the window and the frequency response:

>>> import matplotlib.pyplot as plt
>>> from numpy.fft import fft, fftshift
>>> window = np.hamming(51)
>>> plt.plot(window)
[<matplotlib.lines.Line2D object at 0x...>]
>>> plt.title("Hamming window")
Text(0.5, 1.0, 'Hamming window')
>>> plt.ylabel("Amplitude")
Text(0, 0.5, 'Amplitude')
>>> plt.xlabel("Sample")
Text(0.5, 0, 'Sample')
>>> plt.show()

>>> plt.figure()
<Figure size 640x480 with 0 Axes>
>>> A = fft(window, 2048) / 25.5
>>> mag = np.abs(fftshift(A))
>>> freq = np.linspace(-0.5, 0.5, len(A))
>>> response = 20 * np.log10(mag)
>>> response = np.clip(response, -100, 100)
>>> plt.plot(freq, response)
[<matplotlib.lines.Line2D object at 0x...>]
>>> plt.title("Frequency response of Hamming window")
Text(0.5, 1.0, 'Frequency response of Hamming window')
>>> plt.ylabel("Magnitude [dB]")
Text(0, 0.5, 'Magnitude [dB]')
>>> plt.xlabel("Normalized frequency [cycles per sample]")
Text(0.5, 0, 'Normalized frequency [cycles per sample]')
>>> plt.axis('tight')
...
>>> plt.show()

### hanning(M)
Module: `numpy`

Return the Hanning window.

The Hanning window is a taper formed by using a weighted cosine.

Parameters
----------
M : int
    Number of points in the output window. If zero or less, an
    empty array is returned.

Returns
-------
out : ndarray, shape(M,)
    The window, with the maximum value normalized to one (the value
    one appears only if `M` is odd).

See Also
--------
bartlett, blackman, hamming, kaiser

Notes
-----
The Hanning window is defined as

.. math::  w(n) = 0.5 - 0.5\cos\left(\frac{2\pi{n}}{M-1}\right)
           \qquad 0 \leq n \leq M-1

The Hanning was named for Julius von Hann, an Austrian meteorologist.
It is also known as the Cosine Bell. Some authors prefer that it be
called a Hann window, to help avoid confusion with the very similar
Hamming window.

Most references to the Hanning window come from the signal processing
literature, where it is used as one of many windowing functions for
smoothing values.  It is also known as an apodization (which means
"removing the foot", i.e. smoothing discontinuities at the beginning
and end of the sampled signal) or tapering function.

References
----------
.. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
       spectra, Dover Publications, New York.
.. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
       The University of Alberta Press, 1975, pp. 106-108.
.. [3] Wikipedia, "Window function",
       https://en.wikipedia.org/wiki/Window_function
.. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
       "Numerical Recipes", Cambridge University Press, 1986, page 425.

Examples
--------
>>> np.hanning(12)
array([0.        , 0.07937323, 0.29229249, 0.57115742, 0.82743037,
       0.97974649, 0.97974649, 0.82743037, 0.57115742, 0.29229249,
       0.07937323, 0.        ])

Plot the window and its frequency response:

>>> import matplotlib.pyplot as plt
>>> from numpy.fft import fft, fftshift
>>> window = np.hanning(51)
>>> plt.plot(window)
[<matplotlib.lines.Line2D object at 0x...>]
>>> plt.title("Hann window")
Text(0.5, 1.0, 'Hann window')
>>> plt.ylabel("Amplitude")
Text(0, 0.5, 'Amplitude')
>>> plt.xlabel("Sample")
Text(0.5, 0, 'Sample')
>>> plt.show()

>>> plt.figure()
<Figure size 640x480 with 0 Axes>
>>> A = fft(window, 2048) / 25.5
>>> mag = np.abs(fftshift(A))
>>> freq = np.linspace(-0.5, 0.5, len(A))
>>> with np.errstate(divide='ignore', invalid='ignore'):
...     response = 20 * np.log10(mag)
...
>>> response = np.clip(response, -100, 100)
>>> plt.plot(freq, response)
[<matplotlib.lines.Line2D object at 0x...>]
>>> plt.title("Frequency response of the Hann window")
Text(0.5, 1.0, 'Frequency response of the Hann window')
>>> plt.ylabel("Magnitude [dB]")
Text(0, 0.5, 'Magnitude [dB]')
>>> plt.xlabel("Normalized frequency [cycles per sample]")
Text(0.5, 0, 'Normalized frequency [cycles per sample]')
>>> plt.axis('tight')
...
>>> plt.show()

### identity(n, dtype=None, *, like=None)
Module: `numpy`

Return the identity array.

The identity array is a square array with ones on
the main diagonal.

Parameters
----------
n : int
    Number of rows (and columns) in `n` x `n` output.
dtype : data-type, optional
    Data-type of the output.  Defaults to ``float``.
like : array_like, optional
    Reference object to allow the creation of arrays which are not
    NumPy arrays. If an array-like passed in as ``like`` supports
    the ``__array_function__`` protocol, the result will be defined
    by it. In this case, it ensures the creation of an array object
    compatible with that passed in via this argument.

    .. versionadded:: 1.20.0

Returns
-------
out : ndarray
    `n` x `n` array with its main diagonal set to one,
    and all other elements 0.

Examples
--------
>>> np.identity(3)
array([[1.,  0.,  0.],
       [0.,  1.,  0.],
       [0.,  0.,  1.]])

### indices(dimensions, dtype=<class 'int'>, sparse=False)
Module: `numpy`

Return an array representing the indices of a grid.

Compute an array where the subarrays contain index values 0, 1, ...
varying only along the corresponding axis.

Parameters
----------
dimensions : sequence of ints
    The shape of the grid.
dtype : dtype, optional
    Data type of the result.
sparse : boolean, optional
    Return a sparse representation of the grid instead of a dense
    representation. Default is False.

    .. versionadded:: 1.17

Returns
-------
grid : one ndarray or tuple of ndarrays
    If sparse is False:
        Returns one array of grid indices,
        ``grid.shape = (len(dimensions),) + tuple(dimensions)``.
    If sparse is True:
        Returns a tuple of arrays, with
        ``grid[i].shape = (1, ..., 1, dimensions[i], 1, ..., 1)`` with
        dimensions[i] in the ith place

See Also
--------
mgrid, ogrid, meshgrid

Notes
-----
The output shape in the dense case is obtained by prepending the number
of dimensions in front of the tuple of dimensions, i.e. if `dimensions`
is a tuple ``(r0, ..., rN-1)`` of length ``N``, the output shape is
``(N, r0, ..., rN-1)``.

The subarrays ``grid[k]`` contains the N-D array of indices along the
``k-th`` axis. Explicitly::

    grid[k, i0, i1, ..., iN-1] = ik

Examples
--------
>>> grid = np.indices((2, 3))
>>> grid.shape
(2, 2, 3)
>>> grid[0]        # row indices
array([[0, 0, 0],
       [1, 1, 1]])
>>> grid[1]        # column indices
array([[0, 1, 2],
       [0, 1, 2]])

The indices can be used as an index into an array.

>>> x = np.arange(20).reshape(5, 4)
>>> row, col = np.indices((2, 3))
>>> x[row, col]
array([[0, 1, 2],
       [4, 5, 6]])

Note that it would be more straightforward in the above example to
extract the required elements directly with ``x[:2, :3]``.

If sparse is set to true, the grid will be returned in a sparse
representation.

>>> i, j = np.indices((2, 3), sparse=True)
>>> i.shape
(2, 1)
>>> j.shape
(1, 3)
>>> i        # row indices
array([[0],
       [1]])
>>> j        # column indices
array([[0, 1, 2]])

### info(object=None, maxwidth=76, output=None, toplevel='numpy')
Module: `numpy`

Get help information for an array, function, class, or module.

Parameters
----------
object : object or str, optional
    Input object or name to get information about. If `object` is
    an `ndarray` instance, information about the array is printed.
    If `object` is a numpy object, its docstring is given. If it is
    a string, available modules are searched for matching objects.
    If None, information about `info` itself is returned.
maxwidth : int, optional
    Printing width.
output : file like object, optional
    File like object that the output is written to, default is
    ``None``, in which case ``sys.stdout`` will be used.
    The object has to be opened in 'w' or 'a' mode.
toplevel : str, optional
    Start search at this level.

See Also
--------
source, lookfor

Notes
-----
When used interactively with an object, ``np.info(obj)`` is equivalent
to ``help(obj)`` on the Python prompt or ``obj?`` on the IPython
prompt.

Examples
--------
>>> np.info(np.polyval) # doctest: +SKIP
   polyval(p, x)
     Evaluate the polynomial p at x.
     ...

When using a string for `object` it is possible to get multiple results.

>>> np.info('fft') # doctest: +SKIP
     *** Found in numpy ***
Core FFT routines
...
     *** Found in numpy.fft ***
 fft(a, n=None, axis=-1)
...
     *** Repeat reference found in numpy.fft.fftpack ***
     *** Total of 3 references found. ***

When the argument is an array, information about the array is printed.

>>> a = np.array([[1 + 2j, 3, -4], [-5j, 6, 0]], dtype=np.complex64)
>>> np.info(a)
class:  ndarray
shape:  (2, 3)
strides:  (24, 8)
itemsize:  8
aligned:  True
contiguous:  True
fortran:  False
data pointer: 0x562b6e0d2860  # may vary
byteorder:  little
byteswap:  False
type: complex64

### isfortran(a)
Module: `numpy`

Check if the array is Fortran contiguous but *not* C contiguous.

This function is obsolete and, because of changes due to relaxed stride
checking, its return value for the same array may differ for versions
of NumPy >= 1.10.0 and previous versions. If you only want to check if an
array is Fortran contiguous use ``a.flags.f_contiguous`` instead.

Parameters
----------
a : ndarray
    Input array.

Returns
-------
isfortran : bool
    Returns True if the array is Fortran contiguous but *not* C contiguous.


Examples
--------

np.array allows to specify whether the array is written in C-contiguous
order (last index varies the fastest), or FORTRAN-contiguous order in
memory (first index varies the fastest).

>>> a = np.array([[1, 2, 3], [4, 5, 6]], order='C')
>>> a
array([[1, 2, 3],
       [4, 5, 6]])
>>> np.isfortran(a)
False

>>> b = np.array([[1, 2, 3], [4, 5, 6]], order='F')
>>> b
array([[1, 2, 3],
       [4, 5, 6]])
>>> np.isfortran(b)
True


The transpose of a C-ordered array is a FORTRAN-ordered array.

>>> a = np.array([[1, 2, 3], [4, 5, 6]], order='C')
>>> a
array([[1, 2, 3],
       [4, 5, 6]])
>>> np.isfortran(a)
False
>>> b = a.T
>>> b
array([[1, 4],
       [2, 5],
       [3, 6]])
>>> np.isfortran(b)
True

C-ordered arrays evaluate as False even if they are also FORTRAN-ordered.

>>> np.isfortran(np.array([1, 2], order='F'))
False

### isscalar(element)
Module: `numpy`

Returns True if the type of `element` is a scalar type.

Parameters
----------
element : any
    Input argument, can be of any type and shape.

Returns
-------
val : bool
    True if `element` is a scalar type, False if it is not.

See Also
--------
ndim : Get the number of dimensions of an array

Notes
-----
If you need a stricter way to identify a *numerical* scalar, use
``isinstance(x, numbers.Number)``, as that returns ``False`` for most
non-numerical elements such as strings.

In most cases ``np.ndim(x) == 0`` should be used instead of this function,
as that will also return true for 0d arrays. This is how numpy overloads
functions in the style of the ``dx`` arguments to `gradient` and the ``bins``
argument to `histogram`. Some key differences:

+--------------------------------------+---------------+-------------------+
| x                                    |``isscalar(x)``|``np.ndim(x) == 0``|
+======================================+===============+===================+
| PEP 3141 numeric objects (including  | ``True``      | ``True``          |
| builtins)                            |               |                   |
+--------------------------------------+---------------+-------------------+
| builtin string and buffer objects    | ``True``      | ``True``          |
+--------------------------------------+---------------+-------------------+
| other builtin objects, like          | ``False``     | ``True``          |
| `pathlib.Path`, `Exception`,         |               |                   |
| the result of `re.compile`           |               |                   |
+--------------------------------------+---------------+-------------------+
| third-party objects like             | ``False``     | ``True``          |
| `matplotlib.figure.Figure`           |               |                   |
+--------------------------------------+---------------+-------------------+
| zero-dimensional numpy arrays        | ``False``     | ``True``          |
+--------------------------------------+---------------+-------------------+
| other numpy arrays                   | ``False``     | ``False``         |
+--------------------------------------+---------------+-------------------+
| `list`, `tuple`, and other sequence  | ``False``     | ``False``         |
| objects                              |               |                   |
+--------------------------------------+---------------+-------------------+

Examples
--------
>>> np.isscalar(3.1)
True
>>> np.isscalar(np.array(3.1))
False
>>> np.isscalar([3.1])
False
>>> np.isscalar(False)
True
>>> np.isscalar('numpy')
True

NumPy supports PEP 3141 numbers:

>>> from fractions import Fraction
>>> np.isscalar(Fraction(5, 17))
True
>>> from numbers import Number
>>> np.isscalar(Number())
True

### issctype(rep)
Module: `numpy`

Determines whether the given object represents a scalar data-type.

Parameters
----------
rep : any
    If `rep` is an instance of a scalar dtype, True is returned. If not,
    False is returned.

Returns
-------
out : bool
    Boolean result of check whether `rep` is a scalar dtype.

See Also
--------
issubsctype, issubdtype, obj2sctype, sctype2char

Examples
--------
>>> np.issctype(np.int32)
True
>>> np.issctype(list)
False
>>> np.issctype(1.1)
False

Strings are also a scalar type:

>>> np.issctype(np.dtype('str'))
True

### issubclass_(arg1, arg2)
Module: `numpy`

Determine if a class is a subclass of a second class.

`issubclass_` is equivalent to the Python built-in ``issubclass``,
except that it returns False instead of raising a TypeError if one
of the arguments is not a class.

Parameters
----------
arg1 : class
    Input class. True is returned if `arg1` is a subclass of `arg2`.
arg2 : class or tuple of classes.
    Input class. If a tuple of classes, True is returned if `arg1` is a
    subclass of any of the tuple elements.

Returns
-------
out : bool
    Whether `arg1` is a subclass of `arg2` or not.

See Also
--------
issubsctype, issubdtype, issctype

Examples
--------
>>> np.issubclass_(np.int32, int)
False
>>> np.issubclass_(np.int32, float)
False
>>> np.issubclass_(np.float64, float)
True

### issubdtype(arg1, arg2)
Module: `numpy`

Returns True if first argument is a typecode lower/equal in type hierarchy.

This is like the builtin :func:`issubclass`, but for `dtype`\ s.

Parameters
----------
arg1, arg2 : dtype_like
    `dtype` or object coercible to one

Returns
-------
out : bool

See Also
--------
:ref:`arrays.scalars` : Overview of the numpy type hierarchy.
issubsctype, issubclass_

Examples
--------
`issubdtype` can be used to check the type of arrays:

>>> ints = np.array([1, 2, 3], dtype=np.int32)
>>> np.issubdtype(ints.dtype, np.integer)
True
>>> np.issubdtype(ints.dtype, np.floating)
False

>>> floats = np.array([1, 2, 3], dtype=np.float32)
>>> np.issubdtype(floats.dtype, np.integer)
False
>>> np.issubdtype(floats.dtype, np.floating)
True

Similar types of different sizes are not subdtypes of each other:

>>> np.issubdtype(np.float64, np.float32)
False
>>> np.issubdtype(np.float32, np.float64)
False

but both are subtypes of `floating`:

>>> np.issubdtype(np.float64, np.floating)
True
>>> np.issubdtype(np.float32, np.floating)
True

For convenience, dtype-like objects are allowed too:

>>> np.issubdtype('S1', np.string_)
True
>>> np.issubdtype('i4', np.signedinteger)
True

### issubsctype(arg1, arg2)
Module: `numpy`

Determine if the first argument is a subclass of the second argument.

Parameters
----------
arg1, arg2 : dtype or dtype specifier
    Data-types.

Returns
-------
out : bool
    The result.

See Also
--------
issctype, issubdtype, obj2sctype

Examples
--------
>>> np.issubsctype('S8', str)
False
>>> np.issubsctype(np.array([1]), int)
True
>>> np.issubsctype(np.array([1]), float)
False

### iterable(y)
Module: `numpy`

Check whether or not an object can be iterated over.

Parameters
----------
y : object
  Input object.

Returns
-------
b : bool
  Return ``True`` if the object has an iterator method or is a
  sequence and ``False`` otherwise.


Examples
--------
>>> np.iterable([1, 2, 3])
True
>>> np.iterable(2)
False

Notes
-----
In most cases, the results of ``np.iterable(obj)`` are consistent with
``isinstance(obj, collections.abc.Iterable)``. One notable exception is
the treatment of 0-dimensional arrays::

    >>> from collections.abc import Iterable
    >>> a = np.array(1.0)  # 0-dimensional numpy array
    >>> isinstance(a, Iterable)
    True
    >>> np.iterable(a)
    False

### kaiser(M, beta)
Module: `numpy`

Return the Kaiser window.

The Kaiser window is a taper formed by using a Bessel function.

Parameters
----------
M : int
    Number of points in the output window. If zero or less, an
    empty array is returned.
beta : float
    Shape parameter for window.

Returns
-------
out : array
    The window, with the maximum value normalized to one (the value
    one appears only if the number of samples is odd).

See Also
--------
bartlett, blackman, hamming, hanning

Notes
-----
The Kaiser window is defined as

.. math::  w(n) = I_0\left( \beta \sqrt{1-\frac{4n^2}{(M-1)^2}}
           \right)/I_0(\beta)

with

.. math:: \quad -\frac{M-1}{2} \leq n \leq \frac{M-1}{2},

where :math:`I_0` is the modified zeroth-order Bessel function.

The Kaiser was named for Jim Kaiser, who discovered a simple
approximation to the DPSS window based on Bessel functions.  The Kaiser
window is a very good approximation to the Digital Prolate Spheroidal
Sequence, or Slepian window, which is the transform which maximizes the
energy in the main lobe of the window relative to total energy.

The Kaiser can approximate many other windows by varying the beta
parameter.

====  =======================
beta  Window shape
====  =======================
0     Rectangular
5     Similar to a Hamming
6     Similar to a Hanning
8.6   Similar to a Blackman
====  =======================

A beta value of 14 is probably a good starting point. Note that as beta
gets large, the window narrows, and so the number of samples needs to be
large enough to sample the increasingly narrow spike, otherwise NaNs will
get returned.

Most references to the Kaiser window come from the signal processing
literature, where it is used as one of many windowing functions for
smoothing values.  It is also known as an apodization (which means
"removing the foot", i.e. smoothing discontinuities at the beginning
and end of the sampled signal) or tapering function.

References
----------
.. [1] J. F. Kaiser, "Digital Filters" - Ch 7 in "Systems analysis by
       digital computer", Editors: F.F. Kuo and J.F. Kaiser, p 218-285.
       John Wiley and Sons, New York, (1966).
.. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
       University of Alberta Press, 1975, pp. 177-178.
.. [3] Wikipedia, "Window function",
       https://en.wikipedia.org/wiki/Window_function

Examples
--------
>>> import matplotlib.pyplot as plt
>>> np.kaiser(12, 14)
 array([7.72686684e-06, 3.46009194e-03, 4.65200189e-02, # may vary
        2.29737120e-01, 5.99885316e-01, 9.45674898e-01,
        9.45674898e-01, 5.99885316e-01, 2.29737120e-01,
        4.65200189e-02, 3.46009194e-03, 7.72686684e-06])


Plot the window and the frequency response:

>>> from numpy.fft import fft, fftshift
>>> window = np.kaiser(51, 14)
>>> plt.plot(window)
[<matplotlib.lines.Line2D object at 0x...>]
>>> plt.title("Kaiser window")
Text(0.5, 1.0, 'Kaiser window')
>>> plt.ylabel("Amplitude")
Text(0, 0.5, 'Amplitude')
>>> plt.xlabel("Sample")
Text(0.5, 0, 'Sample')
>>> plt.show()

>>> plt.figure()
<Figure size 640x480 with 0 Axes>
>>> A = fft(window, 2048) / 25.5
>>> mag = np.abs(fftshift(A))
>>> freq = np.linspace(-0.5, 0.5, len(A))
>>> response = 20 * np.log10(mag)
>>> response = np.clip(response, -100, 100)
>>> plt.plot(freq, response)
[<matplotlib.lines.Line2D object at 0x...>]
>>> plt.title("Frequency response of Kaiser window")
Text(0.5, 1.0, 'Frequency response of Kaiser window')
>>> plt.ylabel("Magnitude [dB]")
Text(0, 0.5, 'Magnitude [dB]')
>>> plt.xlabel("Normalized frequency [cycles per sample]")
Text(0.5, 0, 'Normalized frequency [cycles per sample]')
>>> plt.axis('tight')
(-0.5, 0.5, -100.0, ...) # may vary
>>> plt.show()

### load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII', *, max_header_size=10000)
Module: `numpy`

Load arrays or pickled objects from ``.npy``, ``.npz`` or pickled files.

.. warning:: Loading files that contain object arrays uses the ``pickle``
             module, which is not secure against erroneous or maliciously
             constructed data. Consider passing ``allow_pickle=False`` to
             load data that is known not to contain object arrays for the
             safer handling of untrusted sources.

Parameters
----------
file : file-like object, string, or pathlib.Path
    The file to read. File-like objects must support the
    ``seek()`` and ``read()`` methods and must always
    be opened in binary mode.  Pickled files require that the
    file-like object support the ``readline()`` method as well.
mmap_mode : {None, 'r+', 'r', 'w+', 'c'}, optional
    If not None, then memory-map the file, using the given mode (see
    `numpy.memmap` for a detailed description of the modes).  A
    memory-mapped array is kept on disk. However, it can be accessed
    and sliced like any ndarray.  Memory mapping is especially useful
    for accessing small fragments of large files without reading the
    entire file into memory.
allow_pickle : bool, optional
    Allow loading pickled object arrays stored in npy files. Reasons for
    disallowing pickles include security, as loading pickled data can
    execute arbitrary code. If pickles are disallowed, loading object
    arrays will fail. Default: False

    .. versionchanged:: 1.16.3
        Made default False in response to CVE-2019-6446.

fix_imports : bool, optional
    Only useful when loading Python 2 generated pickled files on Python 3,
    which includes npy/npz files containing object arrays. If `fix_imports`
    is True, pickle will try to map the old Python 2 names to the new names
    used in Python 3.
encoding : str, optional
    What encoding to use when reading Python 2 strings. Only useful when
    loading Python 2 generated pickled files in Python 3, which includes
    npy/npz files containing object arrays. Values other than 'latin1',
    'ASCII', and 'bytes' are not allowed, as they can corrupt numerical
    data. Default: 'ASCII'
max_header_size : int, optional
    Maximum allowed size of the header.  Large headers may not be safe
    to load securely and thus require explicitly passing a larger value.
    See :py:func:`ast.literal_eval()` for details.
    This option is ignored when `allow_pickle` is passed.  In that case
    the file is by definition trusted and the limit is unnecessary.

Returns
-------
result : array, tuple, dict, etc.
    Data stored in the file. For ``.npz`` files, the returned instance
    of NpzFile class must be closed to avoid leaking file descriptors.

Raises
------
OSError
    If the input file does not exist or cannot be read.
UnpicklingError
    If ``allow_pickle=True``, but the file cannot be loaded as a pickle.
ValueError
    The file contains an object array, but ``allow_pickle=False`` given.
EOFError
    When calling ``np.load`` multiple times on the same file handle,
    if all data has already been read

See Also
--------
save, savez, savez_compressed, loadtxt
memmap : Create a memory-map to an array stored in a file on disk.
lib.format.open_memmap : Create or load a memory-mapped ``.npy`` file.

Notes
-----
- If the file contains pickle data, then whatever object is stored
  in the pickle is returned.
- If the file is a ``.npy`` file, then a single array is returned.
- If the file is a ``.npz`` file, then a dictionary-like object is
  returned, containing ``{filename: array}`` key-value pairs, one for
  each file in the archive.
- If the file is a ``.npz`` file, the returned value supports the
  context manager protocol in a similar fashion to the open function::

    with load('foo.npz') as data:
        a = data['a']

  The underlying file descriptor is closed when exiting the 'with'
  block.

Examples
--------
Store data to disk, and load it again:

>>> np.save('/tmp/123', np.array([[1, 2, 3], [4, 5, 6]]))
>>> np.load('/tmp/123.npy')
array([[1, 2, 3],
       [4, 5, 6]])

Store compressed data to disk, and load it again:

>>> a=np.array([[1, 2, 3], [4, 5, 6]])
>>> b=np.array([1, 2])
>>> np.savez('/tmp/123.npz', a=a, b=b)
>>> data = np.load('/tmp/123.npz')
>>> data['a']
array([[1, 2, 3],
       [4, 5, 6]])
>>> data['b']
array([1, 2])
>>> data.close()

Mem-map the stored array, and then access the second row
directly from disk:

>>> X = np.load('/tmp/123.npy', mmap_mode='r')
>>> X[1, :]
memmap([4, 5, 6])

### loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, *, quotechar=None, like=None)
Module: `numpy`

Load data from a text file.

Parameters
----------
fname : file, str, pathlib.Path, list of str, generator
    File, filename, list, or generator to read.  If the filename
    extension is ``.gz`` or ``.bz2``, the file is first decompressed. Note
    that generators must return bytes or strings. The strings
    in a list or produced by a generator are treated as lines.
dtype : data-type, optional
    Data-type of the resulting array; default: float.  If this is a
    structured data-type, the resulting array will be 1-dimensional, and
    each row will be interpreted as an element of the array.  In this
    case, the number of columns used must match the number of fields in
    the data-type.
comments : str or sequence of str or None, optional
    The characters or list of characters used to indicate the start of a
    comment. None implies no comments. For backwards compatibility, byte
    strings will be decoded as 'latin1'. The default is '#'.
delimiter : str, optional
    The character used to separate the values. For backwards compatibility,
    byte strings will be decoded as 'latin1'. The default is whitespace.

    .. versionchanged:: 1.23.0
       Only single character delimiters are supported. Newline characters
       cannot be used as the delimiter.

converters : dict or callable, optional
    Converter functions to customize value parsing. If `converters` is
    callable, the function is applied to all columns, else it must be a
    dict that maps column number to a parser function.
    See examples for further details.
    Default: None.

    .. versionchanged:: 1.23.0
       The ability to pass a single callable to be applied to all columns
       was added.

skiprows : int, optional
    Skip the first `skiprows` lines, including comments; default: 0.
usecols : int or sequence, optional
    Which columns to read, with 0 being the first. For example,
    ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
    The default, None, results in all columns being read.

    .. versionchanged:: 1.11.0
        When a single column has to be read it is possible to use
        an integer instead of a tuple. E.g ``usecols = 3`` reads the
        fourth column the same way as ``usecols = (3,)`` would.
unpack : bool, optional
    If True, the returned array is transposed, so that arguments may be
    unpacked using ``x, y, z = loadtxt(...)``.  When used with a
    structured data-type, arrays are returned for each field.
    Default is False.
ndmin : int, optional
    The returned array will have at least `ndmin` dimensions.
    Otherwise mono-dimensional axes will be squeezed.
    Legal values: 0 (default), 1 or 2.

    .. versionadded:: 1.6.0
encoding : str, optional
    Encoding used to decode the inputfile. Does not apply to input streams.
    The special value 'bytes' enables backward compatibility workarounds
    that ensures you receive byte arrays as results if possible and passes
    'latin1' encoded strings to converters. Override this value to receive
    unicode arrays and pass strings as input to converters.  If set to None
    the system default is used. The default value is 'bytes'.

    .. versionadded:: 1.14.0
max_rows : int, optional
    Read `max_rows` rows of content after `skiprows` lines. The default is
    to read all the rows. Note that empty rows containing no data such as
    empty lines and comment lines are not counted towards `max_rows`,
    while such lines are counted in `skiprows`.

    .. versionadded:: 1.16.0

    .. versionchanged:: 1.23.0
        Lines containing no data, including comment lines (e.g., lines
        starting with '#' or as specified via `comments`) are not counted
        towards `max_rows`.
quotechar : unicode character or None, optional
    The character used to denote the start and end of a quoted item.
    Occurrences of the delimiter or comment characters are ignored within
    a quoted item. The default value is ``quotechar=None``, which means
    quoting support is disabled.

    If two consecutive instances of `quotechar` are found within a quoted
    field, the first is treated as an escape character. See examples.

    .. versionadded:: 1.23.0
like : array_like, optional
    Reference object to allow the creation of arrays which are not
    NumPy arrays. If an array-like passed in as ``like`` supports
    the ``__array_function__`` protocol, the result will be defined
    by it. In this case, it ensures the creation of an array object
    compatible with that passed in via this argument.

    .. versionadded:: 1.20.0

Returns
-------
out : ndarray
    Data read from the text file.

See Also
--------
load, fromstring, fromregex
genfromtxt : Load data with missing values handled as specified.
scipy.io.loadmat : reads MATLAB data files

Notes
-----
This function aims to be a fast reader for simply formatted files.  The
`genfromtxt` function provides more sophisticated handling of, e.g.,
lines with missing values.

Each row in the input text file must have the same number of values to be
able to read all values. If all rows do not have same number of values, a
subset of up to n columns (where n is the least number of values present
in all rows) can be read by specifying the columns via `usecols`.

.. versionadded:: 1.10.0

The strings produced by the Python float.hex method can be used as
input for floats.

Examples
--------
>>> from io import StringIO   # StringIO behaves like a file object
>>> c = StringIO("0 1\n2 3")
>>> np.loadtxt(c)
array([[0., 1.],
       [2., 3.]])

>>> d = StringIO("M 21 72\nF 35 58")
>>> np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
...                      'formats': ('S1', 'i4', 'f4')})
array([(b'M', 21, 72.), (b'F', 35, 58.)],
      dtype=[('gender', 'S1'), ('age', '<i4'), ('weight', '<f4')])

>>> c = StringIO("1,0,2\n3,0,4")
>>> x, y = np.loadtxt(c, delimiter=',', usecols=(0, 2), unpack=True)
>>> x
array([1., 3.])
>>> y
array([2., 4.])

The `converters` argument is used to specify functions to preprocess the
text prior to parsing. `converters` can be a dictionary that maps
preprocessing functions to each column:

>>> s = StringIO("1.618, 2.296\n3.141, 4.669\n")
>>> conv = {
...     0: lambda x: np.floor(float(x)),  # conversion fn for column 0
...     1: lambda x: np.ceil(float(x)),  # conversion fn for column 1
... }
>>> np.loadtxt(s, delimiter=",", converters=conv)
array([[1., 3.],
       [3., 5.]])

`converters` can be a callable instead of a dictionary, in which case it
is applied to all columns:

>>> s = StringIO("0xDE 0xAD\n0xC0 0xDE")
>>> import functools
>>> conv = functools.partial(int, base=16)
>>> np.loadtxt(s, converters=conv)
array([[222., 173.],
       [192., 222.]])

This example shows how `converters` can be used to convert a field
with a trailing minus sign into a negative number.

>>> s = StringIO('10.01 31.25-\n19.22 64.31\n17.57- 63.94')
>>> def conv(fld):
...     return -float(fld[:-1]) if fld.endswith(b'-') else float(fld)
...
>>> np.loadtxt(s, converters=conv)
array([[ 10.01, -31.25],
       [ 19.22,  64.31],
       [-17.57,  63.94]])

Using a callable as the converter can be particularly useful for handling
values with different formatting, e.g. floats with underscores:

>>> s = StringIO("1 2.7 100_000")
>>> np.loadtxt(s, converters=float)
array([1.e+00, 2.7e+00, 1.e+05])

This idea can be extended to automatically handle values specified in
many different formats:

>>> def conv(val):
...     try:
...         return float(val)
...     except ValueError:
...         return float.fromhex(val)
>>> s = StringIO("1, 2.5, 3_000, 0b4, 0x1.4000000000000p+2")
>>> np.loadtxt(s, delimiter=",", converters=conv, encoding=None)
array([1.0e+00, 2.5e+00, 3.0e+03, 1.8e+02, 5.0e+00])

Note that with the default ``encoding="bytes"``, the inputs to the
converter function are latin-1 encoded byte strings. To deactivate the
implicit encoding prior to conversion, use ``encoding=None``

>>> s = StringIO('10.01 31.25-\n19.22 64.31\n17.57- 63.94')
>>> conv = lambda x: -float(x[:-1]) if x.endswith('-') else float(x)
>>> np.loadtxt(s, converters=conv, encoding=None)
array([[ 10.01, -31.25],
       [ 19.22,  64.31],
       [-17.57,  63.94]])

Support for quoted fields is enabled with the `quotechar` parameter.
Comment and delimiter characters are ignored when they appear within a
quoted item delineated by `quotechar`:

>>> s = StringIO('"alpha, #42", 10.0\n"beta, #64", 2.0\n')
>>> dtype = np.dtype([("label", "U12"), ("value", float)])
>>> np.loadtxt(s, dtype=dtype, delimiter=",", quotechar='"')
array([('alpha, #42', 10.), ('beta, #64',  2.)],
      dtype=[('label', '<U12'), ('value', '<f8')])

Quoted fields can be separated by multiple whitespace characters:

>>> s = StringIO('"alpha, #42"       10.0\n"beta, #64" 2.0\n')
>>> dtype = np.dtype([("label", "U12"), ("value", float)])
>>> np.loadtxt(s, dtype=dtype, delimiter=None, quotechar='"')
array([('alpha, #42', 10.), ('beta, #64',  2.)],
      dtype=[('label', '<U12'), ('value', '<f8')])

Two consecutive quote characters within a quoted field are treated as a
single escaped character:

>>> s = StringIO('"Hello, my name is ""Monty""!"')
>>> np.loadtxt(s, dtype="U", delimiter=",", quotechar='"')
array('Hello, my name is "Monty"!', dtype='<U26')

Read subset of columns when all rows do not contain equal number of values:

>>> d = StringIO("1 2\n2 4\n3 9 12\n4 16 20")
>>> np.loadtxt(d, usecols=(0, 1))
array([[ 1.,  2.],
       [ 2.,  4.],
       [ 3.,  9.],
       [ 4., 16.]])

### lookfor(what, module=None, import_modules=True, regenerate=False, output=None)
Module: `numpy`

Do a keyword search on docstrings.

A list of objects that matched the search is displayed,
sorted by relevance. All given keywords need to be found in the
docstring for it to be returned as a result, but the order does
not matter.

Parameters
----------
what : str
    String containing words to look for.
module : str or list, optional
    Name of module(s) whose docstrings to go through.
import_modules : bool, optional
    Whether to import sub-modules in packages. Default is True.
regenerate : bool, optional
    Whether to re-generate the docstring cache. Default is False.
output : file-like, optional
    File-like object to write the output to. If omitted, use a pager.

See Also
--------
source, info

Notes
-----
Relevance is determined only roughly, by checking if the keywords occur
in the function name, at the start of a docstring, etc.

Examples
--------
>>> np.lookfor('binary representation') # doctest: +SKIP
Search results for 'binary representation'
------------------------------------------
numpy.binary_repr
    Return the binary representation of the input number as a string.
numpy.core.setup_common.long_double_representation
    Given a binary dump as given by GNU od -b, look for long double
numpy.base_repr
    Return a string representation of a number in the given base system.
...

### mask_indices(n, mask_func, k=0)
Module: `numpy`

Return the indices to access (n, n) arrays, given a masking function.

Assume `mask_func` is a function that, for a square array a of size
``(n, n)`` with a possible offset argument `k`, when called as
``mask_func(a, k)`` returns a new array with zeros in certain locations
(functions like `triu` or `tril` do precisely this). Then this function
returns the indices where the non-zero values would be located.

Parameters
----------
n : int
    The returned indices will be valid to access arrays of shape (n, n).
mask_func : callable
    A function whose call signature is similar to that of `triu`, `tril`.
    That is, ``mask_func(x, k)`` returns a boolean array, shaped like `x`.
    `k` is an optional argument to the function.
k : scalar
    An optional argument which is passed through to `mask_func`. Functions
    like `triu`, `tril` take a second argument that is interpreted as an
    offset.

Returns
-------
indices : tuple of arrays.
    The `n` arrays of indices corresponding to the locations where
    ``mask_func(np.ones((n, n)), k)`` is True.

See Also
--------
triu, tril, triu_indices, tril_indices

Notes
-----
.. versionadded:: 1.4.0

Examples
--------
These are the indices that would allow you to access the upper triangular
part of any 3x3 array:

>>> iu = np.mask_indices(3, np.triu)

For example, if `a` is a 3x3 array:

>>> a = np.arange(9).reshape(3, 3)
>>> a
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
>>> a[iu]
array([0, 1, 2, 4, 5, 8])

An offset can be passed also to the masking function.  This gets us the
indices starting on the first diagonal right of the main one:

>>> iu1 = np.mask_indices(3, np.triu, 1)

with which we now extract only three elements:

>>> a[iu1]
array([1, 2, 5])

### mat(data, dtype=None)
Module: `numpy`

Interpret the input as a matrix.

Unlike `matrix`, `asmatrix` does not make a copy if the input is already
a matrix or an ndarray.  Equivalent to ``matrix(data, copy=False)``.

Parameters
----------
data : array_like
    Input data.
dtype : data-type
   Data-type of the output matrix.

Returns
-------
mat : matrix
    `data` interpreted as a matrix.

Examples
--------
>>> x = np.array([[1, 2], [3, 4]])

>>> m = np.asmatrix(x)

>>> x[0,0] = 5

>>> m
matrix([[5, 2],
        [3, 4]])

### maximum_sctype(t)
Module: `numpy`

Return the scalar type of highest precision of the same kind as the input.

Parameters
----------
t : dtype or dtype specifier
    The input data type. This can be a `dtype` object or an object that
    is convertible to a `dtype`.

Returns
-------
out : dtype
    The highest precision data type of the same kind (`dtype.kind`) as `t`.

See Also
--------
obj2sctype, mintypecode, sctype2char
dtype

Examples
--------
>>> np.maximum_sctype(int)
<class 'numpy.int64'>
>>> np.maximum_sctype(np.uint8)
<class 'numpy.uint64'>
>>> np.maximum_sctype(complex)
<class 'numpy.complex256'> # may vary

>>> np.maximum_sctype(str)
<class 'numpy.str_'>

>>> np.maximum_sctype('i2')
<class 'numpy.int64'>
>>> np.maximum_sctype('f4')
<class 'numpy.float128'> # may vary

### mintypecode(typechars, typeset='GDFgdf', default='d')
Module: `numpy`

Return the character for the minimum-size type to which given types can
be safely cast.

The returned type character must represent the smallest size dtype such
that an array of the returned type can handle the data from an array of
all types in `typechars` (or if `typechars` is an array, then its
dtype.char).

Parameters
----------
typechars : list of str or array_like
    If a list of strings, each string should represent a dtype.
    If array_like, the character representation of the array dtype is used.
typeset : str or list of str, optional
    The set of characters that the returned character is chosen from.
    The default set is 'GDFgdf'.
default : str, optional
    The default character, this is returned if none of the characters in
    `typechars` matches a character in `typeset`.

Returns
-------
typechar : str
    The character representing the minimum-size type that was found.

See Also
--------
dtype, sctype2char, maximum_sctype

Examples
--------
>>> np.mintypecode(['d', 'f', 'S'])
'd'
>>> x = np.array([1.1, 2-3.j])
>>> np.mintypecode(x)
'D'

>>> np.mintypecode('abceh', default='G')
'G'

### obj2sctype(rep, default=None)
Module: `numpy`

Return the scalar dtype or NumPy equivalent of Python type of an object.

Parameters
----------
rep : any
    The object of which the type is returned.
default : any, optional
    If given, this is returned for objects whose types can not be
    determined. If not given, None is returned for those objects.

Returns
-------
dtype : dtype or Python type
    The data type of `rep`.

See Also
--------
sctype2char, issctype, issubsctype, issubdtype, maximum_sctype

Examples
--------
>>> np.obj2sctype(np.int32)
<class 'numpy.int32'>
>>> np.obj2sctype(np.array([1., 2.]))
<class 'numpy.float64'>
>>> np.obj2sctype(np.array([1.j]))
<class 'numpy.complex128'>

>>> np.obj2sctype(dict)
<class 'numpy.object_'>
>>> np.obj2sctype('string')

>>> np.obj2sctype(1, default=list)
<class 'list'>

### ones(shape, dtype=None, order='C', *, like=None)
Module: `numpy`

Return a new array of given shape and type, filled with ones.

Parameters
----------
shape : int or sequence of ints
    Shape of the new array, e.g., ``(2, 3)`` or ``2``.
dtype : data-type, optional
    The desired data-type for the array, e.g., `numpy.int8`.  Default is
    `numpy.float64`.
order : {'C', 'F'}, optional, default: C
    Whether to store multi-dimensional data in row-major
    (C-style) or column-major (Fortran-style) order in
    memory.
like : array_like, optional
    Reference object to allow the creation of arrays which are not
    NumPy arrays. If an array-like passed in as ``like`` supports
    the ``__array_function__`` protocol, the result will be defined
    by it. In this case, it ensures the creation of an array object
    compatible with that passed in via this argument.

    .. versionadded:: 1.20.0

Returns
-------
out : ndarray
    Array of ones with the given shape, dtype, and order.

See Also
--------
ones_like : Return an array of ones with shape and type of input.
empty : Return a new uninitialized array.
zeros : Return a new array setting values to zero.
full : Return a new array of given shape filled with value.


Examples
--------
>>> np.ones(5)
array([1., 1., 1., 1., 1.])

>>> np.ones((5,), dtype=int)
array([1, 1, 1, 1, 1])

>>> np.ones((2, 1))
array([[1.],
       [1.]])

>>> s = (2,2)
>>> np.ones(s)
array([[1.,  1.],
       [1.,  1.]])

### printoptions(*args, **kwargs)
Module: `numpy`

Context manager for setting print options.

Set print options for the scope of the `with` block, and restore the old
options at the end. See `set_printoptions` for the full description of
available options.

Examples
--------

>>> from numpy.testing import assert_equal
>>> with np.printoptions(precision=2):
...     np.array([2.0]) / 3
array([0.67])

The `as`-clause of the `with`-statement gives the current print options:

>>> with np.printoptions(precision=2) as opts:
...      assert_equal(opts, np.get_printoptions())

See Also
--------
set_printoptions, get_printoptions

### recfromcsv(fname, **kwargs)
Module: `numpy.lib.npyio`

Load ASCII data stored in a comma-separated file.

The returned array is a record array (if ``usemask=False``, see
`recarray`) or a masked record array (if ``usemask=True``,
see `ma.mrecords.MaskedRecords`).

Parameters
----------
fname, kwargs : For a description of input parameters, see `genfromtxt`.

See Also
--------
numpy.genfromtxt : generic function to load ASCII data.

Notes
-----
By default, `dtype` is None, which means that the data-type of the output
array will be determined from the data.

### recfromtxt(fname, **kwargs)
Module: `numpy.lib.npyio`

Load ASCII data from a file and return it in a record array.

If ``usemask=False`` a standard `recarray` is returned,
if ``usemask=True`` a MaskedRecords array is returned.

Parameters
----------
fname, kwargs : For a description of input parameters, see `genfromtxt`.

See Also
--------
numpy.genfromtxt : generic function

Notes
-----
By default, `dtype` is None, which means that the data-type of the output
array will be determined from the data.

### require(a, dtype=None, requirements=None, *, like=None)
Module: `numpy`

Return an ndarray of the provided type that satisfies requirements.

This function is useful to be sure that an array with the correct flags
is returned for passing to compiled code (perhaps through ctypes).

Parameters
----------
a : array_like
   The object to be converted to a type-and-requirement-satisfying array.
dtype : data-type
   The required data-type. If None preserve the current dtype. If your
   application requires the data to be in native byteorder, include
   a byteorder specification as a part of the dtype specification.
requirements : str or sequence of str
   The requirements list can be any of the following

   * 'F_CONTIGUOUS' ('F') - ensure a Fortran-contiguous array
   * 'C_CONTIGUOUS' ('C') - ensure a C-contiguous array
   * 'ALIGNED' ('A')      - ensure a data-type aligned array
   * 'WRITEABLE' ('W')    - ensure a writable array
   * 'OWNDATA' ('O')      - ensure an array that owns its own data
   * 'ENSUREARRAY', ('E') - ensure a base array, instead of a subclass
like : array_like, optional
    Reference object to allow the creation of arrays which are not
    NumPy arrays. If an array-like passed in as ``like`` supports
    the ``__array_function__`` protocol, the result will be defined
    by it. In this case, it ensures the creation of an array object
    compatible with that passed in via this argument.

    .. versionadded:: 1.20.0

Returns
-------
out : ndarray
    Array with specified requirements and type if given.

See Also
--------
asarray : Convert input to an ndarray.
asanyarray : Convert to an ndarray, but pass through ndarray subclasses.
ascontiguousarray : Convert input to a contiguous array.
asfortranarray : Convert input to an ndarray with column-major
                 memory order.
ndarray.flags : Information about the memory layout of the array.

Notes
-----
The returned array will be guaranteed to have the listed requirements
by making a copy if needed.

Examples
--------
>>> x = np.arange(6).reshape(2,3)
>>> x.flags
  C_CONTIGUOUS : True
  F_CONTIGUOUS : False
  OWNDATA : False
  WRITEABLE : True
  ALIGNED : True
  WRITEBACKIFCOPY : False

>>> y = np.require(x, dtype=np.float32, requirements=['A', 'O', 'W', 'F'])
>>> y.flags
  C_CONTIGUOUS : False
  F_CONTIGUOUS : True
  OWNDATA : True
  WRITEABLE : True
  ALIGNED : True
  WRITEBACKIFCOPY : False

### safe_eval(source)
Module: `numpy.lib.utils`

Protected string evaluation.

Evaluate a string containing a Python literal expression without
allowing the execution of arbitrary non-literal code.

.. warning::

    This function is identical to :py:meth:`ast.literal_eval` and
    has the same security implications.  It may not always be safe
    to evaluate large input strings.

Parameters
----------
source : str
    The string to evaluate.

Returns
-------
obj : object
   The result of evaluating `source`.

Raises
------
SyntaxError
    If the code has invalid Python syntax, or if it contains
    non-literal code.

Examples
--------
>>> np.safe_eval('1')
1
>>> np.safe_eval('[1, 2, 3]')
[1, 2, 3]
>>> np.safe_eval('{"foo": ("bar", 10.0)}')
{'foo': ('bar', 10.0)}

>>> np.safe_eval('import os')
Traceback (most recent call last):
  ...
SyntaxError: invalid syntax

>>> np.safe_eval('open("/home/user/.ssh/id_dsa").read()')
Traceback (most recent call last):
  ...
ValueError: malformed node or string: <_ast.Call object at 0x...>

### sctype2char(sctype)
Module: `numpy`

Return the string representation of a scalar dtype.

Parameters
----------
sctype : scalar dtype or object
    If a scalar dtype, the corresponding string character is
    returned. If an object, `sctype2char` tries to infer its scalar type
    and then return the corresponding string character.

Returns
-------
typechar : str
    The string character corresponding to the scalar type.

Raises
------
ValueError
    If `sctype` is an object for which the type can not be inferred.

See Also
--------
obj2sctype, issctype, issubsctype, mintypecode

Examples
--------
>>> for sctype in [np.int32, np.double, np.complex_, np.string_, np.ndarray]:
...     print(np.sctype2char(sctype))
l # may vary
d
D
S
O

>>> x = np.array([1., 2-1.j])
>>> np.sctype2char(x)
'D'
>>> np.sctype2char(list)
'O'

### set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, suppress=None, nanstr=None, infstr=None, formatter=None, sign=None, floatmode=None, *, legacy=None)
Module: `numpy`

Set printing options.

These options determine the way floating point numbers, arrays and
other NumPy objects are displayed.

Parameters
----------
precision : int or None, optional
    Number of digits of precision for floating point output (default 8).
    May be None if `floatmode` is not `fixed`, to print as many digits as
    necessary to uniquely specify the value.
threshold : int, optional
    Total number of array elements which trigger summarization
    rather than full repr (default 1000).
    To always use the full repr without summarization, pass `sys.maxsize`.
edgeitems : int, optional
    Number of array items in summary at beginning and end of
    each dimension (default 3).
linewidth : int, optional
    The number of characters per line for the purpose of inserting
    line breaks (default 75).
suppress : bool, optional
    If True, always print floating point numbers using fixed point
    notation, in which case numbers equal to zero in the current precision
    will print as zero.  If False, then scientific notation is used when
    absolute value of the smallest number is < 1e-4 or the ratio of the
    maximum absolute value to the minimum is > 1e3. The default is False.
nanstr : str, optional
    String representation of floating point not-a-number (default nan).
infstr : str, optional
    String representation of floating point infinity (default inf).
sign : string, either '-', '+', or ' ', optional
    Controls printing of the sign of floating-point types. If '+', always
    print the sign of positive values. If ' ', always prints a space
    (whitespace character) in the sign position of positive values.  If
    '-', omit the sign character of positive values. (default '-')
formatter : dict of callables, optional
    If not None, the keys should indicate the type(s) that the respective
    formatting function applies to.  Callables should return a string.
    Types that are not specified (by their corresponding keys) are handled
    by the default formatters.  Individual types for which a formatter
    can be set are:

    - 'bool'
    - 'int'
    - 'timedelta' : a `numpy.timedelta64`
    - 'datetime' : a `numpy.datetime64`
    - 'float'
    - 'longfloat' : 128-bit floats
    - 'complexfloat'
    - 'longcomplexfloat' : composed of two 128-bit floats
    - 'numpystr' : types `numpy.bytes_` and `numpy.str_`
    - 'object' : `np.object_` arrays

    Other keys that can be used to set a group of types at once are:

    - 'all' : sets all types
    - 'int_kind' : sets 'int'
    - 'float_kind' : sets 'float' and 'longfloat'
    - 'complex_kind' : sets 'complexfloat' and 'longcomplexfloat'
    - 'str_kind' : sets 'numpystr'
floatmode : str, optional
    Controls the interpretation of the `precision` option for
    floating-point types. Can take the following values
    (default maxprec_equal):

    * 'fixed': Always print exactly `precision` fractional digits,
            even if this would print more or fewer digits than
            necessary to specify the value uniquely.
    * 'unique': Print the minimum number of fractional digits necessary
            to represent each value uniquely. Different elements may
            have a different number of digits. The value of the
            `precision` option is ignored.
    * 'maxprec': Print at most `precision` fractional digits, but if
            an element can be uniquely represented with fewer digits
            only print it with that many.
    * 'maxprec_equal': Print at most `precision` fractional digits,
            but if every element in the array can be uniquely
            represented with an equal number of fewer digits, use that
            many digits for all elements.
legacy : string or `False`, optional
    If set to the string `'1.13'` enables 1.13 legacy printing mode. This
    approximates numpy 1.13 print output by including a space in the sign
    position of floats and different behavior for 0d arrays. This also
    enables 1.21 legacy printing mode (described below).

    If set to the string `'1.21'` enables 1.21 legacy printing mode. This
    approximates numpy 1.21 print output of complex structured dtypes
    by not inserting spaces after commas that separate fields and after
    colons.

    If set to `False`, disables legacy mode.

    Unrecognized strings will be ignored with a warning for forward
    compatibility.

    .. versionadded:: 1.14.0
    .. versionchanged:: 1.22.0

See Also
--------
get_printoptions, printoptions, set_string_function, array2string

Notes
-----
`formatter` is always reset with a call to `set_printoptions`.

Use `printoptions` as a context manager to set the values temporarily.

Examples
--------
Floating point precision can be set:

>>> np.set_printoptions(precision=4)
>>> np.array([1.123456789])
[1.1235]

Long arrays can be summarised:

>>> np.set_printoptions(threshold=5)
>>> np.arange(10)
array([0, 1, 2, ..., 7, 8, 9])

Small results can be suppressed:

>>> eps = np.finfo(float).eps
>>> x = np.arange(4.)
>>> x**2 - (x + eps)**2
array([-4.9304e-32, -4.4409e-16,  0.0000e+00,  0.0000e+00])
>>> np.set_printoptions(suppress=True)
>>> x**2 - (x + eps)**2
array([-0., -0.,  0.,  0.])

A custom formatter can be used to display array elements as desired:

>>> np.set_printoptions(formatter={'all':lambda x: 'int: '+str(-x)})
>>> x = np.arange(3)
>>> x
array([int: 0, int: -1, int: -2])
>>> np.set_printoptions()  # formatter gets reset
>>> x
array([0, 1, 2])

To put back the default options, you can use:

>>> np.set_printoptions(edgeitems=3, infstr='inf',
... linewidth=75, nanstr='nan', precision=8,
... suppress=False, threshold=1000, formatter=None)

Also to temporarily override options, use `printoptions` as a context manager:

>>> with np.printoptions(precision=2, suppress=True, threshold=5):
...     np.linspace(0, 10, 10)
array([ 0.  ,  1.11,  2.22, ...,  7.78,  8.89, 10.  ])

### set_string_function(f, repr=True)
Module: `numpy.core.arrayprint`

Set a Python function to be used when pretty printing arrays.

Parameters
----------
f : function or None
    Function to be used to pretty print arrays. The function should expect
    a single array argument and return a string of the representation of
    the array. If None, the function is reset to the default NumPy function
    to print arrays.
repr : bool, optional
    If True (default), the function for pretty printing (``__repr__``)
    is set, if False the function that returns the default string
    representation (``__str__``) is set.

See Also
--------
set_printoptions, get_printoptions

Examples
--------
>>> def pprint(arr):
...     return 'HA! - What are you going to do now?'
...
>>> np.set_string_function(pprint)
>>> a = np.arange(10)
>>> a
HA! - What are you going to do now?
>>> _ = a
>>> # [0 1 2 3 4 5 6 7 8 9]

We can reset the function to the default:

>>> np.set_string_function(None)
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

`repr` affects either pretty printing or normal string representation.
Note that ``__repr__`` is still affected by setting ``__str__``
because the width of each array element in the returned string becomes
equal to the length of the result of ``__str__()``.

>>> x = np.arange(4)
>>> np.set_string_function(lambda x:'random', repr=False)
>>> x.__str__()
'random'
>>> x.__repr__()
'array([0, 1, 2, 3])'

### setbufsize(size)
Module: `numpy`

Set the size of the buffer used in ufuncs.

Parameters
----------
size : int
    Size of buffer.

### seterr(all=None, divide=None, over=None, under=None, invalid=None)
Module: `numpy`

Set how floating-point errors are handled.

Note that operations on integer scalar types (such as `int16`) are
handled like floating point, and are affected by these settings.

Parameters
----------
all : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
    Set treatment for all types of floating-point errors at once:

    - ignore: Take no action when the exception occurs.
    - warn: Print a `RuntimeWarning` (via the Python `warnings` module).
    - raise: Raise a `FloatingPointError`.
    - call: Call a function specified using the `seterrcall` function.
    - print: Print a warning directly to ``stdout``.
    - log: Record error in a Log object specified by `seterrcall`.

    The default is not to change the current behavior.
divide : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
    Treatment for division by zero.
over : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
    Treatment for floating-point overflow.
under : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
    Treatment for floating-point underflow.
invalid : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
    Treatment for invalid floating-point operation.

Returns
-------
old_settings : dict
    Dictionary containing the old settings.

See also
--------
seterrcall : Set a callback function for the 'call' mode.
geterr, geterrcall, errstate

Notes
-----
The floating-point exceptions are defined in the IEEE 754 standard [1]_:

- Division by zero: infinite result obtained from finite numbers.
- Overflow: result too large to be expressed.
- Underflow: result so close to zero that some precision
  was lost.
- Invalid operation: result is not an expressible number, typically
  indicates that a NaN was produced.

.. [1] https://en.wikipedia.org/wiki/IEEE_754

Examples
--------
>>> old_settings = np.seterr(all='ignore')  #seterr to known value
>>> np.seterr(over='raise')
{'divide': 'ignore', 'over': 'ignore', 'under': 'ignore', 'invalid': 'ignore'}
>>> np.seterr(**old_settings)  # reset to default
{'divide': 'ignore', 'over': 'raise', 'under': 'ignore', 'invalid': 'ignore'}

>>> np.int16(32000) * np.int16(3)
30464
>>> old_settings = np.seterr(all='warn', over='raise')
>>> np.int16(32000) * np.int16(3)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
FloatingPointError: overflow encountered in scalar multiply

>>> old_settings = np.seterr(all='print')
>>> np.geterr()
{'divide': 'print', 'over': 'print', 'under': 'print', 'invalid': 'print'}
>>> np.int16(32000) * np.int16(3)
30464

### seterrcall(func)
Module: `numpy`

Set the floating-point error callback function or log object.

There are two ways to capture floating-point error messages.  The first
is to set the error-handler to 'call', using `seterr`.  Then, set
the function to call using this function.

The second is to set the error-handler to 'log', using `seterr`.
Floating-point errors then trigger a call to the 'write' method of
the provided object.

Parameters
----------
func : callable f(err, flag) or object with write method
    Function to call upon floating-point errors ('call'-mode) or
    object whose 'write' method is used to log such message ('log'-mode).

    The call function takes two arguments. The first is a string describing
    the type of error (such as "divide by zero", "overflow", "underflow",
    or "invalid value"), and the second is the status flag.  The flag is a
    byte, whose four least-significant bits indicate the type of error, one
    of "divide", "over", "under", "invalid"::

      [0 0 0 0 divide over under invalid]

    In other words, ``flags = divide + 2*over + 4*under + 8*invalid``.

    If an object is provided, its write method should take one argument,
    a string.

Returns
-------
h : callable, log instance or None
    The old error handler.

See Also
--------
seterr, geterr, geterrcall

Examples
--------
Callback upon error:

>>> def err_handler(type, flag):
...     print("Floating point error (%s), with flag %s" % (type, flag))
...

>>> saved_handler = np.seterrcall(err_handler)
>>> save_err = np.seterr(all='call')

>>> np.array([1, 2, 3]) / 0.0
Floating point error (divide by zero), with flag 1
array([inf, inf, inf])

>>> np.seterrcall(saved_handler)
<function err_handler at 0x...>
>>> np.seterr(**save_err)
{'divide': 'call', 'over': 'call', 'under': 'call', 'invalid': 'call'}

Log error message:

>>> class Log:
...     def write(self, msg):
...         print("LOG: %s" % msg)
...

>>> log = Log()
>>> saved_handler = np.seterrcall(log)
>>> save_err = np.seterr(all='log')

>>> np.array([1, 2, 3]) / 0.0
LOG: Warning: divide by zero encountered in divide
array([inf, inf, inf])

>>> np.seterrcall(saved_handler)
<numpy.core.numeric.Log object at 0x...>
>>> np.seterr(**save_err)
{'divide': 'log', 'over': 'log', 'under': 'log', 'invalid': 'log'}

### show_config(mode='stdout')
Module: `numpy.__config__`

Show libraries and system information on which NumPy was built
and is being used

Parameters
----------
mode : {`'stdout'`, `'dicts'`}, optional.
    Indicates how to display the config information.
    `'stdout'` prints to console, `'dicts'` returns a dictionary
    of the configuration.

Returns
-------
out : {`dict`, `None`}
    If mode is `'dicts'`, a dict is returned, else None

See Also
--------
get_include : Returns the directory containing NumPy C
              header files.

Notes
-----
1. The `'stdout'` mode will give more readable
   output if ``pyyaml`` is installed

### show_runtime()
Module: `numpy.lib.utils`

Print information about various resources in the system
including available intrinsic support and BLAS/LAPACK library
in use

.. versionadded:: 1.24.0

See Also
--------
show_config : Show libraries in the system on which NumPy was built.

Notes
-----
1. Information is derived with the help of `threadpoolctl <https://pypi.org/project/threadpoolctl/>`_
   library if available.
2. SIMD related information is derived from ``__cpu_features__``,
   ``__cpu_baseline__`` and ``__cpu_dispatch__``

### source(object, output=<_io.TextIOWrapper name='<stdout>' mode='w' encoding='utf-8'>)
Module: `numpy`

Print or write to a file the source code for a NumPy object.

The source code is only returned for objects written in Python. Many
functions and classes are defined in C and will therefore not return
useful information.

Parameters
----------
object : numpy object
    Input object. This can be any object (function, class, module,
    ...).
output : file object, optional
    If `output` not supplied then source code is printed to screen
    (sys.stdout).  File object must be created with either write 'w' or
    append 'a' modes.

See Also
--------
lookfor, info

Examples
--------
>>> np.source(np.interp)                        #doctest: +SKIP
In file: /usr/lib/python2.6/dist-packages/numpy/lib/function_base.py
def interp(x, xp, fp, left=None, right=None):
    """.... (full docstring printed)"""
    if isinstance(x, (float, int, number)):
        return compiled_interp([x], xp, fp, left, right).item()
    else:
        return compiled_interp(x, xp, fp, left, right)

The source code is only returned for objects written in Python.

>>> np.source(np.array)                         #doctest: +SKIP
Not available for this object.

### tri(N, M=None, k=0, dtype=<class 'float'>, *, like=None)
Module: `numpy`

An array with ones at and below the given diagonal and zeros elsewhere.

Parameters
----------
N : int
    Number of rows in the array.
M : int, optional
    Number of columns in the array.
    By default, `M` is taken equal to `N`.
k : int, optional
    The sub-diagonal at and below which the array is filled.
    `k` = 0 is the main diagonal, while `k` < 0 is below it,
    and `k` > 0 is above.  The default is 0.
dtype : dtype, optional
    Data type of the returned array.  The default is float.
like : array_like, optional
    Reference object to allow the creation of arrays which are not
    NumPy arrays. If an array-like passed in as ``like`` supports
    the ``__array_function__`` protocol, the result will be defined
    by it. In this case, it ensures the creation of an array object
    compatible with that passed in via this argument.

    .. versionadded:: 1.20.0

Returns
-------
tri : ndarray of shape (N, M)
    Array with its lower triangle filled with ones and zero elsewhere;
    in other words ``T[i,j] == 1`` for ``j <= i + k``, 0 otherwise.

Examples
--------
>>> np.tri(3, 5, 2, dtype=int)
array([[1, 1, 1, 0, 0],
       [1, 1, 1, 1, 0],
       [1, 1, 1, 1, 1]])

>>> np.tri(3, 5, -1)
array([[0.,  0.,  0.,  0.,  0.],
       [1.,  0.,  0.,  0.,  0.],
       [1.,  1.,  0.,  0.,  0.]])

### tril_indices(n, k=0, m=None)
Module: `numpy`

Return the indices for the lower-triangle of an (n, m) array.

Parameters
----------
n : int
    The row dimension of the arrays for which the returned
    indices will be valid.
k : int, optional
    Diagonal offset (see `tril` for details).
m : int, optional
    .. versionadded:: 1.9.0

    The column dimension of the arrays for which the returned
    arrays will be valid.
    By default `m` is taken equal to `n`.


Returns
-------
inds : tuple of arrays
    The indices for the triangle. The returned tuple contains two arrays,
    each with the indices along one dimension of the array.

See also
--------
triu_indices : similar function, for upper-triangular.
mask_indices : generic function accepting an arbitrary mask function.
tril, triu

Notes
-----
.. versionadded:: 1.4.0

Examples
--------
Compute two different sets of indices to access 4x4 arrays, one for the
lower triangular part starting at the main diagonal, and one starting two
diagonals further right:

>>> il1 = np.tril_indices(4)
>>> il2 = np.tril_indices(4, 2)

Here is how they can be used with a sample array:

>>> a = np.arange(16).reshape(4, 4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])

Both for indexing:

>>> a[il1]
array([ 0,  4,  5, ..., 13, 14, 15])

And for assigning values:

>>> a[il1] = -1
>>> a
array([[-1,  1,  2,  3],
       [-1, -1,  6,  7],
       [-1, -1, -1, 11],
       [-1, -1, -1, -1]])

These cover almost the whole array (two diagonals right of the main one):

>>> a[il2] = -10
>>> a
array([[-10, -10, -10,   3],
       [-10, -10, -10, -10],
       [-10, -10, -10, -10],
       [-10, -10, -10, -10]])

### triu_indices(n, k=0, m=None)
Module: `numpy`

Return the indices for the upper-triangle of an (n, m) array.

Parameters
----------
n : int
    The size of the arrays for which the returned indices will
    be valid.
k : int, optional
    Diagonal offset (see `triu` for details).
m : int, optional
    .. versionadded:: 1.9.0

    The column dimension of the arrays for which the returned
    arrays will be valid.
    By default `m` is taken equal to `n`.


Returns
-------
inds : tuple, shape(2) of ndarrays, shape(`n`)
    The indices for the triangle. The returned tuple contains two arrays,
    each with the indices along one dimension of the array.  Can be used
    to slice a ndarray of shape(`n`, `n`).

See also
--------
tril_indices : similar function, for lower-triangular.
mask_indices : generic function accepting an arbitrary mask function.
triu, tril

Notes
-----
.. versionadded:: 1.4.0

Examples
--------
Compute two different sets of indices to access 4x4 arrays, one for the
upper triangular part starting at the main diagonal, and one starting two
diagonals further right:

>>> iu1 = np.triu_indices(4)
>>> iu2 = np.triu_indices(4, 2)

Here is how they can be used with a sample array:

>>> a = np.arange(16).reshape(4, 4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])

Both for indexing:

>>> a[iu1]
array([ 0,  1,  2, ..., 10, 11, 15])

And for assigning values:

>>> a[iu1] = -1
>>> a
array([[-1, -1, -1, -1],
       [ 4, -1, -1, -1],
       [ 8,  9, -1, -1],
       [12, 13, 14, -1]])

These cover only a small part of the whole array (two diagonals right
of the main one):

>>> a[iu2] = -10
>>> a
array([[ -1,  -1, -10, -10],
       [  4,  -1,  -1, -10],
       [  8,   9,  -1,  -1],
       [ 12,  13,  14,  -1]])

### typename(char)
Module: `numpy`

Return a description for the given data type code.

Parameters
----------
char : str
    Data type code.

Returns
-------
out : str
    Description of the input data type code.

See Also
--------
dtype, typecodes

Examples
--------
>>> typechars = ['S1', '?', 'B', 'D', 'G', 'F', 'I', 'H', 'L', 'O', 'Q',
...              'S', 'U', 'V', 'b', 'd', 'g', 'f', 'i', 'h', 'l', 'q']
>>> for typechar in typechars:
...     print(typechar, ' : ', np.typename(typechar))
...
S1  :  character
?  :  bool
B  :  unsigned char
D  :  complex double precision
G  :  complex long double precision
F  :  complex single precision
I  :  unsigned integer
H  :  unsigned short
L  :  unsigned long integer
O  :  object
Q  :  unsigned long long integer
S  :  string
U  :  unicode
V  :  void
b  :  signed char
d  :  double precision
g  :  long precision
f  :  single precision
i  :  integer
h  :  short
l  :  long integer
q  :  long long integer

### who(vardict=None)
Module: `numpy.lib.utils`

Print the NumPy arrays in the given dictionary.

If there is no dictionary passed in or `vardict` is None then returns
NumPy arrays in the globals() dictionary (all NumPy arrays in the
namespace).

Parameters
----------
vardict : dict, optional
    A dictionary possibly containing ndarrays.  Default is globals().

Returns
-------
out : None
    Returns 'None'.

Notes
-----
Prints out the name, shape, bytes and type of all of the ndarrays
present in `vardict`.

Examples
--------
>>> a = np.arange(10)
>>> b = np.ones(20)
>>> np.who()
Name            Shape            Bytes            Type
===========================================================
a               10               80               int64
b               20               160              float64
Upper bound on total bytes  =       240

>>> d = {'x': np.arange(2.0), 'y': np.arange(3.0), 'txt': 'Some str',
... 'idx':5}
>>> np.who(d)
Name            Shape            Bytes            Type
===========================================================
x               2                16               float64
y               3                24               float64
Upper bound on total bytes  =       40

## Classes

### DataSource
Module: `numpy`

DataSource(destpath='.')

A generic data source file (file, http, ftp, ...).

DataSources can be local files or remote files/URLs.  The files may
also be compressed or uncompressed. DataSource hides some of the
low-level details of downloading the file, allowing you to simply pass
in a valid file path (or URL) and obtain a file object.

Parameters
----------
destpath : str or None, optional
    Path to the directory where the source file gets downloaded to for
    use.  If `destpath` is None, a temporary directory will be created.
    The default path is the current directory.

Notes
-----
URLs require a scheme string (``http://``) to be used, without it they
will fail::

    >>> repos = np.DataSource()
    >>> repos.exists('www.google.com/index.html')
    False
    >>> repos.exists('http://www.google.com/index.html')
    True

Temporary directories are deleted when the DataSource is deleted.

Examples
--------
::

    >>> ds = np.DataSource('/home/guido')
    >>> urlname = 'http://www.google.com/'
    >>> gfile = ds.open('http://www.google.com/')
    >>> ds.abspath(urlname)
    '/home/guido/www.google.com/index.html'

    >>> ds = np.DataSource(None)  # use with temporary file
    >>> ds.open('/home/guido/foobar.txt')
    <open file '/home/guido.foobar.txt', mode 'r' at 0x91d4430>
    >>> ds.abspath('/home/guido/foobar.txt')
    '/tmp/.../home/guido/foobar.txt'

#### Methods

**`abspath(self, path)`**

Return absolute path of file in the DataSource directory.

If `path` is an URL, then `abspath` will return either the location
the file exists locally or the location it would exist when opened
using the `open` method.

Parameters
----------
path : str
Can be a local file or a remote URL.

Returns
-------
out : str
Complete path, including the `DataSource` destination directory.

Notes
-----
The functionality is based on `os.path.abspath`.

**`exists(self, path)`**

Test if path exists.

Test if `path` exists as (and in this order):

- a local file.
- a remote URL that has been downloaded and stored locally in the
`DataSource` directory.
- a remote URL that has not been downloaded, but is valid and
accessible.

Parameters
----------
path : str
Can be a local file or a remote URL.

Returns
-------
out : bool
True if `path` exists.

Notes
-----
When `path` is an URL, `exists` will return True if it's either
stored locally in the `DataSource` directory, or is a valid remote
URL.  `DataSource` does not discriminate between the two, the file
is accessible if it exists in either location.

**`open(self, path, mode='r', encoding=None, newline=None)`**

Open and return file-like object.

If `path` is an URL, it will be downloaded, stored in the
`DataSource` directory and opened from there.

Parameters
----------
path : str
Local file path or URL to open.
mode : {'r', 'w', 'a'}, optional
Mode to open `path`.  Mode 'r' for reading, 'w' for writing,
'a' to append. Available modes depend on the type of object
specified by `path`. Default is 'r'.
encoding : {None, str}, optional
Open text file with given encoding. The default encoding will be
what `io.open` uses.
newline : {None, str}, optional
Newline to use when reading text file.

Returns
-------
out : file object
File object.

### RankWarning
Module: `numpy`

Issued by `polyfit` when the Vandermonde matrix is rank deficient.

For more information, a way to suppress the warning, and an example of
`RankWarning` being issued, see `polyfit`.

### bool_
Module: `numpy`

Boolean type (True or False), stored as a byte.

.. warning::

   The :class:`bool_` type is not a subclass of the :class:`int_` type
   (the :class:`bool_` is not even a number type). This is different
   than Python's default implementation of :class:`bool` as a
   sub-class of :class:`int`.

:Character code: ``'?'``

### broadcast
Module: `numpy`

Produce an object that mimics broadcasting.

Parameters
----------
in1, in2, ... : array_like
    Input parameters.

Returns
-------
b : broadcast object
    Broadcast the input parameters against one another, and
    return an object that encapsulates the result.
    Amongst others, it has ``shape`` and ``nd`` properties, and
    may be used as an iterator.

See Also
--------
broadcast_arrays
broadcast_to
broadcast_shapes

Examples
--------

Manually adding two vectors, using broadcasting:

>>> x = np.array([[1], [2], [3]])
>>> y = np.array([4, 5, 6])
>>> b = np.broadcast(x, y)

>>> out = np.empty(b.shape)
>>> out.flat = [u+v for (u,v) in b]
>>> out
array([[5.,  6.,  7.],
       [6.,  7.,  8.],
       [7.,  8.,  9.]])

Compare against built-in broadcasting:

>>> x + y
array([[5, 6, 7],
       [6, 7, 8],
       [7, 8, 9]])

#### Methods

**`reset(...)`**

reset()

Reset the broadcasted result's iterator(s).

Parameters
----------
None

Returns
-------
None

Examples
--------
>>> x = np.array([1, 2, 3])
>>> y = np.array([[4], [5], [6]])
>>> b = np.broadcast(x, y)
>>> b.index
0
>>> next(b), next(b), next(b)
((1, 4), (2, 4), (3, 4))
>>> b.index
3
>>> b.reset()
>>> b.index
0

### busdaycalendar
Module: `numpy`

busdaycalendar(weekmask='1111100', holidays=None)

A business day calendar object that efficiently stores information
defining valid days for the busday family of functions.

The default valid days are Monday through Friday ("business days").
A busdaycalendar object can be specified with any set of weekly
valid days, plus an optional "holiday" dates that always will be invalid.

Once a busdaycalendar object is created, the weekmask and holidays
cannot be modified.

.. versionadded:: 1.7.0

Parameters
----------
weekmask : str or array_like of bool, optional
    A seven-element array indicating which of Monday through Sunday are
    valid days. May be specified as a length-seven list or array, like
    [1,1,1,1,1,0,0]; a length-seven string, like '1111100'; or a string
    like "Mon Tue Wed Thu Fri", made up of 3-character abbreviations for
    weekdays, optionally separated by white space. Valid abbreviations
    are: Mon Tue Wed Thu Fri Sat Sun
holidays : array_like of datetime64[D], optional
    An array of dates to consider as invalid dates, no matter which
    weekday they fall upon.  Holiday dates may be specified in any
    order, and NaT (not-a-time) dates are ignored.  This list is
    saved in a normalized form that is suited for fast calculations
    of valid days.

Returns
-------
out : busdaycalendar
    A business day calendar object containing the specified
    weekmask and holidays values.

See Also
--------
is_busday : Returns a boolean array indicating valid days.
busday_offset : Applies an offset counted in valid days.
busday_count : Counts how many valid days are in a half-open date range.

Attributes
----------
Note: once a busdaycalendar object is created, you cannot modify the
weekmask or holidays.  The attributes return copies of internal data.
weekmask : (copy) seven-element array of bool
holidays : (copy) sorted array of datetime64[D]

Examples
--------
>>> # Some important days in July
... bdd = np.busdaycalendar(
...             holidays=['2011-07-01', '2011-07-04', '2011-07-17'])
>>> # Default is Monday to Friday weekdays
... bdd.weekmask
array([ True,  True,  True,  True,  True, False, False])
>>> # Any holidays already on the weekend are removed
... bdd.holidays
array(['2011-07-01', '2011-07-04'], dtype='datetime64[D]')

### byte
Module: `numpy`

Signed integer type, compatible with C ``char``.

:Character code: ``'b'``
:Canonical name: `numpy.byte`
:Alias on this platform (win32 AMD64): `numpy.int8`: 8-bit signed integer (``-128`` to ``127``).

#### Methods

**`bit_count(...)`**

int8.bit_count() -> int

Computes the number of 1-bits in the absolute value of the input.
Analogous to the builtin `int.bit_count` or ``popcount`` in C++.

Examples
--------
>>> np.int8(127).bit_count()
7
>>> np.int8(-127).bit_count()
7

### bytes_
Module: `numpy`

A byte string.

When used in arrays, this type strips trailing null bytes.

:Character code: ``'S'``
:Alias: `numpy.string_`

### cdouble
Module: `numpy`

Complex number type composed of two double-precision floating-point
numbers, compatible with Python `complex`.

:Character code: ``'D'``
:Canonical name: `numpy.cdouble`
:Alias: `numpy.cfloat`
:Alias: `numpy.complex_`
:Alias on this platform (win32 AMD64): `numpy.complex128`: Complex number type composed of 2 64-bit-precision floating-point numbers.

### cfloat
Module: `numpy`

Complex number type composed of two double-precision floating-point
numbers, compatible with Python `complex`.

:Character code: ``'D'``
:Canonical name: `numpy.cdouble`
:Alias: `numpy.cfloat`
:Alias: `numpy.complex_`
:Alias on this platform (win32 AMD64): `numpy.complex128`: Complex number type composed of 2 64-bit-precision floating-point numbers.

### character
Module: `numpy`

Abstract base class of all character string scalar types.

### chararray
Module: `numpy`

chararray(shape, itemsize=1, unicode=False, buffer=None, offset=0,
          strides=None, order=None)

Provides a convenient view on arrays of string and unicode values.

.. note::
   The `chararray` class exists for backwards compatibility with
   Numarray, it is not recommended for new development. Starting from numpy
   1.4, if one needs arrays of strings, it is recommended to use arrays of
   `dtype` `object_`, `bytes_` or `str_`, and use the free functions
   in the `numpy.char` module for fast vectorized string operations.

Versus a regular NumPy array of type `str` or `unicode`, this
class adds the following functionality:

  1) values automatically have whitespace removed from the end
     when indexed

  2) comparison operators automatically remove whitespace from the
     end when comparing values

  3) vectorized string operations are provided as methods
     (e.g. `.endswith`) and infix operators (e.g. ``"+", "*", "%"``)

chararrays should be created using `numpy.char.array` or
`numpy.char.asarray`, rather than this constructor directly.

This constructor creates the array, using `buffer` (with `offset`
and `strides`) if it is not ``None``. If `buffer` is ``None``, then
constructs a new array with `strides` in "C order", unless both
``len(shape) >= 2`` and ``order='F'``, in which case `strides`
is in "Fortran order".

Methods
-------
astype
argsort
copy
count
decode
dump
dumps
encode
endswith
expandtabs
fill
find
flatten
getfield
index
isalnum
isalpha
isdecimal
isdigit
islower
isnumeric
isspace
istitle
isupper
item
join
ljust
lower
lstrip
nonzero
put
ravel
repeat
replace
reshape
resize
rfind
rindex
rjust
rsplit
rstrip
searchsorted
setfield
setflags
sort
split
splitlines
squeeze
startswith
strip
swapaxes
swapcase
take
title
tofile
tolist
tostring
translate
transpose
upper
view
zfill

Parameters
----------
shape : tuple
    Shape of the array.
itemsize : int, optional
    Length of each array element, in number of characters. Default is 1.
unicode : bool, optional
    Are the array elements of type unicode (True) or string (False).
    Default is False.
buffer : object exposing the buffer interface or str, optional
    Memory address of the start of the array data.  Default is None,
    in which case a new array is created.
offset : int, optional
    Fixed stride displacement from the beginning of an axis?
    Default is 0. Needs to be >=0.
strides : array_like of ints, optional
    Strides for the array (see `ndarray.strides` for full description).
    Default is None.
order : {'C', 'F'}, optional
    The order in which the array data is stored in memory: 'C' ->
    "row major" order (the default), 'F' -> "column major"
    (Fortran) order.

Examples
--------
>>> charar = np.chararray((3, 3))
>>> charar[:] = 'a'
>>> charar
chararray([[b'a', b'a', b'a'],
           [b'a', b'a', b'a'],
           [b'a', b'a', b'a']], dtype='|S1')

>>> charar = np.chararray(charar.shape, itemsize=5)
>>> charar[:] = 'abc'
>>> charar
chararray([[b'abc', b'abc', b'abc'],
           [b'abc', b'abc', b'abc'],
           [b'abc', b'abc', b'abc']], dtype='|S5')

#### Methods

**`argsort(self, axis=-1, kind=None, order=None)`**

a.argsort(axis=-1, kind=None, order=None)

Returns the indices that would sort this array.

Refer to `numpy.argsort` for full documentation.

See Also
--------
numpy.argsort : equivalent function

**`capitalize(self)`**

Return a copy of `self` with only the first character of each element
capitalized.

See Also
--------
char.capitalize

**`center(self, width, fillchar=' ')`**

Return a copy of `self` with its elements centered in a
string of length `width`.

See Also
--------
center

**`count(self, sub, start=0, end=None)`**

Returns an array with the number of non-overlapping occurrences of
substring `sub` in the range [`start`, `end`].

See Also
--------
char.count

**`decode(self, encoding=None, errors=None)`**

Calls ``bytes.decode`` element-wise.

See Also
--------
char.decode

**`encode(self, encoding=None, errors=None)`**

Calls `str.encode` element-wise.

See Also
--------
char.encode

**`endswith(self, suffix, start=0, end=None)`**

Returns a boolean array which is `True` where the string element
in `self` ends with `suffix`, otherwise `False`.

See Also
--------
char.endswith

**`expandtabs(self, tabsize=8)`**

Return a copy of each string element where all tab characters are
replaced by one or more spaces.

See Also
--------
char.expandtabs

**`find(self, sub, start=0, end=None)`**

For each element, return the lowest index in the string where
substring `sub` is found.

See Also
--------
char.find

**`index(self, sub, start=0, end=None)`**

Like `find`, but raises `ValueError` when the substring is not found.

See Also
--------
char.index

**`isalnum(self)`**

Returns true for each element if all characters in the string
are alphanumeric and there is at least one character, false
otherwise.

See Also
--------
char.isalnum

**`isalpha(self)`**

Returns true for each element if all characters in the string
are alphabetic and there is at least one character, false
otherwise.

See Also
--------
char.isalpha

**`isdigit(self)`**

Returns true for each element if all characters in the string are
digits and there is at least one character, false otherwise.

See Also
--------
char.isdigit

**`islower(self)`**

Returns true for each element if all cased characters in the
string are lowercase and there is at least one cased character,
false otherwise.

See Also
--------
char.islower

**`isspace(self)`**

Returns true for each element if there are only whitespace
characters in the string and there is at least one character,
false otherwise.

See Also
--------
char.isspace

**`istitle(self)`**

Returns true for each element if the element is a titlecased
string and there is at least one character, false otherwise.

See Also
--------
char.istitle

**`isupper(self)`**

Returns true for each element if all cased characters in the
string are uppercase and there is at least one character, false
otherwise.

See Also
--------
char.isupper

**`join(self, seq)`**

Return a string which is the concatenation of the strings in the
sequence `seq`.

See Also
--------
char.join

**`ljust(self, width, fillchar=' ')`**

Return an array with the elements of `self` left-justified in a
string of length `width`.

See Also
--------
char.ljust

**`lower(self)`**

Return an array with the elements of `self` converted to
lowercase.

See Also
--------
char.lower

**`lstrip(self, chars=None)`**

For each element in `self`, return a copy with the leading characters
removed.

See Also
--------
char.lstrip

**`partition(self, sep)`**

Partition each element in `self` around `sep`.

See Also
--------
partition

**`replace(self, old, new, count=None)`**

For each element in `self`, return a copy of the string with all
occurrences of substring `old` replaced by `new`.

See Also
--------
char.replace

**`rfind(self, sub, start=0, end=None)`**

For each element in `self`, return the highest index in the string
where substring `sub` is found, such that `sub` is contained
within [`start`, `end`].

See Also
--------
char.rfind

**`rindex(self, sub, start=0, end=None)`**

Like `rfind`, but raises `ValueError` when the substring `sub` is
not found.

See Also
--------
char.rindex

**`rjust(self, width, fillchar=' ')`**

Return an array with the elements of `self`
right-justified in a string of length `width`.

See Also
--------
char.rjust

**`rpartition(self, sep)`**

Partition each element in `self` around `sep`.

See Also
--------
rpartition

**`rsplit(self, sep=None, maxsplit=None)`**

For each element in `self`, return a list of the words in
the string, using `sep` as the delimiter string.

See Also
--------
char.rsplit

**`rstrip(self, chars=None)`**

For each element in `self`, return a copy with the trailing
characters removed.

See Also
--------
char.rstrip

**`split(self, sep=None, maxsplit=None)`**

For each element in `self`, return a list of the words in the
string, using `sep` as the delimiter string.

See Also
--------
char.split

**`splitlines(self, keepends=None)`**

For each element in `self`, return a list of the lines in the
element, breaking at line boundaries.

See Also
--------
char.splitlines

**`startswith(self, prefix, start=0, end=None)`**

Returns a boolean array which is `True` where the string element
in `self` starts with `prefix`, otherwise `False`.

See Also
--------
char.startswith

**`strip(self, chars=None)`**

For each element in `self`, return a copy with the leading and
trailing characters removed.

See Also
--------
char.strip

**`swapcase(self)`**

For each element in `self`, return a copy of the string with
uppercase characters converted to lowercase and vice versa.

See Also
--------
char.swapcase

**`title(self)`**

For each element in `self`, return a titlecased version of the
string: words start with uppercase characters, all remaining cased
characters are lowercase.

See Also
--------
char.title

**`translate(self, table, deletechars=None)`**

For each element in `self`, return a copy of the string where
all characters occurring in the optional argument
`deletechars` are removed, and the remaining characters have
been mapped through the given translation table.

See Also
--------
char.translate

**`upper(self)`**

Return an array with the elements of `self` converted to
uppercase.

See Also
--------
char.upper

**`zfill(self, width)`**

Return the numeric string left-filled with zeros in a string of
length `width`.

See Also
--------
char.zfill

**`isnumeric(self)`**

For each element in `self`, return True if there are only
numeric characters in the element.

See Also
--------
char.isnumeric

**`isdecimal(self)`**

For each element in `self`, return True if there are only
decimal characters in the element.

See Also
--------
char.isdecimal

### clongdouble
Module: `numpy`

Complex number type composed of two extended-precision floating-point
numbers.

:Character code: ``'G'``
:Alias: `numpy.clongfloat`
:Alias: `numpy.longcomplex`

### clongfloat
Module: `numpy`

Complex number type composed of two extended-precision floating-point
numbers.

:Character code: ``'G'``
:Alias: `numpy.clongfloat`
:Alias: `numpy.longcomplex`

### complex128
Module: `numpy`

Complex number type composed of two double-precision floating-point
numbers, compatible with Python `complex`.

:Character code: ``'D'``
:Canonical name: `numpy.cdouble`
:Alias: `numpy.cfloat`
:Alias: `numpy.complex_`
:Alias on this platform (win32 AMD64): `numpy.complex128`: Complex number type composed of 2 64-bit-precision floating-point numbers.

### complex64
Module: `numpy`

Complex number type composed of two single-precision floating-point
numbers.

:Character code: ``'F'``
:Canonical name: `numpy.csingle`
:Alias: `numpy.singlecomplex`
:Alias on this platform (win32 AMD64): `numpy.complex64`: Complex number type composed of 2 32-bit-precision floating-point numbers.

### complex_
Module: `numpy`

Complex number type composed of two double-precision floating-point
numbers, compatible with Python `complex`.

:Character code: ``'D'``
:Canonical name: `numpy.cdouble`
:Alias: `numpy.cfloat`
:Alias: `numpy.complex_`
:Alias on this platform (win32 AMD64): `numpy.complex128`: Complex number type composed of 2 64-bit-precision floating-point numbers.

### complexfloating
Module: `numpy`

Abstract base class of all complex number scalar types that are made up of
floating-point numbers.

### csingle
Module: `numpy`

Complex number type composed of two single-precision floating-point
numbers.

:Character code: ``'F'``
:Canonical name: `numpy.csingle`
:Alias: `numpy.singlecomplex`
:Alias on this platform (win32 AMD64): `numpy.complex64`: Complex number type composed of 2 32-bit-precision floating-point numbers.

### datetime64
Module: `numpy`

If created from a 64-bit integer, it represents an offset from
``1970-01-01T00:00:00``.
If created from string, the string can be in ISO 8601 date
or datetime format.

>>> np.datetime64(10, 'Y')
numpy.datetime64('1980')
>>> np.datetime64('1980', 'Y')
numpy.datetime64('1980')
>>> np.datetime64(10, 'D')
numpy.datetime64('1970-01-11')

See :ref:`arrays.datetime` for more information.

:Character code: ``'M'``

### double
Module: `numpy`

Double-precision floating-point number type, compatible with Python `float`
and C ``double``.

:Character code: ``'d'``
:Canonical name: `numpy.double`
:Alias: `numpy.float_`
:Alias on this platform (win32 AMD64): `numpy.float64`: 64-bit precision floating-point number type: sign bit, 11 bits exponent, 52 bits mantissa.

#### Methods

**`as_integer_ratio(...)`**

double.as_integer_ratio() -> (int, int)

Return a pair of integers, whose ratio is exactly equal to the original
floating point number, and with a positive denominator.
Raise `OverflowError` on infinities and a `ValueError` on NaNs.

>>> np.double(10.0).as_integer_ratio()
(10, 1)
>>> np.double(0.0).as_integer_ratio()
(0, 1)
>>> np.double(-.25).as_integer_ratio()
(-1, 4)

**`is_integer(...)`**

double.is_integer() -> bool

Return ``True`` if the floating point number is finite with integral
value, and ``False`` otherwise.

.. versionadded:: 1.22

Examples
--------
>>> np.double(-2.0).is_integer()
True
>>> np.double(3.2).is_integer()
False

### dtype
Module: `numpy`

dtype(dtype, align=False, copy=False, [metadata])

Create a data type object.

A numpy array is homogeneous, and contains elements described by a
dtype object. A dtype object can be constructed from different
combinations of fundamental numeric types.

Parameters
----------
dtype
    Object to be converted to a data type object.
align : bool, optional
    Add padding to the fields to match what a C compiler would output
    for a similar C-struct. Can be ``True`` only if `obj` is a dictionary
    or a comma-separated string. If a struct dtype is being created,
    this also sets a sticky alignment flag ``isalignedstruct``.
copy : bool, optional
    Make a new copy of the data-type object. If ``False``, the result
    may just be a reference to a built-in data-type object.
metadata : dict, optional
    An optional dictionary with dtype metadata.

See also
--------
result_type

Examples
--------
Using array-scalar type:

>>> np.dtype(np.int16)
dtype('int16')

Structured type, one field name 'f1', containing int16:

>>> np.dtype([('f1', np.int16)])
dtype([('f1', '<i2')])

Structured type, one field named 'f1', in itself containing a structured
type with one field:

>>> np.dtype([('f1', [('f1', np.int16)])])
dtype([('f1', [('f1', '<i2')])])

Structured type, two fields: the first field contains an unsigned int, the
second an int32:

>>> np.dtype([('f1', np.uint64), ('f2', np.int32)])
dtype([('f1', '<u8'), ('f2', '<i4')])

Using array-protocol type strings:

>>> np.dtype([('a','f8'),('b','S10')])
dtype([('a', '<f8'), ('b', 'S10')])

Using comma-separated field formats.  The shape is (2,3):

>>> np.dtype("i4, (2,3)f8")
dtype([('f0', '<i4'), ('f1', '<f8', (2, 3))])

Using tuples.  ``int`` is a fixed type, 3 the field's shape.  ``void``
is a flexible type, here of size 10:

>>> np.dtype([('hello',(np.int64,3)),('world',np.void,10)])
dtype([('hello', '<i8', (3,)), ('world', 'V10')])

Subdivide ``int16`` into 2 ``int8``'s, called x and y.  0 and 1 are
the offsets in bytes:

>>> np.dtype((np.int16, {'x':(np.int8,0), 'y':(np.int8,1)}))
dtype((numpy.int16, [('x', 'i1'), ('y', 'i1')]))

Using dictionaries.  Two fields named 'gender' and 'age':

>>> np.dtype({'names':['gender','age'], 'formats':['S1',np.uint8]})
dtype([('gender', 'S1'), ('age', 'u1')])

Offsets in bytes, here 0 and 25:

>>> np.dtype({'surname':('S25',0),'age':(np.uint8,25)})
dtype([('surname', 'S25'), ('age', 'u1')])

#### Methods

**`newbyteorder(...)`**

newbyteorder(new_order='S', /)

Return a new dtype with a different byte order.

Changes are also made in all fields and sub-arrays of the data type.

Parameters
----------
new_order : string, optional
Byte order to force; a value from the byte order specifications
below.  The default value ('S') results in swapping the current
byte order.  `new_order` codes can be any of:

* 'S' - swap dtype from current to opposite endian
* {'<', 'little'} - little endian
* {'>', 'big'} - big endian
* {'=', 'native'} - native order
* {'|', 'I'} - ignore (no change to byte order)

Returns
-------
new_dtype : dtype
New dtype object with the given change to the byte order.

Notes
-----
Changes are also made in all fields and sub-arrays of the data type.

Examples
--------
>>> import sys
>>> sys_is_le = sys.byteorder == 'little'
>>> native_code = '<' if sys_is_le else '>'
>>> swapped_code = '>' if sys_is_le else '<'
>>> native_dt = np.dtype(native_code+'i2')
>>> swapped_dt = np.dtype(swapped_code+'i2')
>>> native_dt.newbyteorder('S') == swapped_dt
True
>>> native_dt.newbyteorder() == swapped_dt
True
>>> native_dt == swapped_dt.newbyteorder('S')
True
>>> native_dt == swapped_dt.newbyteorder('=')
True
>>> native_dt == swapped_dt.newbyteorder('N')
True
>>> native_dt == native_dt.newbyteorder('|')
True
>>> np.dtype('<i2') == native_dt.newbyteorder('<')
True
>>> np.dtype('<i2') == native_dt.newbyteorder('L')
True
>>> np.dtype('>i2') == native_dt.newbyteorder('>')
True
>>> np.dtype('>i2') == native_dt.newbyteorder('B')
True

### errstate
Module: `numpy`

errstate(**kwargs)

Context manager for floating-point error handling.

Using an instance of `errstate` as a context manager allows statements in
that context to execute with a known error handling behavior. Upon entering
the context the error handling is set with `seterr` and `seterrcall`, and
upon exiting it is reset to what it was before.

..  versionchanged:: 1.17.0
    `errstate` is also usable as a function decorator, saving
    a level of indentation if an entire function is wrapped.
    See :py:class:`contextlib.ContextDecorator` for more information.

Parameters
----------
kwargs : {divide, over, under, invalid}
    Keyword arguments. The valid keywords are the possible floating-point
    exceptions. Each keyword should have a string value that defines the
    treatment for the particular error. Possible values are
    {'ignore', 'warn', 'raise', 'call', 'print', 'log'}.

See Also
--------
seterr, geterr, seterrcall, geterrcall

Notes
-----
For complete documentation of the types of floating-point exceptions and
treatment options, see `seterr`.

Examples
--------
>>> olderr = np.seterr(all='ignore')  # Set error handling to known state.

>>> np.arange(3) / 0.
array([nan, inf, inf])
>>> with np.errstate(divide='warn'):
...     np.arange(3) / 0.
array([nan, inf, inf])

>>> np.sqrt(-1)
nan
>>> with np.errstate(invalid='raise'):
...     np.sqrt(-1)
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
FloatingPointError: invalid value encountered in sqrt

Outside the context the error handling behavior has not changed:

>>> np.geterr()
{'divide': 'ignore', 'over': 'ignore', 'under': 'ignore', 'invalid': 'ignore'}

### finfo
Module: `numpy`

finfo(dtype)

Machine limits for floating point types.

Attributes
----------
bits : int
    The number of bits occupied by the type.
dtype : dtype
    Returns the dtype for which `finfo` returns information. For complex
    input, the returned dtype is the associated ``float*`` dtype for its
    real and complex components.
eps : float
    The difference between 1.0 and the next smallest representable float
    larger than 1.0. For example, for 64-bit binary floats in the IEEE-754
    standard, ``eps = 2**-52``, approximately 2.22e-16.
epsneg : float
    The difference between 1.0 and the next smallest representable float
    less than 1.0. For example, for 64-bit binary floats in the IEEE-754
    standard, ``epsneg = 2**-53``, approximately 1.11e-16.
iexp : int
    The number of bits in the exponent portion of the floating point
    representation.
machep : int
    The exponent that yields `eps`.
max : floating point number of the appropriate type
    The largest representable number.
maxexp : int
    The smallest positive power of the base (2) that causes overflow.
min : floating point number of the appropriate type
    The smallest representable number, typically ``-max``.
minexp : int
    The most negative power of the base (2) consistent with there
    being no leading 0's in the mantissa.
negep : int
    The exponent that yields `epsneg`.
nexp : int
    The number of bits in the exponent including its sign and bias.
nmant : int
    The number of bits in the mantissa.
precision : int
    The approximate number of decimal digits to which this kind of
    float is precise.
resolution : floating point number of the appropriate type
    The approximate decimal resolution of this type, i.e.,
    ``10**-precision``.
tiny : float
    An alias for `smallest_normal`, kept for backwards compatibility.
smallest_normal : float
    The smallest positive floating point number with 1 as leading bit in
    the mantissa following IEEE-754 (see Notes).
smallest_subnormal : float
    The smallest positive floating point number with 0 as leading bit in
    the mantissa following IEEE-754.

Parameters
----------
dtype : float, dtype, or instance
    Kind of floating point or complex floating point
    data-type about which to get information.

See Also
--------
iinfo : The equivalent for integer data types.
spacing : The distance between a value and the nearest adjacent number
nextafter : The next floating point value after x1 towards x2

Notes
-----
For developers of NumPy: do not instantiate this at the module level.
The initial calculation of these parameters is expensive and negatively
impacts import times.  These objects are cached, so calling ``finfo()``
repeatedly inside your functions is not a problem.

Note that ``smallest_normal`` is not actually the smallest positive
representable value in a NumPy floating point type. As in the IEEE-754
standard [1]_, NumPy floating point types make use of subnormal numbers to
fill the gap between 0 and ``smallest_normal``. However, subnormal numbers
may have significantly reduced precision [2]_.

This function can also be used for complex data types as well. If used,
the output will be the same as the corresponding real float type
(e.g. numpy.finfo(numpy.csingle) is the same as numpy.finfo(numpy.single)).
However, the output is true for the real and imaginary components.

References
----------
.. [1] IEEE Standard for Floating-Point Arithmetic, IEEE Std 754-2008,
       pp.1-70, 2008, http://www.doi.org/10.1109/IEEESTD.2008.4610935
.. [2] Wikipedia, "Denormal Numbers",
       https://en.wikipedia.org/wiki/Denormal_number

Examples
--------
>>> np.finfo(np.float64).dtype
dtype('float64')
>>> np.finfo(np.complex64).dtype
dtype('float32')

### flatiter
Module: `numpy`

Flat iterator object to iterate over arrays.

A `flatiter` iterator is returned by ``x.flat`` for any array `x`.
It allows iterating over the array as if it were a 1-D array,
either in a for-loop or by calling its `next` method.

Iteration is done in row-major, C-style order (the last
index varying the fastest). The iterator can also be indexed using
basic slicing or advanced indexing.

See Also
--------
ndarray.flat : Return a flat iterator over an array.
ndarray.flatten : Returns a flattened copy of an array.

Notes
-----
A `flatiter` iterator can not be constructed directly from Python code
by calling the `flatiter` constructor.

Examples
--------
>>> x = np.arange(6).reshape(2, 3)
>>> fl = x.flat
>>> type(fl)
<class 'numpy.flatiter'>
>>> for item in fl:
...     print(item)
...
0
1
2
3
4
5

>>> fl[2:4]
array([2, 3])

#### Methods

**`copy(...)`**

copy()

Get a copy of the iterator as a 1-D array.

Examples
--------
>>> x = np.arange(6).reshape(2, 3)
>>> x
array([[0, 1, 2],
[3, 4, 5]])
>>> fl = x.flat
>>> fl.copy()
array([0, 1, 2, 3, 4, 5])

### flexible
Module: `numpy`

Abstract base class of all scalar types without predefined length.
The actual size of these types depends on the specific `np.dtype`
instantiation.

### float16
Module: `numpy`

Half-precision floating-point number type.

:Character code: ``'e'``
:Canonical name: `numpy.half`
:Alias on this platform (win32 AMD64): `numpy.float16`: 16-bit-precision floating-point number type: sign bit, 5 bits exponent, 10 bits mantissa.

#### Methods

**`as_integer_ratio(...)`**

half.as_integer_ratio() -> (int, int)

Return a pair of integers, whose ratio is exactly equal to the original
floating point number, and with a positive denominator.
Raise `OverflowError` on infinities and a `ValueError` on NaNs.

>>> np.half(10.0).as_integer_ratio()
(10, 1)
>>> np.half(0.0).as_integer_ratio()
(0, 1)
>>> np.half(-.25).as_integer_ratio()
(-1, 4)

**`is_integer(...)`**

half.is_integer() -> bool

Return ``True`` if the floating point number is finite with integral
value, and ``False`` otherwise.

.. versionadded:: 1.22

Examples
--------
>>> np.half(-2.0).is_integer()
True
>>> np.half(3.2).is_integer()
False

### float32
Module: `numpy`

Single-precision floating-point number type, compatible with C ``float``.

:Character code: ``'f'``
:Canonical name: `numpy.single`
:Alias on this platform (win32 AMD64): `numpy.float32`: 32-bit-precision floating-point number type: sign bit, 8 bits exponent, 23 bits mantissa.

#### Methods

**`as_integer_ratio(...)`**

single.as_integer_ratio() -> (int, int)

Return a pair of integers, whose ratio is exactly equal to the original
floating point number, and with a positive denominator.
Raise `OverflowError` on infinities and a `ValueError` on NaNs.

>>> np.single(10.0).as_integer_ratio()
(10, 1)
>>> np.single(0.0).as_integer_ratio()
(0, 1)
>>> np.single(-.25).as_integer_ratio()
(-1, 4)

**`is_integer(...)`**

single.is_integer() -> bool

Return ``True`` if the floating point number is finite with integral
value, and ``False`` otherwise.

.. versionadded:: 1.22

Examples
--------
>>> np.single(-2.0).is_integer()
True
>>> np.single(3.2).is_integer()
False

### float64
Module: `numpy`

Double-precision floating-point number type, compatible with Python `float`
and C ``double``.

:Character code: ``'d'``
:Canonical name: `numpy.double`
:Alias: `numpy.float_`
:Alias on this platform (win32 AMD64): `numpy.float64`: 64-bit precision floating-point number type: sign bit, 11 bits exponent, 52 bits mantissa.

#### Methods

**`as_integer_ratio(...)`**

double.as_integer_ratio() -> (int, int)

Return a pair of integers, whose ratio is exactly equal to the original
floating point number, and with a positive denominator.
Raise `OverflowError` on infinities and a `ValueError` on NaNs.

>>> np.double(10.0).as_integer_ratio()
(10, 1)
>>> np.double(0.0).as_integer_ratio()
(0, 1)
>>> np.double(-.25).as_integer_ratio()
(-1, 4)

**`is_integer(...)`**

double.is_integer() -> bool

Return ``True`` if the floating point number is finite with integral
value, and ``False`` otherwise.

.. versionadded:: 1.22

Examples
--------
>>> np.double(-2.0).is_integer()
True
>>> np.double(3.2).is_integer()
False

### float_
Module: `numpy`

Double-precision floating-point number type, compatible with Python `float`
and C ``double``.

:Character code: ``'d'``
:Canonical name: `numpy.double`
:Alias: `numpy.float_`
:Alias on this platform (win32 AMD64): `numpy.float64`: 64-bit precision floating-point number type: sign bit, 11 bits exponent, 52 bits mantissa.

#### Methods

**`as_integer_ratio(...)`**

double.as_integer_ratio() -> (int, int)

Return a pair of integers, whose ratio is exactly equal to the original
floating point number, and with a positive denominator.
Raise `OverflowError` on infinities and a `ValueError` on NaNs.

>>> np.double(10.0).as_integer_ratio()
(10, 1)
>>> np.double(0.0).as_integer_ratio()
(0, 1)
>>> np.double(-.25).as_integer_ratio()
(-1, 4)

**`is_integer(...)`**

double.is_integer() -> bool

Return ``True`` if the floating point number is finite with integral
value, and ``False`` otherwise.

.. versionadded:: 1.22

Examples
--------
>>> np.double(-2.0).is_integer()
True
>>> np.double(3.2).is_integer()
False

### floating
Module: `numpy`

Abstract base class of all floating-point scalar types.

### format_parser
Module: `numpy`

Class to convert formats, names, titles description to a dtype.

After constructing the format_parser object, the dtype attribute is
the converted data-type:
``dtype = format_parser(formats, names, titles).dtype``

Attributes
----------
dtype : dtype
    The converted data-type.

Parameters
----------
formats : str or list of str
    The format description, either specified as a string with
    comma-separated format descriptions in the form ``'f8, i4, a5'``, or
    a list of format description strings  in the form
    ``['f8', 'i4', 'a5']``.
names : str or list/tuple of str
    The field names, either specified as a comma-separated string in the
    form ``'col1, col2, col3'``, or as a list or tuple of strings in the
    form ``['col1', 'col2', 'col3']``.
    An empty list can be used, in that case default field names
    ('f0', 'f1', ...) are used.
titles : sequence
    Sequence of title strings. An empty list can be used to leave titles
    out.
aligned : bool, optional
    If True, align the fields by padding as the C-compiler would.
    Default is False.
byteorder : str, optional
    If specified, all the fields will be changed to the
    provided byte-order.  Otherwise, the default byte-order is
    used. For all available string specifiers, see `dtype.newbyteorder`.

See Also
--------
dtype, typename, sctype2char

Examples
--------
>>> np.format_parser(['<f8', '<i4', '<a5'], ['col1', 'col2', 'col3'],
...                  ['T1', 'T2', 'T3']).dtype
dtype([(('T1', 'col1'), '<f8'), (('T2', 'col2'), '<i4'), (('T3', 'col3'), 'S5')])

`names` and/or `titles` can be empty lists. If `titles` is an empty list,
titles will simply not appear. If `names` is empty, default field names
will be used.

>>> np.format_parser(['f8', 'i4', 'a5'], ['col1', 'col2', 'col3'],
...                  []).dtype
dtype([('col1', '<f8'), ('col2', '<i4'), ('col3', '<S5')])
>>> np.format_parser(['<f8', '<i4', '<a5'], [], []).dtype
dtype([('f0', '<f8'), ('f1', '<i4'), ('f2', 'S5')])

### generic
Module: `numpy`

Base class for numpy scalar types.

Class from which most (all?) numpy scalar types are derived.  For
consistency, exposes the same API as `ndarray`, despite many
consequent attributes being either "get-only," or completely irrelevant.
This is the class from which it is strongly suggested users should derive
custom scalar types.

#### Methods

**`tolist(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.tolist`.

**`item(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.item`.

**`itemset(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.itemset`.

**`tobytes(...)`**

*No documentation available.*

**`tofile(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.tofile`.

**`tostring(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.tostring`.

**`byteswap(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.byteswap`.

**`astype(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.astype`.

**`getfield(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.getfield`.

**`setfield(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.setfield`.

**`copy(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.copy`.

**`resize(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.resize`.

**`dumps(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.dumps`.

**`dump(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.dump`.

**`fill(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.fill`.

**`transpose(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.transpose`.

**`take(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.take`.

**`put(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.put`.

**`repeat(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.repeat`.

**`choose(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.choose`.

**`sort(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.sort`.

**`argsort(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.argsort`.

**`searchsorted(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.searchsorted`.

**`argmax(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.argmax`.

**`argmin(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.argmin`.

**`reshape(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.reshape`.

**`squeeze(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.squeeze`.

**`view(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.view`.

**`swapaxes(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.swapaxes`.

**`max(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.max`.

**`min(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.min`.

**`ptp(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.ptp`.

**`mean(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.mean`.

**`trace(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.trace`.

**`diagonal(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.diagonal`.

**`clip(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.clip`.

**`conj(...)`**

*No documentation available.*

**`conjugate(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.conjugate`.

**`nonzero(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.nonzero`.

**`std(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.std`.

**`var(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.var`.

**`sum(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.sum`.

**`cumsum(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.cumsum`.

**`prod(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.prod`.

**`cumprod(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.cumprod`.

**`all(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.all`.

**`any(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.any`.

**`compress(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.compress`.

**`flatten(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.flatten`.

**`ravel(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.ravel`.

**`round(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.round`.

**`setflags(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.setflags`.

**`newbyteorder(...)`**

newbyteorder(new_order='S', /)

Return a new `dtype` with a different byte order.

Changes are also made in all fields and sub-arrays of the data type.

The `new_order` code can be any from the following:

* 'S' - swap dtype from current to opposite endian
* {'<', 'little'} - little endian
* {'>', 'big'} - big endian
* {'=', 'native'} - native order
* {'|', 'I'} - ignore (no change to byte order)

Parameters
----------
new_order : str, optional
Byte order to force; a value from the byte order specifications
above.  The default value ('S') results in swapping the current
byte order.


Returns
-------
new_dtype : dtype
New `dtype` object with the given change to the byte order.

### half
Module: `numpy`

Half-precision floating-point number type.

:Character code: ``'e'``
:Canonical name: `numpy.half`
:Alias on this platform (win32 AMD64): `numpy.float16`: 16-bit-precision floating-point number type: sign bit, 5 bits exponent, 10 bits mantissa.

#### Methods

**`as_integer_ratio(...)`**

half.as_integer_ratio() -> (int, int)

Return a pair of integers, whose ratio is exactly equal to the original
floating point number, and with a positive denominator.
Raise `OverflowError` on infinities and a `ValueError` on NaNs.

>>> np.half(10.0).as_integer_ratio()
(10, 1)
>>> np.half(0.0).as_integer_ratio()
(0, 1)
>>> np.half(-.25).as_integer_ratio()
(-1, 4)

**`is_integer(...)`**

half.is_integer() -> bool

Return ``True`` if the floating point number is finite with integral
value, and ``False`` otherwise.

.. versionadded:: 1.22

Examples
--------
>>> np.half(-2.0).is_integer()
True
>>> np.half(3.2).is_integer()
False

### iinfo
Module: `numpy`

iinfo(type)

Machine limits for integer types.

Attributes
----------
bits : int
    The number of bits occupied by the type.
dtype : dtype
    Returns the dtype for which `iinfo` returns information.
min : int
    The smallest integer expressible by the type.
max : int
    The largest integer expressible by the type.

Parameters
----------
int_type : integer type, dtype, or instance
    The kind of integer data type to get information about.

See Also
--------
finfo : The equivalent for floating point data types.

Examples
--------
With types:

>>> ii16 = np.iinfo(np.int16)
>>> ii16.min
-32768
>>> ii16.max
32767
>>> ii32 = np.iinfo(np.int32)
>>> ii32.min
-2147483648
>>> ii32.max
2147483647

With instances:

>>> ii32 = np.iinfo(np.int32(10))
>>> ii32.min
-2147483648
>>> ii32.max
2147483647

### inexact
Module: `numpy`

Abstract base class of all numeric scalar types with a (potentially)
inexact representation of the values in its range, such as
floating-point numbers.

### int16
Module: `numpy`

Signed integer type, compatible with C ``short``.

:Character code: ``'h'``
:Canonical name: `numpy.short`
:Alias on this platform (win32 AMD64): `numpy.int16`: 16-bit signed integer (``-32_768`` to ``32_767``).

#### Methods

**`bit_count(...)`**

int16.bit_count() -> int

Computes the number of 1-bits in the absolute value of the input.
Analogous to the builtin `int.bit_count` or ``popcount`` in C++.

Examples
--------
>>> np.int16(127).bit_count()
7
>>> np.int16(-127).bit_count()
7

### int32
Module: `numpy`

Signed integer type, compatible with Python `int` and C ``long``.

:Character code: ``'l'``
:Canonical name: `numpy.int_`
:Alias on this platform (win32 AMD64): `numpy.int32`: 32-bit signed integer (``-2_147_483_648`` to ``2_147_483_647``).

#### Methods

**`bit_count(...)`**

int32.bit_count() -> int

Computes the number of 1-bits in the absolute value of the input.
Analogous to the builtin `int.bit_count` or ``popcount`` in C++.

Examples
--------
>>> np.int32(127).bit_count()
7
>>> np.int32(-127).bit_count()
7

### int64
Module: `numpy`

Signed integer type, compatible with C ``long long``.

:Character code: ``'q'``
:Canonical name: `numpy.longlong`
:Alias on this platform (win32 AMD64): `numpy.int64`: 64-bit signed integer (``-9_223_372_036_854_775_808`` to ``9_223_372_036_854_775_807``).
:Alias on this platform (win32 AMD64): `numpy.intp`: Signed integer large enough to fit pointer, compatible with C ``intptr_t``.

#### Methods

**`bit_count(...)`**

int64.bit_count() -> int

Computes the number of 1-bits in the absolute value of the input.
Analogous to the builtin `int.bit_count` or ``popcount`` in C++.

Examples
--------
>>> np.int64(127).bit_count()
7
>>> np.int64(-127).bit_count()
7

### int8
Module: `numpy`

Signed integer type, compatible with C ``char``.

:Character code: ``'b'``
:Canonical name: `numpy.byte`
:Alias on this platform (win32 AMD64): `numpy.int8`: 8-bit signed integer (``-128`` to ``127``).

#### Methods

**`bit_count(...)`**

int8.bit_count() -> int

Computes the number of 1-bits in the absolute value of the input.
Analogous to the builtin `int.bit_count` or ``popcount`` in C++.

Examples
--------
>>> np.int8(127).bit_count()
7
>>> np.int8(-127).bit_count()
7

### int_
Module: `numpy`

Signed integer type, compatible with Python `int` and C ``long``.

:Character code: ``'l'``
:Canonical name: `numpy.int_`
:Alias on this platform (win32 AMD64): `numpy.int32`: 32-bit signed integer (``-2_147_483_648`` to ``2_147_483_647``).

#### Methods

**`bit_count(...)`**

int32.bit_count() -> int

Computes the number of 1-bits in the absolute value of the input.
Analogous to the builtin `int.bit_count` or ``popcount`` in C++.

Examples
--------
>>> np.int32(127).bit_count()
7
>>> np.int32(-127).bit_count()
7

### intc
Module: `numpy`

Signed integer type, compatible with C ``int``.

:Character code: ``'i'``

#### Methods

**`bit_count(...)`**

*No documentation available.*

### integer
Module: `numpy`

Abstract base class of all integer scalar types.

#### Methods

**`is_integer(...)`**

integer.is_integer() -> bool

Return ``True`` if the number is finite with integral value.

.. versionadded:: 1.22

Examples
--------
>>> np.int64(-2).is_integer()
True
>>> np.uint32(5).is_integer()
True

### intp
Module: `numpy`

Signed integer type, compatible with C ``long long``.

:Character code: ``'q'``
:Canonical name: `numpy.longlong`
:Alias on this platform (win32 AMD64): `numpy.int64`: 64-bit signed integer (``-9_223_372_036_854_775_808`` to ``9_223_372_036_854_775_807``).
:Alias on this platform (win32 AMD64): `numpy.intp`: Signed integer large enough to fit pointer, compatible with C ``intptr_t``.

#### Methods

**`bit_count(...)`**

int64.bit_count() -> int

Computes the number of 1-bits in the absolute value of the input.
Analogous to the builtin `int.bit_count` or ``popcount`` in C++.

Examples
--------
>>> np.int64(127).bit_count()
7
>>> np.int64(-127).bit_count()
7

### longcomplex
Module: `numpy`

Complex number type composed of two extended-precision floating-point
numbers.

:Character code: ``'G'``
:Alias: `numpy.clongfloat`
:Alias: `numpy.longcomplex`

### longdouble
Module: `numpy`

Extended-precision floating-point number type, compatible with C
``long double`` but not necessarily with IEEE 754 quadruple-precision.

:Character code: ``'g'``
:Alias: `numpy.longfloat`

#### Methods

**`as_integer_ratio(...)`**

longdouble.as_integer_ratio() -> (int, int)

Return a pair of integers, whose ratio is exactly equal to the original
floating point number, and with a positive denominator.
Raise `OverflowError` on infinities and a `ValueError` on NaNs.

>>> np.longdouble(10.0).as_integer_ratio()
(10, 1)
>>> np.longdouble(0.0).as_integer_ratio()
(0, 1)
>>> np.longdouble(-.25).as_integer_ratio()
(-1, 4)

**`is_integer(...)`**

longdouble.is_integer() -> bool

Return ``True`` if the floating point number is finite with integral
value, and ``False`` otherwise.

.. versionadded:: 1.22

Examples
--------
>>> np.longdouble(-2.0).is_integer()
True
>>> np.longdouble(3.2).is_integer()
False

### longfloat
Module: `numpy`

Extended-precision floating-point number type, compatible with C
``long double`` but not necessarily with IEEE 754 quadruple-precision.

:Character code: ``'g'``
:Alias: `numpy.longfloat`

#### Methods

**`as_integer_ratio(...)`**

longdouble.as_integer_ratio() -> (int, int)

Return a pair of integers, whose ratio is exactly equal to the original
floating point number, and with a positive denominator.
Raise `OverflowError` on infinities and a `ValueError` on NaNs.

>>> np.longdouble(10.0).as_integer_ratio()
(10, 1)
>>> np.longdouble(0.0).as_integer_ratio()
(0, 1)
>>> np.longdouble(-.25).as_integer_ratio()
(-1, 4)

**`is_integer(...)`**

longdouble.is_integer() -> bool

Return ``True`` if the floating point number is finite with integral
value, and ``False`` otherwise.

.. versionadded:: 1.22

Examples
--------
>>> np.longdouble(-2.0).is_integer()
True
>>> np.longdouble(3.2).is_integer()
False

### longlong
Module: `numpy`

Signed integer type, compatible with C ``long long``.

:Character code: ``'q'``
:Canonical name: `numpy.longlong`
:Alias on this platform (win32 AMD64): `numpy.int64`: 64-bit signed integer (``-9_223_372_036_854_775_808`` to ``9_223_372_036_854_775_807``).
:Alias on this platform (win32 AMD64): `numpy.intp`: Signed integer large enough to fit pointer, compatible with C ``intptr_t``.

#### Methods

**`bit_count(...)`**

int64.bit_count() -> int

Computes the number of 1-bits in the absolute value of the input.
Analogous to the builtin `int.bit_count` or ``popcount`` in C++.

Examples
--------
>>> np.int64(127).bit_count()
7
>>> np.int64(-127).bit_count()
7

### matrix
Module: `numpy`

matrix(data, dtype=None, copy=True)

.. note:: It is no longer recommended to use this class, even for linear
          algebra. Instead use regular arrays. The class may be removed
          in the future.

Returns a matrix from an array-like object, or from a string of data.
A matrix is a specialized 2-D array that retains its 2-D nature
through operations.  It has certain special operators, such as ``*``
(matrix multiplication) and ``**`` (matrix power).

Parameters
----------
data : array_like or string
   If `data` is a string, it is interpreted as a matrix with commas
   or spaces separating columns, and semicolons separating rows.
dtype : data-type
   Data-type of the output matrix.
copy : bool
   If `data` is already an `ndarray`, then this flag determines
   whether the data is copied (the default), or whether a view is
   constructed.

See Also
--------
array

Examples
--------
>>> a = np.matrix('1 2; 3 4')
>>> a
matrix([[1, 2],
        [3, 4]])

>>> np.matrix([[1, 2], [3, 4]])
matrix([[1, 2],
        [3, 4]])

#### Methods

**`tolist(self)`**

Return the matrix as a (possibly nested) list.

See `ndarray.tolist` for full documentation.

See Also
--------
ndarray.tolist

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
[ 4,  5,  6,  7],
[ 8,  9, 10, 11]])
>>> x.tolist()
[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

**`sum(self, axis=None, dtype=None, out=None)`**

Returns the sum of the matrix elements, along the given axis.

Refer to `numpy.sum` for full documentation.

See Also
--------
numpy.sum

Notes
-----
This is the same as `ndarray.sum`, except that where an `ndarray` would
be returned, a `matrix` object is returned instead.

Examples
--------
>>> x = np.matrix([[1, 2], [4, 3]])
>>> x.sum()
10
>>> x.sum(axis=1)
matrix([[3],
[7]])
>>> x.sum(axis=1, dtype='float')
matrix([[3.],
[7.]])
>>> out = np.zeros((2, 1), dtype='float')
>>> x.sum(axis=1, dtype='float', out=np.asmatrix(out))
matrix([[3.],
[7.]])

**`squeeze(self, axis=None)`**

Return a possibly reshaped matrix.

Refer to `numpy.squeeze` for more documentation.

Parameters
----------
axis : None or int or tuple of ints, optional
Selects a subset of the axes of length one in the shape.
If an axis is selected with shape entry greater than one,
an error is raised.

Returns
-------
squeezed : matrix
The matrix, but as a (1, N) matrix if it had shape (N, 1).

See Also
--------
numpy.squeeze : related function

Notes
-----
If `m` has a single column then that column is returned
as the single row of a matrix.  Otherwise `m` is returned.
The returned matrix is always either `m` itself or a view into `m`.
Supplying an axis keyword argument will not affect the returned matrix
but it may cause an error to be raised.

Examples
--------
>>> c = np.matrix([[1], [2]])
>>> c
matrix([[1],
[2]])
>>> c.squeeze()
matrix([[1, 2]])
>>> r = c.T
>>> r
matrix([[1, 2]])
>>> r.squeeze()
matrix([[1, 2]])
>>> m = np.matrix([[1, 2], [3, 4]])
>>> m.squeeze()
matrix([[1, 2],
[3, 4]])

**`flatten(self, order='C')`**

Return a flattened copy of the matrix.

All `N` elements of the matrix are placed into a single row.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
'C' means to flatten in row-major (C-style) order. 'F' means to
flatten in column-major (Fortran-style) order. 'A' means to
flatten in column-major order if `m` is Fortran *contiguous* in
memory, row-major order otherwise. 'K' means to flatten `m` in
the order the elements occur in memory. The default is 'C'.

Returns
-------
y : matrix
A copy of the matrix, flattened to a `(1, N)` matrix where `N`
is the number of elements in the original matrix.

See Also
--------
ravel : Return a flattened array.
flat : A 1-D flat iterator over the matrix.

Examples
--------
>>> m = np.matrix([[1,2], [3,4]])
>>> m.flatten()
matrix([[1, 2, 3, 4]])
>>> m.flatten('F')
matrix([[1, 3, 2, 4]])

**`mean(self, axis=None, dtype=None, out=None)`**

Returns the average of the matrix elements along the given axis.

Refer to `numpy.mean` for full documentation.

See Also
--------
numpy.mean

Notes
-----
Same as `ndarray.mean` except that, where that returns an `ndarray`,
this returns a `matrix` object.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3, 4)))
>>> x
matrix([[ 0,  1,  2,  3],
[ 4,  5,  6,  7],
[ 8,  9, 10, 11]])
>>> x.mean()
5.5
>>> x.mean(0)
matrix([[4., 5., 6., 7.]])
>>> x.mean(1)
matrix([[ 1.5],
[ 5.5],
[ 9.5]])

**`std(self, axis=None, dtype=None, out=None, ddof=0)`**

Return the standard deviation of the array elements along the given axis.

Refer to `numpy.std` for full documentation.

See Also
--------
numpy.std

Notes
-----
This is the same as `ndarray.std`, except that where an `ndarray` would
be returned, a `matrix` object is returned instead.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3, 4)))
>>> x
matrix([[ 0,  1,  2,  3],
[ 4,  5,  6,  7],
[ 8,  9, 10, 11]])
>>> x.std()
3.4520525295346629 # may vary
>>> x.std(0)
matrix([[ 3.26598632,  3.26598632,  3.26598632,  3.26598632]]) # may vary
>>> x.std(1)
matrix([[ 1.11803399],
[ 1.11803399],
[ 1.11803399]])

**`var(self, axis=None, dtype=None, out=None, ddof=0)`**

Returns the variance of the matrix elements, along the given axis.

Refer to `numpy.var` for full documentation.

See Also
--------
numpy.var

Notes
-----
This is the same as `ndarray.var`, except that where an `ndarray` would
be returned, a `matrix` object is returned instead.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3, 4)))
>>> x
matrix([[ 0,  1,  2,  3],
[ 4,  5,  6,  7],
[ 8,  9, 10, 11]])
>>> x.var()
11.916666666666666
>>> x.var(0)
matrix([[ 10.66666667,  10.66666667,  10.66666667,  10.66666667]]) # may vary
>>> x.var(1)
matrix([[1.25],
[1.25],
[1.25]])

**`prod(self, axis=None, dtype=None, out=None)`**

Return the product of the array elements over the given axis.

Refer to `prod` for full documentation.

See Also
--------
prod, ndarray.prod

Notes
-----
Same as `ndarray.prod`, except, where that returns an `ndarray`, this
returns a `matrix` object instead.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
[ 4,  5,  6,  7],
[ 8,  9, 10, 11]])
>>> x.prod()
0
>>> x.prod(0)
matrix([[  0,  45, 120, 231]])
>>> x.prod(1)
matrix([[   0],
[ 840],
[7920]])

**`any(self, axis=None, out=None)`**

Test whether any array element along a given axis evaluates to True.

Refer to `numpy.any` for full documentation.

Parameters
----------
axis : int, optional
Axis along which logical OR is performed
out : ndarray, optional
Output to existing array instead of creating new one, must have
same shape as expected output

Returns
-------
any : bool, ndarray
Returns a single bool if `axis` is ``None``; otherwise,
returns `ndarray`

**`all(self, axis=None, out=None)`**

Test whether all matrix elements along a given axis evaluate to True.

Parameters
----------
See `numpy.all` for complete descriptions

See Also
--------
numpy.all

Notes
-----
This is the same as `ndarray.all`, but it returns a `matrix` object.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
[ 4,  5,  6,  7],
[ 8,  9, 10, 11]])
>>> y = x[0]; y
matrix([[0, 1, 2, 3]])
>>> (x == y)
matrix([[ True,  True,  True,  True],
[False, False, False, False],
[False, False, False, False]])
>>> (x == y).all()
False
>>> (x == y).all(0)
matrix([[False, False, False, False]])
>>> (x == y).all(1)
matrix([[ True],
[False],
[False]])

**`max(self, axis=None, out=None)`**

Return the maximum value along an axis.

Parameters
----------
See `amax` for complete descriptions

See Also
--------
amax, ndarray.max

Notes
-----
This is the same as `ndarray.max`, but returns a `matrix` object
where `ndarray.max` would return an ndarray.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
[ 4,  5,  6,  7],
[ 8,  9, 10, 11]])
>>> x.max()
11
>>> x.max(0)
matrix([[ 8,  9, 10, 11]])
>>> x.max(1)
matrix([[ 3],
[ 7],
[11]])

**`argmax(self, axis=None, out=None)`**

Indexes of the maximum values along an axis.

Return the indexes of the first occurrences of the maximum values
along the specified axis.  If axis is None, the index is for the
flattened matrix.

Parameters
----------
See `numpy.argmax` for complete descriptions

See Also
--------
numpy.argmax

Notes
-----
This is the same as `ndarray.argmax`, but returns a `matrix` object
where `ndarray.argmax` would return an `ndarray`.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
[ 4,  5,  6,  7],
[ 8,  9, 10, 11]])
>>> x.argmax()
11
>>> x.argmax(0)
matrix([[2, 2, 2, 2]])
>>> x.argmax(1)
matrix([[3],
[3],
[3]])

**`min(self, axis=None, out=None)`**

Return the minimum value along an axis.

Parameters
----------
See `amin` for complete descriptions.

See Also
--------
amin, ndarray.min

Notes
-----
This is the same as `ndarray.min`, but returns a `matrix` object
where `ndarray.min` would return an ndarray.

Examples
--------
>>> x = -np.matrix(np.arange(12).reshape((3,4))); x
matrix([[  0,  -1,  -2,  -3],
[ -4,  -5,  -6,  -7],
[ -8,  -9, -10, -11]])
>>> x.min()
-11
>>> x.min(0)
matrix([[ -8,  -9, -10, -11]])
>>> x.min(1)
matrix([[ -3],
[ -7],
[-11]])

**`argmin(self, axis=None, out=None)`**

Indexes of the minimum values along an axis.

Return the indexes of the first occurrences of the minimum values
along the specified axis.  If axis is None, the index is for the
flattened matrix.

Parameters
----------
See `numpy.argmin` for complete descriptions.

See Also
--------
numpy.argmin

Notes
-----
This is the same as `ndarray.argmin`, but returns a `matrix` object
where `ndarray.argmin` would return an `ndarray`.

Examples
--------
>>> x = -np.matrix(np.arange(12).reshape((3,4))); x
matrix([[  0,  -1,  -2,  -3],
[ -4,  -5,  -6,  -7],
[ -8,  -9, -10, -11]])
>>> x.argmin()
11
>>> x.argmin(0)
matrix([[2, 2, 2, 2]])
>>> x.argmin(1)
matrix([[3],
[3],
[3]])

**`ptp(self, axis=None, out=None)`**

Peak-to-peak (maximum - minimum) value along the given axis.

Refer to `numpy.ptp` for full documentation.

See Also
--------
numpy.ptp

Notes
-----
Same as `ndarray.ptp`, except, where that would return an `ndarray` object,
this returns a `matrix` object.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
[ 4,  5,  6,  7],
[ 8,  9, 10, 11]])
>>> x.ptp()
11
>>> x.ptp(0)
matrix([[8, 8, 8, 8]])
>>> x.ptp(1)
matrix([[3],
[3],
[3]])

**`ravel(self, order='C')`**

Return a flattened matrix.

Refer to `numpy.ravel` for more documentation.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
The elements of `m` are read using this index order. 'C' means to
index the elements in C-like order, with the last axis index
changing fastest, back to the first axis index changing slowest.
'F' means to index the elements in Fortran-like index order, with
the first index changing fastest, and the last index changing
slowest. Note that the 'C' and 'F' options take no account of the
memory layout of the underlying array, and only refer to the order
of axis indexing.  'A' means to read the elements in Fortran-like
index order if `m` is Fortran *contiguous* in memory, C-like order
otherwise.  'K' means to read the elements in the order they occur
in memory, except for reversing the data when strides are negative.
By default, 'C' index order is used.

Returns
-------
ret : matrix
Return the matrix flattened to shape `(1, N)` where `N`
is the number of elements in the original matrix.
A copy is made only if necessary.

See Also
--------
matrix.flatten : returns a similar output matrix but always a copy
matrix.flat : a flat iterator on the array.
numpy.ravel : related function which returns an ndarray

**`getT(self)`**

Returns the transpose of the matrix.

Does *not* conjugate!  For the complex conjugate transpose, use ``.H``.

Parameters
----------
None

Returns
-------
ret : matrix object
The (non-conjugated) transpose of the matrix.

See Also
--------
transpose, getH

Examples
--------
>>> m = np.matrix('[1, 2; 3, 4]')
>>> m
matrix([[1, 2],
[3, 4]])
>>> m.getT()
matrix([[1, 3],
[2, 4]])

**`getA(self)`**

Return `self` as an `ndarray` object.

Equivalent to ``np.asarray(self)``.

Parameters
----------
None

Returns
-------
ret : ndarray
`self` as an `ndarray`

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
[ 4,  5,  6,  7],
[ 8,  9, 10, 11]])
>>> x.getA()
array([[ 0,  1,  2,  3],
[ 4,  5,  6,  7],
[ 8,  9, 10, 11]])

**`getA1(self)`**

Return `self` as a flattened `ndarray`.

Equivalent to ``np.asarray(x).ravel()``

Parameters
----------
None

Returns
-------
ret : ndarray
`self`, 1-D, as an `ndarray`

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
[ 4,  5,  6,  7],
[ 8,  9, 10, 11]])
>>> x.getA1()
array([ 0,  1,  2, ...,  9, 10, 11])

**`getH(self)`**

Returns the (complex) conjugate transpose of `self`.

Equivalent to ``np.transpose(self)`` if `self` is real-valued.

Parameters
----------
None

Returns
-------
ret : matrix object
complex conjugate transpose of `self`

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4)))
>>> z = x - 1j*x; z
matrix([[  0. +0.j,   1. -1.j,   2. -2.j,   3. -3.j],
[  4. -4.j,   5. -5.j,   6. -6.j,   7. -7.j],
[  8. -8.j,   9. -9.j,  10.-10.j,  11.-11.j]])
>>> z.getH()
matrix([[ 0. -0.j,  4. +4.j,  8. +8.j],
[ 1. +1.j,  5. +5.j,  9. +9.j],
[ 2. +2.j,  6. +6.j, 10.+10.j],
[ 3. +3.j,  7. +7.j, 11.+11.j]])

**`getI(self)`**

Returns the (multiplicative) inverse of invertible `self`.

Parameters
----------
None

Returns
-------
ret : matrix object
If `self` is non-singular, `ret` is such that ``ret * self`` ==
``self * ret`` == ``np.matrix(np.eye(self[0,:].size))`` all return
``True``.

Raises
------
numpy.linalg.LinAlgError: Singular matrix
If `self` is singular.

See Also
--------
linalg.inv

Examples
--------
>>> m = np.matrix('[1, 2; 3, 4]'); m
matrix([[1, 2],
[3, 4]])
>>> m.getI()
matrix([[-2. ,  1. ],
[ 1.5, -0.5]])
>>> m.getI() * m
matrix([[ 1.,  0.], # may vary
[ 0.,  1.]])

### memmap
Module: `numpy`

Create a memory-map to an array stored in a *binary* file on disk.

Memory-mapped files are used for accessing small segments of large files
on disk, without reading the entire file into memory.  NumPy's
memmap's are array-like objects.  This differs from Python's ``mmap``
module, which uses file-like objects.

This subclass of ndarray has some unpleasant interactions with
some operations, because it doesn't quite fit properly as a subclass.
An alternative to using this subclass is to create the ``mmap``
object yourself, then create an ndarray with ndarray.__new__ directly,
passing the object created in its 'buffer=' parameter.

This class may at some point be turned into a factory function
which returns a view into an mmap buffer.

Flush the memmap instance to write the changes to the file. Currently there
is no API to close the underlying ``mmap``. It is tricky to ensure the
resource is actually closed, since it may be shared between different
memmap instances.


Parameters
----------
filename : str, file-like object, or pathlib.Path instance
    The file name or file object to be used as the array data buffer.
dtype : data-type, optional
    The data-type used to interpret the file contents.
    Default is `uint8`.
mode : {'r+', 'r', 'w+', 'c'}, optional
    The file is opened in this mode:

    +------+-------------------------------------------------------------+
    | 'r'  | Open existing file for reading only.                        |
    +------+-------------------------------------------------------------+
    | 'r+' | Open existing file for reading and writing.                 |
    +------+-------------------------------------------------------------+
    | 'w+' | Create or overwrite existing file for reading and writing.  |
    |      | If ``mode == 'w+'`` then `shape` must also be specified.    |
    +------+-------------------------------------------------------------+
    | 'c'  | Copy-on-write: assignments affect data in memory, but       |
    |      | changes are not saved to disk.  The file on disk is         |
    |      | read-only.                                                  |
    +------+-------------------------------------------------------------+

    Default is 'r+'.
offset : int, optional
    In the file, array data starts at this offset. Since `offset` is
    measured in bytes, it should normally be a multiple of the byte-size
    of `dtype`. When ``mode != 'r'``, even positive offsets beyond end of
    file are valid; The file will be extended to accommodate the
    additional data. By default, ``memmap`` will start at the beginning of
    the file, even if ``filename`` is a file pointer ``fp`` and
    ``fp.tell() != 0``.
shape : tuple, optional
    The desired shape of the array. If ``mode == 'r'`` and the number
    of remaining bytes after `offset` is not a multiple of the byte-size
    of `dtype`, you must specify `shape`. By default, the returned array
    will be 1-D with the number of elements determined by file size
    and data-type.
order : {'C', 'F'}, optional
    Specify the order of the ndarray memory layout:
    :term:`row-major`, C-style or :term:`column-major`,
    Fortran-style.  This only has an effect if the shape is
    greater than 1-D.  The default order is 'C'.

Attributes
----------
filename : str or pathlib.Path instance
    Path to the mapped file.
offset : int
    Offset position in the file.
mode : str
    File mode.

Methods
-------
flush
    Flush any changes in memory to file on disk.
    When you delete a memmap object, flush is called first to write
    changes to disk.


See also
--------
lib.format.open_memmap : Create or load a memory-mapped ``.npy`` file.

Notes
-----
The memmap object can be used anywhere an ndarray is accepted.
Given a memmap ``fp``, ``isinstance(fp, numpy.ndarray)`` returns
``True``.

Memory-mapped files cannot be larger than 2GB on 32-bit systems.

When a memmap causes a file to be created or extended beyond its
current size in the filesystem, the contents of the new part are
unspecified. On systems with POSIX filesystem semantics, the extended
part will be filled with zero bytes.

Examples
--------
>>> data = np.arange(12, dtype='float32')
>>> data.resize((3,4))

This example uses a temporary file so that doctest doesn't write
files to your directory. You would use a 'normal' filename.

>>> from tempfile import mkdtemp
>>> import os.path as path
>>> filename = path.join(mkdtemp(), 'newfile.dat')

Create a memmap with dtype and shape that matches our data:

>>> fp = np.memmap(filename, dtype='float32', mode='w+', shape=(3,4))
>>> fp
memmap([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]], dtype=float32)

Write data to memmap array:

>>> fp[:] = data[:]
>>> fp
memmap([[  0.,   1.,   2.,   3.],
        [  4.,   5.,   6.,   7.],
        [  8.,   9.,  10.,  11.]], dtype=float32)

>>> fp.filename == path.abspath(filename)
True

Flushes memory changes to disk in order to read them back

>>> fp.flush()

Load the memmap and verify data was stored:

>>> newfp = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))
>>> newfp
memmap([[  0.,   1.,   2.,   3.],
        [  4.,   5.,   6.,   7.],
        [  8.,   9.,  10.,  11.]], dtype=float32)

Read-only memmap:

>>> fpr = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))
>>> fpr.flags.writeable
False

Copy-on-write memmap:

>>> fpc = np.memmap(filename, dtype='float32', mode='c', shape=(3,4))
>>> fpc.flags.writeable
True

It's possible to assign to copy-on-write array, but values are only
written into the memory copy of the array, and not written to disk:

>>> fpc
memmap([[  0.,   1.,   2.,   3.],
        [  4.,   5.,   6.,   7.],
        [  8.,   9.,  10.,  11.]], dtype=float32)
>>> fpc[0,:] = 0
>>> fpc
memmap([[  0.,   0.,   0.,   0.],
        [  4.,   5.,   6.,   7.],
        [  8.,   9.,  10.,  11.]], dtype=float32)

File on disk is unchanged:

>>> fpr
memmap([[  0.,   1.,   2.,   3.],
        [  4.,   5.,   6.,   7.],
        [  8.,   9.,  10.,  11.]], dtype=float32)

Offset into a memmap:

>>> fpo = np.memmap(filename, dtype='float32', mode='r', offset=16)
>>> fpo
memmap([  4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.], dtype=float32)

#### Methods

**`flush(self)`**

Write any changes in the array to the file on disk.

For further information, see `memmap`.

Parameters
----------
None

See Also
--------
memmap

### ndarray
Module: `numpy`

ndarray(shape, dtype=float, buffer=None, offset=0,
        strides=None, order=None)

An array object represents a multidimensional, homogeneous array
of fixed-size items.  An associated data-type object describes the
format of each element in the array (its byte-order, how many bytes it
occupies in memory, whether it is an integer, a floating point number,
or something else, etc.)

Arrays should be constructed using `array`, `zeros` or `empty` (refer
to the See Also section below).  The parameters given here refer to
a low-level method (`ndarray(...)`) for instantiating an array.

For more information, refer to the `numpy` module and examine the
methods and attributes of an array.

Parameters
----------
(for the __new__ method; see Notes below)

shape : tuple of ints
    Shape of created array.
dtype : data-type, optional
    Any object that can be interpreted as a numpy data type.
buffer : object exposing buffer interface, optional
    Used to fill the array with data.
offset : int, optional
    Offset of array data in buffer.
strides : tuple of ints, optional
    Strides of data in memory.
order : {'C', 'F'}, optional
    Row-major (C-style) or column-major (Fortran-style) order.

Attributes
----------
T : ndarray
    Transpose of the array.
data : buffer
    The array's elements, in memory.
dtype : dtype object
    Describes the format of the elements in the array.
flags : dict
    Dictionary containing information related to memory use, e.g.,
    'C_CONTIGUOUS', 'OWNDATA', 'WRITEABLE', etc.
flat : numpy.flatiter object
    Flattened version of the array as an iterator.  The iterator
    allows assignments, e.g., ``x.flat = 3`` (See `ndarray.flat` for
    assignment examples; TODO).
imag : ndarray
    Imaginary part of the array.
real : ndarray
    Real part of the array.
size : int
    Number of elements in the array.
itemsize : int
    The memory use of each array element in bytes.
nbytes : int
    The total number of bytes required to store the array data,
    i.e., ``itemsize * size``.
ndim : int
    The array's number of dimensions.
shape : tuple of ints
    Shape of the array.
strides : tuple of ints
    The step-size required to move from one element to the next in
    memory. For example, a contiguous ``(3, 4)`` array of type
    ``int16`` in C-order has strides ``(8, 2)``.  This implies that
    to move from element to element in memory requires jumps of 2 bytes.
    To move from row-to-row, one needs to jump 8 bytes at a time
    (``2 * 4``).
ctypes : ctypes object
    Class containing properties of the array needed for interaction
    with ctypes.
base : ndarray
    If the array is a view into another array, that array is its `base`
    (unless that array is also a view).  The `base` array is where the
    array data is actually stored.

See Also
--------
array : Construct an array.
zeros : Create an array, each element of which is zero.
empty : Create an array, but leave its allocated memory unchanged (i.e.,
        it contains "garbage").
dtype : Create a data-type.
numpy.typing.NDArray : An ndarray alias :term:`generic <generic type>`
                       w.r.t. its `dtype.type <numpy.dtype.type>`.

Notes
-----
There are two modes of creating an array using ``__new__``:

1. If `buffer` is None, then only `shape`, `dtype`, and `order`
   are used.
2. If `buffer` is an object exposing the buffer interface, then
   all keywords are interpreted.

No ``__init__`` method is needed because the array is fully initialized
after the ``__new__`` method.

Examples
--------
These examples illustrate the low-level `ndarray` constructor.  Refer
to the `See Also` section above for easier ways of constructing an
ndarray.

First mode, `buffer` is None:

>>> np.ndarray(shape=(2,2), dtype=float, order='F')
array([[0.0e+000, 0.0e+000], # random
       [     nan, 2.5e-323]])

Second mode:

>>> np.ndarray((2,), buffer=np.array([1,2,3]),
...            offset=np.int_().itemsize,
...            dtype=int) # offset = 1*itemsize, i.e. skip first element
array([2, 3])

#### Methods

**`dumps(...)`**

a.dumps()

Returns the pickle of the array as a string.
pickle.loads will convert the string back to an array.

Parameters
----------
None

**`dump(...)`**

a.dump(file)

Dump a pickle of the array to the specified file.
The array can be read back with pickle.load or numpy.load.

Parameters
----------
file : str or Path
A string naming the dump file.

.. versionchanged:: 1.17.0
`pathlib.Path` objects are now accepted.

**`all(...)`**

a.all(axis=None, out=None, keepdims=False, *, where=True)

Returns True if all elements evaluate to True.

Refer to `numpy.all` for full documentation.

See Also
--------
numpy.all : equivalent function

**`any(...)`**

a.any(axis=None, out=None, keepdims=False, *, where=True)

Returns True if any of the elements of `a` evaluate to True.

Refer to `numpy.any` for full documentation.

See Also
--------
numpy.any : equivalent function

**`argmax(...)`**

a.argmax(axis=None, out=None, *, keepdims=False)

Return indices of the maximum values along the given axis.

Refer to `numpy.argmax` for full documentation.

See Also
--------
numpy.argmax : equivalent function

**`argmin(...)`**

a.argmin(axis=None, out=None, *, keepdims=False)

Return indices of the minimum values along the given axis.

Refer to `numpy.argmin` for detailed documentation.

See Also
--------
numpy.argmin : equivalent function

**`argpartition(...)`**

a.argpartition(kth, axis=-1, kind='introselect', order=None)

Returns the indices that would partition this array.

Refer to `numpy.argpartition` for full documentation.

.. versionadded:: 1.8.0

See Also
--------
numpy.argpartition : equivalent function

**`argsort(...)`**

a.argsort(axis=-1, kind=None, order=None)

Returns the indices that would sort this array.

Refer to `numpy.argsort` for full documentation.

See Also
--------
numpy.argsort : equivalent function

**`astype(...)`**

a.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)

Copy of the array, cast to a specified type.

Parameters
----------
dtype : str or dtype
Typecode or data-type to which the array is cast.
order : {'C', 'F', 'A', 'K'}, optional
Controls the memory layout order of the result.
'C' means C order, 'F' means Fortran order, 'A'
means 'F' order if all the arrays are Fortran contiguous,
'C' order otherwise, and 'K' means as close to the
order the array elements appear in memory as possible.
Default is 'K'.
casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
Controls what kind of data casting may occur. Defaults to 'unsafe'
for backwards compatibility.

* 'no' means the data types should not be cast at all.
* 'equiv' means only byte-order changes are allowed.
* 'safe' means only casts which can preserve values are allowed.
* 'same_kind' means only safe casts or casts within a kind,
like float64 to float32, are allowed.
* 'unsafe' means any data conversions may be done.
subok : bool, optional
If True, then sub-classes will be passed-through (default), otherwise
the returned array will be forced to be a base-class array.
copy : bool, optional
By default, astype always returns a newly allocated array. If this
is set to false, and the `dtype`, `order`, and `subok`
requirements are satisfied, the input array is returned instead
of a copy.

Returns
-------
arr_t : ndarray
Unless `copy` is False and the other conditions for returning the input
array are satisfied (see description for `copy` input parameter), `arr_t`
is a new array of the same shape as the input array, with dtype, order
given by `dtype`, `order`.

Notes
-----
.. versionchanged:: 1.17.0
Casting between a simple data type and a structured one is possible only
for "unsafe" casting.  Casting to multiple fields is allowed, but
casting from multiple fields is not.

.. versionchanged:: 1.9.0
Casting from numeric to string types in 'safe' casting mode requires
that the string dtype length is long enough to store the max
integer/float value converted.

Raises
------
ComplexWarning
When casting from complex to float or int. To avoid this,
one should use ``a.real.astype(t)``.

Examples
--------
>>> x = np.array([1, 2, 2.5])
>>> x
array([1. ,  2. ,  2.5])

>>> x.astype(int)
array([1, 2, 2])

**`byteswap(...)`**

a.byteswap(inplace=False)

Swap the bytes of the array elements

Toggle between low-endian and big-endian data representation by
returning a byteswapped array, optionally swapped in-place.
Arrays of byte-strings are not swapped. The real and imaginary
parts of a complex number are swapped individually.

Parameters
----------
inplace : bool, optional
If ``True``, swap bytes in-place, default is ``False``.

Returns
-------
out : ndarray
The byteswapped array. If `inplace` is ``True``, this is
a view to self.

Examples
--------
>>> A = np.array([1, 256, 8755], dtype=np.int16)
>>> list(map(hex, A))
['0x1', '0x100', '0x2233']
>>> A.byteswap(inplace=True)
array([  256,     1, 13090], dtype=int16)
>>> list(map(hex, A))
['0x100', '0x1', '0x3322']

Arrays of byte-strings are not swapped

>>> A = np.array([b'ceg', b'fac'])
>>> A.byteswap()
array([b'ceg', b'fac'], dtype='|S3')

``A.newbyteorder().byteswap()`` produces an array with the same values
but different representation in memory

>>> A = np.array([1, 2, 3])
>>> A.view(np.uint8)
array([1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0,
0, 0], dtype=uint8)
>>> A.newbyteorder().byteswap(inplace=True)
array([1, 2, 3])
>>> A.view(np.uint8)
array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
0, 3], dtype=uint8)

**`choose(...)`**

a.choose(choices, out=None, mode='raise')

Use an index array to construct a new array from a set of choices.

Refer to `numpy.choose` for full documentation.

See Also
--------
numpy.choose : equivalent function

**`clip(...)`**

a.clip(min=None, max=None, out=None, **kwargs)

Return an array whose values are limited to ``[min, max]``.
One of max or min must be given.

Refer to `numpy.clip` for full documentation.

See Also
--------
numpy.clip : equivalent function

**`compress(...)`**

a.compress(condition, axis=None, out=None)

Return selected slices of this array along given axis.

Refer to `numpy.compress` for full documentation.

See Also
--------
numpy.compress : equivalent function

**`conj(...)`**

a.conj()

Complex-conjugate all elements.

Refer to `numpy.conjugate` for full documentation.

See Also
--------
numpy.conjugate : equivalent function

**`conjugate(...)`**

a.conjugate()

Return the complex conjugate, element-wise.

Refer to `numpy.conjugate` for full documentation.

See Also
--------
numpy.conjugate : equivalent function

**`copy(...)`**

a.copy(order='C')

Return a copy of the array.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
Controls the memory layout of the copy. 'C' means C-order,
'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
'C' otherwise. 'K' means match the layout of `a` as closely
as possible. (Note that this function and :func:`numpy.copy` are very
similar but have different default values for their order=
arguments, and this function always passes sub-classes through.)

See also
--------
numpy.copy : Similar function with different default behavior
numpy.copyto

Notes
-----
This function is the preferred method for creating an array copy.  The
function :func:`numpy.copy` is similar, but it defaults to using order 'K',
and will not pass sub-classes through by default.

Examples
--------
>>> x = np.array([[1,2,3],[4,5,6]], order='F')

>>> y = x.copy()

>>> x.fill(0)

>>> x
array([[0, 0, 0],
[0, 0, 0]])

>>> y
array([[1, 2, 3],
[4, 5, 6]])

>>> y.flags['C_CONTIGUOUS']
True

**`cumprod(...)`**

a.cumprod(axis=None, dtype=None, out=None)

Return the cumulative product of the elements along the given axis.

Refer to `numpy.cumprod` for full documentation.

See Also
--------
numpy.cumprod : equivalent function

**`cumsum(...)`**

a.cumsum(axis=None, dtype=None, out=None)

Return the cumulative sum of the elements along the given axis.

Refer to `numpy.cumsum` for full documentation.

See Also
--------
numpy.cumsum : equivalent function

**`diagonal(...)`**

a.diagonal(offset=0, axis1=0, axis2=1)

Return specified diagonals. In NumPy 1.9 the returned array is a
read-only view instead of a copy as in previous NumPy versions.  In
a future version the read-only restriction will be removed.

Refer to :func:`numpy.diagonal` for full documentation.

See Also
--------
numpy.diagonal : equivalent function

**`dot(...)`**

*No documentation available.*

**`fill(...)`**

a.fill(value)

Fill the array with a scalar value.

Parameters
----------
value : scalar
All elements of `a` will be assigned this value.

Examples
--------
>>> a = np.array([1, 2])
>>> a.fill(0)
>>> a
array([0, 0])
>>> a = np.empty(2)
>>> a.fill(1)
>>> a
array([1.,  1.])

Fill expects a scalar value and always behaves the same as assigning
to a single array element.  The following is a rare example where this
distinction is important:

>>> a = np.array([None, None], dtype=object)
>>> a[0] = np.array(3)
>>> a
array([array(3), None], dtype=object)
>>> a.fill(np.array(3))
>>> a
array([array(3), array(3)], dtype=object)

Where other forms of assignments will unpack the array being assigned:

>>> a[...] = np.array(3)
>>> a
array([3, 3], dtype=object)

**`flatten(...)`**

a.flatten(order='C')

Return a copy of the array collapsed into one dimension.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
'C' means to flatten in row-major (C-style) order.
'F' means to flatten in column-major (Fortran-
style) order. 'A' means to flatten in column-major
order if `a` is Fortran *contiguous* in memory,
row-major order otherwise. 'K' means to flatten
`a` in the order the elements occur in memory.
The default is 'C'.

Returns
-------
y : ndarray
A copy of the input array, flattened to one dimension.

See Also
--------
ravel : Return a flattened array.
flat : A 1-D flat iterator over the array.

Examples
--------
>>> a = np.array([[1,2], [3,4]])
>>> a.flatten()
array([1, 2, 3, 4])
>>> a.flatten('F')
array([1, 3, 2, 4])

**`getfield(...)`**

a.getfield(dtype, offset=0)

Returns a field of the given array as a certain type.

A field is a view of the array data with a given data-type. The values in
the view are determined by the given type and the offset into the current
array in bytes. The offset needs to be such that the view dtype fits in the
array dtype; for example an array of dtype complex128 has 16-byte elements.
If taking a view with a 32-bit integer (4 bytes), the offset needs to be
between 0 and 12 bytes.

Parameters
----------
dtype : str or dtype
The data type of the view. The dtype size of the view can not be larger
than that of the array itself.
offset : int
Number of bytes to skip before beginning the element view.

Examples
--------
>>> x = np.diag([1.+1.j]*2)
>>> x[1, 1] = 2 + 4.j
>>> x
array([[1.+1.j,  0.+0.j],
[0.+0.j,  2.+4.j]])
>>> x.getfield(np.float64)
array([[1.,  0.],
[0.,  2.]])

By choosing an offset of 8 bytes we can select the complex part of the
array for our view:

>>> x.getfield(np.float64, offset=8)
array([[1.,  0.],
[0.,  4.]])

**`item(...)`**

a.item(*args)

Copy an element of an array to a standard Python scalar and return it.

Parameters
----------
\*args : Arguments (variable number and type)

* none: in this case, the method only works for arrays
with one element (`a.size == 1`), which element is
copied into a standard Python scalar object and returned.

* int_type: this argument is interpreted as a flat index into
the array, specifying which element to copy and return.

* tuple of int_types: functions as does a single int_type argument,
except that the argument is interpreted as an nd-index into the
array.

Returns
-------
z : Standard Python scalar object
A copy of the specified element of the array as a suitable
Python scalar

Notes
-----
When the data type of `a` is longdouble or clongdouble, item() returns
a scalar array object because there is no available Python scalar that
would not lose information. Void arrays return a buffer object for item(),
unless fields are defined, in which case a tuple is returned.

`item` is very similar to a[args], except, instead of an array scalar,
a standard Python scalar is returned. This can be useful for speeding up
access to elements of the array and doing arithmetic on elements of the
array using Python's optimized math.

Examples
--------
>>> np.random.seed(123)
>>> x = np.random.randint(9, size=(3, 3))
>>> x
array([[2, 2, 6],
[1, 3, 6],
[1, 0, 1]])
>>> x.item(3)
1
>>> x.item(7)
0
>>> x.item((0, 1))
2
>>> x.item((2, 2))
1

**`itemset(...)`**

a.itemset(*args)

Insert scalar into an array (scalar is cast to array's dtype, if possible)

There must be at least 1 argument, and define the last argument
as *item*.  Then, ``a.itemset(*args)`` is equivalent to but faster
than ``a[args] = item``.  The item should be a scalar value and `args`
must select a single item in the array `a`.

Parameters
----------
\*args : Arguments
If one argument: a scalar, only used in case `a` is of size 1.
If two arguments: the last argument is the value to be set
and must be a scalar, the first argument specifies a single array
element location. It is either an int or a tuple.

Notes
-----
Compared to indexing syntax, `itemset` provides some speed increase
for placing a scalar into a particular location in an `ndarray`,
if you must do this.  However, generally this is discouraged:
among other problems, it complicates the appearance of the code.
Also, when using `itemset` (and `item`) inside a loop, be sure
to assign the methods to a local variable to avoid the attribute
look-up at each loop iteration.

Examples
--------
>>> np.random.seed(123)
>>> x = np.random.randint(9, size=(3, 3))
>>> x
array([[2, 2, 6],
[1, 3, 6],
[1, 0, 1]])
>>> x.itemset(4, 0)
>>> x.itemset((2, 2), 9)
>>> x
array([[2, 2, 6],
[1, 0, 6],
[1, 0, 9]])

**`max(...)`**

a.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)

Return the maximum along a given axis.

Refer to `numpy.amax` for full documentation.

See Also
--------
numpy.amax : equivalent function

**`mean(...)`**

a.mean(axis=None, dtype=None, out=None, keepdims=False, *, where=True)

Returns the average of the array elements along given axis.

Refer to `numpy.mean` for full documentation.

See Also
--------
numpy.mean : equivalent function

**`min(...)`**

a.min(axis=None, out=None, keepdims=False, initial=<no value>, where=True)

Return the minimum along a given axis.

Refer to `numpy.amin` for full documentation.

See Also
--------
numpy.amin : equivalent function

**`newbyteorder(...)`**

arr.newbyteorder(new_order='S', /)

Return the array with the same data viewed with a different byte order.

Equivalent to::

arr.view(arr.dtype.newbytorder(new_order))

Changes are also made in all fields and sub-arrays of the array data
type.



Parameters
----------
new_order : string, optional
Byte order to force; a value from the byte order specifications
below. `new_order` codes can be any of:

* 'S' - swap dtype from current to opposite endian
* {'<', 'little'} - little endian
* {'>', 'big'} - big endian
* {'=', 'native'} - native order, equivalent to `sys.byteorder`
* {'|', 'I'} - ignore (no change to byte order)

The default value ('S') results in swapping the current
byte order.


Returns
-------
new_arr : array
New array object with the dtype reflecting given change to the
byte order.

**`nonzero(...)`**

a.nonzero()

Return the indices of the elements that are non-zero.

Refer to `numpy.nonzero` for full documentation.

See Also
--------
numpy.nonzero : equivalent function

**`partition(...)`**

a.partition(kth, axis=-1, kind='introselect', order=None)

Rearranges the elements in the array in such a way that the value of the
element in kth position is in the position it would be in a sorted array.
All elements smaller than the kth element are moved before this element and
all equal or greater are moved behind it. The ordering of the elements in
the two partitions is undefined.

.. versionadded:: 1.8.0

Parameters
----------
kth : int or sequence of ints
Element index to partition by. The kth element value will be in its
final sorted position and all smaller elements will be moved before it
and all equal or greater elements behind it.
The order of all elements in the partitions is undefined.
If provided with a sequence of kth it will partition all elements
indexed by kth of them into their sorted position at once.

.. deprecated:: 1.22.0
Passing booleans as index is deprecated.
axis : int, optional
Axis along which to sort. Default is -1, which means sort along the
last axis.
kind : {'introselect'}, optional
Selection algorithm. Default is 'introselect'.
order : str or list of str, optional
When `a` is an array with fields defined, this argument specifies
which fields to compare first, second, etc. A single field can
be specified as a string, and not all fields need to be specified,
but unspecified fields will still be used, in the order in which
they come up in the dtype, to break ties.

See Also
--------
numpy.partition : Return a partitioned copy of an array.
argpartition : Indirect partition.
sort : Full sort.

Notes
-----
See ``np.partition`` for notes on the different algorithms.

Examples
--------
>>> a = np.array([3, 4, 2, 1])
>>> a.partition(3)
>>> a
array([2, 1, 3, 4])

>>> a.partition((1, 3))
>>> a
array([1, 2, 3, 4])

**`prod(...)`**

a.prod(axis=None, dtype=None, out=None, keepdims=False, initial=1, where=True)

Return the product of the array elements over the given axis

Refer to `numpy.prod` for full documentation.

See Also
--------
numpy.prod : equivalent function

**`ptp(...)`**

a.ptp(axis=None, out=None, keepdims=False)

Peak to peak (maximum - minimum) value along a given axis.

Refer to `numpy.ptp` for full documentation.

See Also
--------
numpy.ptp : equivalent function

**`put(...)`**

a.put(indices, values, mode='raise')

Set ``a.flat[n] = values[n]`` for all `n` in indices.

Refer to `numpy.put` for full documentation.

See Also
--------
numpy.put : equivalent function

**`ravel(...)`**

a.ravel([order])

Return a flattened array.

Refer to `numpy.ravel` for full documentation.

See Also
--------
numpy.ravel : equivalent function

ndarray.flat : a flat iterator on the array.

**`repeat(...)`**

a.repeat(repeats, axis=None)

Repeat elements of an array.

Refer to `numpy.repeat` for full documentation.

See Also
--------
numpy.repeat : equivalent function

**`reshape(...)`**

a.reshape(shape, order='C')

Returns an array containing the same data with a new shape.

Refer to `numpy.reshape` for full documentation.

See Also
--------
numpy.reshape : equivalent function

Notes
-----
Unlike the free function `numpy.reshape`, this method on `ndarray` allows
the elements of the shape parameter to be passed in as separate arguments.
For example, ``a.reshape(10, 11)`` is equivalent to
``a.reshape((10, 11))``.

**`resize(...)`**

a.resize(new_shape, refcheck=True)

Change shape and size of array in-place.

Parameters
----------
new_shape : tuple of ints, or `n` ints
Shape of resized array.
refcheck : bool, optional
If False, reference count will not be checked. Default is True.

Returns
-------
None

Raises
------
ValueError
If `a` does not own its own data or references or views to it exist,
and the data memory must be changed.
PyPy only: will always raise if the data memory must be changed, since
there is no reliable way to determine if references or views to it
exist.

SystemError
If the `order` keyword argument is specified. This behaviour is a
bug in NumPy.

See Also
--------
resize : Return a new array with the specified shape.

Notes
-----
This reallocates space for the data area if necessary.

Only contiguous arrays (data elements consecutive in memory) can be
resized.

The purpose of the reference count check is to make sure you
do not use this array as a buffer for another Python object and then
reallocate the memory. However, reference counts can increase in
other ways so if you are sure that you have not shared the memory
for this array with another Python object, then you may safely set
`refcheck` to False.

Examples
--------
Shrinking an array: array is flattened (in the order that the data are
stored in memory), resized, and reshaped:

>>> a = np.array([[0, 1], [2, 3]], order='C')
>>> a.resize((2, 1))
>>> a
array([[0],
[1]])

>>> a = np.array([[0, 1], [2, 3]], order='F')
>>> a.resize((2, 1))
>>> a
array([[0],
[2]])

Enlarging an array: as above, but missing entries are filled with zeros:

>>> b = np.array([[0, 1], [2, 3]])
>>> b.resize(2, 3) # new_shape parameter doesn't have to be a tuple
>>> b
array([[0, 1, 2],
[3, 0, 0]])

Referencing an array prevents resizing...

>>> c = a
>>> a.resize((1, 1))
Traceback (most recent call last):
...
ValueError: cannot resize an array that references or is referenced ...

Unless `refcheck` is False:

>>> a.resize((1, 1), refcheck=False)
>>> a
array([[0]])
>>> c
array([[0]])

**`round(...)`**

a.round(decimals=0, out=None)

Return `a` with each element rounded to the given number of decimals.

Refer to `numpy.around` for full documentation.

See Also
--------
numpy.around : equivalent function

**`searchsorted(...)`**

a.searchsorted(v, side='left', sorter=None)

Find indices where elements of v should be inserted in a to maintain order.

For full documentation, see `numpy.searchsorted`

See Also
--------
numpy.searchsorted : equivalent function

**`setfield(...)`**

a.setfield(val, dtype, offset=0)

Put a value into a specified place in a field defined by a data-type.

Place `val` into `a`'s field defined by `dtype` and beginning `offset`
bytes into the field.

Parameters
----------
val : object
Value to be placed in field.
dtype : dtype object
Data-type of the field in which to place `val`.
offset : int, optional
The number of bytes into the field at which to place `val`.

Returns
-------
None

See Also
--------
getfield

Examples
--------
>>> x = np.eye(3)
>>> x.getfield(np.float64)
array([[1.,  0.,  0.],
[0.,  1.,  0.],
[0.,  0.,  1.]])
>>> x.setfield(3, np.int32)
>>> x.getfield(np.int32)
array([[3, 3, 3],
[3, 3, 3],
[3, 3, 3]], dtype=int32)
>>> x
array([[1.0e+000, 1.5e-323, 1.5e-323],
[1.5e-323, 1.0e+000, 1.5e-323],
[1.5e-323, 1.5e-323, 1.0e+000]])
>>> x.setfield(np.eye(3), np.int32)
>>> x
array([[1.,  0.,  0.],
[0.,  1.,  0.],
[0.,  0.,  1.]])

**`setflags(...)`**

a.setflags(write=None, align=None, uic=None)

Set array flags WRITEABLE, ALIGNED, WRITEBACKIFCOPY,
respectively.

These Boolean-valued flags affect how numpy interprets the memory
area used by `a` (see Notes below). The ALIGNED flag can only
be set to True if the data is actually aligned according to the type.
The WRITEBACKIFCOPY and flag can never be set
to True. The flag WRITEABLE can only be set to True if the array owns its
own memory, or the ultimate owner of the memory exposes a writeable buffer
interface, or is a string. (The exception for string is made so that
unpickling can be done without copying memory.)

Parameters
----------
write : bool, optional
Describes whether or not `a` can be written to.
align : bool, optional
Describes whether or not `a` is aligned properly for its type.
uic : bool, optional
Describes whether or not `a` is a copy of another "base" array.

Notes
-----
Array flags provide information about how the memory area used
for the array is to be interpreted. There are 7 Boolean flags
in use, only four of which can be changed by the user:
WRITEBACKIFCOPY, WRITEABLE, and ALIGNED.

WRITEABLE (W) the data area can be written to;

ALIGNED (A) the data and strides are aligned appropriately for the hardware
(as determined by the compiler);

WRITEBACKIFCOPY (X) this array is a copy of some other array (referenced
by .base). When the C-API function PyArray_ResolveWritebackIfCopy is
called, the base array will be updated with the contents of this array.

All flags can be accessed using the single (upper case) letter as well
as the full name.

Examples
--------
>>> y = np.array([[3, 1, 7],
...               [2, 0, 0],
...               [8, 5, 9]])
>>> y
array([[3, 1, 7],
[2, 0, 0],
[8, 5, 9]])
>>> y.flags
C_CONTIGUOUS : True
F_CONTIGUOUS : False
OWNDATA : True
WRITEABLE : True
ALIGNED : True
WRITEBACKIFCOPY : False
>>> y.setflags(write=0, align=0)
>>> y.flags
C_CONTIGUOUS : True
F_CONTIGUOUS : False
OWNDATA : True
WRITEABLE : False
ALIGNED : False
WRITEBACKIFCOPY : False
>>> y.setflags(uic=1)
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
ValueError: cannot set WRITEBACKIFCOPY flag to True

**`sort(...)`**

a.sort(axis=-1, kind=None, order=None)

Sort an array in-place. Refer to `numpy.sort` for full documentation.

Parameters
----------
axis : int, optional
Axis along which to sort. Default is -1, which means sort along the
last axis.
kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
and 'mergesort' use timsort under the covers and, in general, the
actual implementation will vary with datatype. The 'mergesort' option
is retained for backwards compatibility.

.. versionchanged:: 1.15.0
The 'stable' option was added.

order : str or list of str, optional
When `a` is an array with fields defined, this argument specifies
which fields to compare first, second, etc.  A single field can
be specified as a string, and not all fields need be specified,
but unspecified fields will still be used, in the order in which
they come up in the dtype, to break ties.

See Also
--------
numpy.sort : Return a sorted copy of an array.
numpy.argsort : Indirect sort.
numpy.lexsort : Indirect stable sort on multiple keys.
numpy.searchsorted : Find elements in sorted array.
numpy.partition: Partial sort.

Notes
-----
See `numpy.sort` for notes on the different sorting algorithms.

Examples
--------
>>> a = np.array([[1,4], [3,1]])
>>> a.sort(axis=1)
>>> a
array([[1, 4],
[1, 3]])
>>> a.sort(axis=0)
>>> a
array([[1, 3],
[1, 4]])

Use the `order` keyword to specify a field to use when sorting a
structured array:

>>> a = np.array([('a', 2), ('c', 1)], dtype=[('x', 'S1'), ('y', int)])
>>> a.sort(order='y')
>>> a
array([(b'c', 1), (b'a', 2)],
dtype=[('x', 'S1'), ('y', '<i8')])

**`squeeze(...)`**

a.squeeze(axis=None)

Remove axes of length one from `a`.

Refer to `numpy.squeeze` for full documentation.

See Also
--------
numpy.squeeze : equivalent function

**`std(...)`**

a.std(axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True)

Returns the standard deviation of the array elements along given axis.

Refer to `numpy.std` for full documentation.

See Also
--------
numpy.std : equivalent function

**`sum(...)`**

a.sum(axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True)

Return the sum of the array elements over the given axis.

Refer to `numpy.sum` for full documentation.

See Also
--------
numpy.sum : equivalent function

**`swapaxes(...)`**

a.swapaxes(axis1, axis2)

Return a view of the array with `axis1` and `axis2` interchanged.

Refer to `numpy.swapaxes` for full documentation.

See Also
--------
numpy.swapaxes : equivalent function

**`take(...)`**

a.take(indices, axis=None, out=None, mode='raise')

Return an array formed from the elements of `a` at the given indices.

Refer to `numpy.take` for full documentation.

See Also
--------
numpy.take : equivalent function

**`tobytes(...)`**

a.tobytes(order='C')

Construct Python bytes containing the raw data bytes in the array.

Constructs Python bytes showing a copy of the raw contents of
data memory. The bytes object is produced in C-order by default.
This behavior is controlled by the ``order`` parameter.

.. versionadded:: 1.9.0

Parameters
----------
order : {'C', 'F', 'A'}, optional
Controls the memory layout of the bytes object. 'C' means C-order,
'F' means F-order, 'A' (short for *Any*) means 'F' if `a` is
Fortran contiguous, 'C' otherwise. Default is 'C'.

Returns
-------
s : bytes
Python bytes exhibiting a copy of `a`'s raw data.

See also
--------
frombuffer
Inverse of this operation, construct a 1-dimensional array from Python
bytes.

Examples
--------
>>> x = np.array([[0, 1], [2, 3]], dtype='<u2')
>>> x.tobytes()
b'\x00\x00\x01\x00\x02\x00\x03\x00'
>>> x.tobytes('C') == x.tobytes()
True
>>> x.tobytes('F')
b'\x00\x00\x02\x00\x01\x00\x03\x00'

**`tofile(...)`**

a.tofile(fid, sep="", format="%s")

Write array to a file as text or binary (default).

Data is always written in 'C' order, independent of the order of `a`.
The data produced by this method can be recovered using the function
fromfile().

Parameters
----------
fid : file or str or Path
An open file object, or a string containing a filename.

.. versionchanged:: 1.17.0
`pathlib.Path` objects are now accepted.

sep : str
Separator between array items for text output.
If "" (empty), a binary file is written, equivalent to
``file.write(a.tobytes())``.
format : str
Format string for text file output.
Each entry in the array is formatted to text by first converting
it to the closest Python type, and then using "format" % item.

Notes
-----
This is a convenience function for quick storage of array data.
Information on endianness and precision is lost, so this method is not a
good choice for files intended to archive data or transport data between
machines with different endianness. Some of these problems can be overcome
by outputting the data as text files, at the expense of speed and file
size.

When fid is a file object, array contents are directly written to the
file, bypassing the file object's ``write`` method. As a result, tofile
cannot be used with files objects supporting compression (e.g., GzipFile)
or file-like objects that do not support ``fileno()`` (e.g., BytesIO).

**`tolist(...)`**

a.tolist()

Return the array as an ``a.ndim``-levels deep nested list of Python scalars.

Return a copy of the array data as a (nested) Python list.
Data items are converted to the nearest compatible builtin Python type, via
the `~numpy.ndarray.item` function.

If ``a.ndim`` is 0, then since the depth of the nested list is 0, it will
not be a list at all, but a simple Python scalar.

Parameters
----------
none

Returns
-------
y : object, or list of object, or list of list of object, or ...
The possibly nested list of array elements.

Notes
-----
The array may be recreated via ``a = np.array(a.tolist())``, although this
may sometimes lose precision.

Examples
--------
For a 1D array, ``a.tolist()`` is almost the same as ``list(a)``,
except that ``tolist`` changes numpy scalars to Python scalars:

>>> a = np.uint32([1, 2])
>>> a_list = list(a)
>>> a_list
[1, 2]
>>> type(a_list[0])
<class 'numpy.uint32'>
>>> a_tolist = a.tolist()
>>> a_tolist
[1, 2]
>>> type(a_tolist[0])
<class 'int'>

Additionally, for a 2D array, ``tolist`` applies recursively:

>>> a = np.array([[1, 2], [3, 4]])
>>> list(a)
[array([1, 2]), array([3, 4])]
>>> a.tolist()
[[1, 2], [3, 4]]

The base case for this recursion is a 0D array:

>>> a = np.array(1)
>>> list(a)
Traceback (most recent call last):
...
TypeError: iteration over a 0-d array
>>> a.tolist()
1

**`tostring(...)`**

a.tostring(order='C')

A compatibility alias for `tobytes`, with exactly the same behavior.

Despite its name, it returns `bytes` not `str`\ s.

.. deprecated:: 1.19.0

**`trace(...)`**

a.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)

Return the sum along diagonals of the array.

Refer to `numpy.trace` for full documentation.

See Also
--------
numpy.trace : equivalent function

**`transpose(...)`**

a.transpose(*axes)

Returns a view of the array with axes transposed.

Refer to `numpy.transpose` for full documentation.

Parameters
----------
axes : None, tuple of ints, or `n` ints

* None or no argument: reverses the order of the axes.

* tuple of ints: `i` in the `j`-th place in the tuple means that the
array's `i`-th axis becomes the transposed array's `j`-th axis.

* `n` ints: same as an n-tuple of the same ints (this form is
intended simply as a "convenience" alternative to the tuple form).

Returns
-------
p : ndarray
View of the array with its axes suitably permuted.

See Also
--------
transpose : Equivalent function.
ndarray.T : Array property returning the array transposed.
ndarray.reshape : Give a new shape to an array without changing its data.

Examples
--------
>>> a = np.array([[1, 2], [3, 4]])
>>> a
array([[1, 2],
[3, 4]])
>>> a.transpose()
array([[1, 3],
[2, 4]])
>>> a.transpose((1, 0))
array([[1, 3],
[2, 4]])
>>> a.transpose(1, 0)
array([[1, 3],
[2, 4]])

>>> a = np.array([1, 2, 3, 4])
>>> a
array([1, 2, 3, 4])
>>> a.transpose()
array([1, 2, 3, 4])

**`var(...)`**

a.var(axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True)

Returns the variance of the array elements, along given axis.

Refer to `numpy.var` for full documentation.

See Also
--------
numpy.var : equivalent function

**`view(...)`**

a.view([dtype][, type])

New view of array with the same data.

.. note::
Passing None for ``dtype`` is different from omitting the parameter,
since the former invokes ``dtype(None)`` which is an alias for
``dtype('float_')``.

Parameters
----------
dtype : data-type or ndarray sub-class, optional
Data-type descriptor of the returned view, e.g., float32 or int16.
Omitting it results in the view having the same data-type as `a`.
This argument can also be specified as an ndarray sub-class, which
then specifies the type of the returned object (this is equivalent to
setting the ``type`` parameter).
type : Python type, optional
Type of the returned view, e.g., ndarray or matrix.  Again, omission
of the parameter results in type preservation.

Notes
-----
``a.view()`` is used two different ways:

``a.view(some_dtype)`` or ``a.view(dtype=some_dtype)`` constructs a view
of the array's memory with a different data-type.  This can cause a
reinterpretation of the bytes of memory.

``a.view(ndarray_subclass)`` or ``a.view(type=ndarray_subclass)`` just
returns an instance of `ndarray_subclass` that looks at the same array
(same shape, dtype, etc.)  This does not cause a reinterpretation of the
memory.

For ``a.view(some_dtype)``, if ``some_dtype`` has a different number of
bytes per entry than the previous dtype (for example, converting a regular
array to a structured array), then the last axis of ``a`` must be
contiguous. This axis will be resized in the result.

.. versionchanged:: 1.23.0
Only the last axis needs to be contiguous. Previously, the entire array
had to be C-contiguous.

Examples
--------
>>> x = np.array([(1, 2)], dtype=[('a', np.int8), ('b', np.int8)])

Viewing array data using a different type and dtype:

>>> y = x.view(dtype=np.int16, type=np.matrix)
>>> y
matrix([[513]], dtype=int16)
>>> print(type(y))
<class 'numpy.matrix'>

Creating a view on a structured array so it can be used in calculations

>>> x = np.array([(1, 2),(3,4)], dtype=[('a', np.int8), ('b', np.int8)])
>>> xv = x.view(dtype=np.int8).reshape(-1,2)
>>> xv
array([[1, 2],
[3, 4]], dtype=int8)
>>> xv.mean(0)
array([2.,  3.])

Making changes to the view changes the underlying array

>>> xv[0,1] = 20
>>> x
array([(1, 20), (3,  4)], dtype=[('a', 'i1'), ('b', 'i1')])

Using a view to convert an array to a recarray:

>>> z = x.view(np.recarray)
>>> z.a
array([1, 3], dtype=int8)

Views share data:

>>> x[0] = (9, 10)
>>> z[0]
(9, 10)

Views that change the dtype size (bytes per entry) should normally be
avoided on arrays defined by slices, transposes, fortran-ordering, etc.:

>>> x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
>>> y = x[:, ::2]
>>> y
array([[1, 3],
[4, 6]], dtype=int16)
>>> y.view(dtype=[('width', np.int16), ('length', np.int16)])
Traceback (most recent call last):
...
ValueError: To change to a dtype of a different size, the last axis must be contiguous
>>> z = y.copy()
>>> z.view(dtype=[('width', np.int16), ('length', np.int16)])
array([[(1, 3)],
[(4, 6)]], dtype=[('width', '<i2'), ('length', '<i2')])

However, views that change dtype are totally fine for arrays with a
contiguous last axis, even if the rest of the axes are not C-contiguous:

>>> x = np.arange(2 * 3 * 4, dtype=np.int8).reshape(2, 3, 4)
>>> x.transpose(1, 0, 2).view(np.int16)
array([[[ 256,  770],
[3340, 3854]],
<BLANKLINE>
[[1284, 1798],
[4368, 4882]],
<BLANKLINE>
[[2312, 2826],
[5396, 5910]]], dtype=int16)

### ndenumerate
Module: `numpy`

Multidimensional index iterator.

Return an iterator yielding pairs of array coordinates and values.

Parameters
----------
arr : ndarray
  Input array.

See Also
--------
ndindex, flatiter

Examples
--------
>>> a = np.array([[1, 2], [3, 4]])
>>> for index, x in np.ndenumerate(a):
...     print(index, x)
(0, 0) 1
(0, 1) 2
(1, 0) 3
(1, 1) 4

### ndindex
Module: `numpy`

An N-dimensional iterator object to index arrays.

Given the shape of an array, an `ndindex` instance iterates over
the N-dimensional index of the array. At each iteration a tuple
of indices is returned, the last dimension is iterated over first.

Parameters
----------
shape : ints, or a single tuple of ints
    The size of each dimension of the array can be passed as
    individual parameters or as the elements of a tuple.

See Also
--------
ndenumerate, flatiter

Examples
--------
Dimensions as individual arguments

>>> for index in np.ndindex(3, 2, 1):
...     print(index)
(0, 0, 0)
(0, 1, 0)
(1, 0, 0)
(1, 1, 0)
(2, 0, 0)
(2, 1, 0)

Same dimensions - but in a tuple ``(3, 2, 1)``

>>> for index in np.ndindex((3, 2, 1)):
...     print(index)
(0, 0, 0)
(0, 1, 0)
(1, 0, 0)
(1, 1, 0)
(2, 0, 0)
(2, 1, 0)

#### Methods

**`ndincr(self)`**

Increment the multi-dimensional index by one.

This method is for backward compatibility only: do not use.

.. deprecated:: 1.20.0
This method has been advised against since numpy 1.8.0, but only
started emitting DeprecationWarning as of this version.

### nditer
Module: `numpy`

nditer(op, flags=None, op_flags=None, op_dtypes=None, order='K', casting='safe', op_axes=None, itershape=None, buffersize=0)

Efficient multi-dimensional iterator object to iterate over arrays.
To get started using this object, see the
:ref:`introductory guide to array iteration <arrays.nditer>`.

Parameters
----------
op : ndarray or sequence of array_like
    The array(s) to iterate over.

flags : sequence of str, optional
      Flags to control the behavior of the iterator.

      * ``buffered`` enables buffering when required.
      * ``c_index`` causes a C-order index to be tracked.
      * ``f_index`` causes a Fortran-order index to be tracked.
      * ``multi_index`` causes a multi-index, or a tuple of indices
        with one per iteration dimension, to be tracked.
      * ``common_dtype`` causes all the operands to be converted to
        a common data type, with copying or buffering as necessary.
      * ``copy_if_overlap`` causes the iterator to determine if read
        operands have overlap with write operands, and make temporary
        copies as necessary to avoid overlap. False positives (needless
        copying) are possible in some cases.
      * ``delay_bufalloc`` delays allocation of the buffers until
        a reset() call is made. Allows ``allocate`` operands to
        be initialized before their values are copied into the buffers.
      * ``external_loop`` causes the ``values`` given to be
        one-dimensional arrays with multiple values instead of
        zero-dimensional arrays.
      * ``grow_inner`` allows the ``value`` array sizes to be made
        larger than the buffer size when both ``buffered`` and
        ``external_loop`` is used.
      * ``ranged`` allows the iterator to be restricted to a sub-range
        of the iterindex values.
      * ``refs_ok`` enables iteration of reference types, such as
        object arrays.
      * ``reduce_ok`` enables iteration of ``readwrite`` operands
        which are broadcasted, also known as reduction operands.
      * ``zerosize_ok`` allows `itersize` to be zero.
op_flags : list of list of str, optional
      This is a list of flags for each operand. At minimum, one of
      ``readonly``, ``readwrite``, or ``writeonly`` must be specified.

      * ``readonly`` indicates the operand will only be read from.
      * ``readwrite`` indicates the operand will be read from and written to.
      * ``writeonly`` indicates the operand will only be written to.
      * ``no_broadcast`` prevents the operand from being broadcasted.
      * ``contig`` forces the operand data to be contiguous.
      * ``aligned`` forces the operand data to be aligned.
      * ``nbo`` forces the operand data to be in native byte order.
      * ``copy`` allows a temporary read-only copy if required.
      * ``updateifcopy`` allows a temporary read-write copy if required.
      * ``allocate`` causes the array to be allocated if it is None
        in the ``op`` parameter.
      * ``no_subtype`` prevents an ``allocate`` operand from using a subtype.
      * ``arraymask`` indicates that this operand is the mask to use
        for selecting elements when writing to operands with the
        'writemasked' flag set. The iterator does not enforce this,
        but when writing from a buffer back to the array, it only
        copies those elements indicated by this mask.
      * ``writemasked`` indicates that only elements where the chosen
        ``arraymask`` operand is True will be written to.
      * ``overlap_assume_elementwise`` can be used to mark operands that are
        accessed only in the iterator order, to allow less conservative
        copying when ``copy_if_overlap`` is present.
op_dtypes : dtype or tuple of dtype(s), optional
    The required data type(s) of the operands. If copying or buffering
    is enabled, the data will be converted to/from their original types.
order : {'C', 'F', 'A', 'K'}, optional
    Controls the iteration order. 'C' means C order, 'F' means
    Fortran order, 'A' means 'F' order if all the arrays are Fortran
    contiguous, 'C' order otherwise, and 'K' means as close to the
    order the array elements appear in memory as possible. This also
    affects the element memory order of ``allocate`` operands, as they
    are allocated to be compatible with iteration order.
    Default is 'K'.
casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
    Controls what kind of data casting may occur when making a copy
    or buffering.  Setting this to 'unsafe' is not recommended,
    as it can adversely affect accumulations.

    * 'no' means the data types should not be cast at all.
    * 'equiv' means only byte-order changes are allowed.
    * 'safe' means only casts which can preserve values are allowed.
    * 'same_kind' means only safe casts or casts within a kind,
      like float64 to float32, are allowed.
    * 'unsafe' means any data conversions may be done.
op_axes : list of list of ints, optional
    If provided, is a list of ints or None for each operands.
    The list of axes for an operand is a mapping from the dimensions
    of the iterator to the dimensions of the operand. A value of
    -1 can be placed for entries, causing that dimension to be
    treated as `newaxis`.
itershape : tuple of ints, optional
    The desired shape of the iterator. This allows ``allocate`` operands
    with a dimension mapped by op_axes not corresponding to a dimension
    of a different operand to get a value not equal to 1 for that
    dimension.
buffersize : int, optional
    When buffering is enabled, controls the size of the temporary
    buffers. Set to 0 for the default value.

Attributes
----------
dtypes : tuple of dtype(s)
    The data types of the values provided in `value`. This may be
    different from the operand data types if buffering is enabled.
    Valid only before the iterator is closed.
finished : bool
    Whether the iteration over the operands is finished or not.
has_delayed_bufalloc : bool
    If True, the iterator was created with the ``delay_bufalloc`` flag,
    and no reset() function was called on it yet.
has_index : bool
    If True, the iterator was created with either the ``c_index`` or
    the ``f_index`` flag, and the property `index` can be used to
    retrieve it.
has_multi_index : bool
    If True, the iterator was created with the ``multi_index`` flag,
    and the property `multi_index` can be used to retrieve it.
index
    When the ``c_index`` or ``f_index`` flag was used, this property
    provides access to the index. Raises a ValueError if accessed
    and ``has_index`` is False.
iterationneedsapi : bool
    Whether iteration requires access to the Python API, for example
    if one of the operands is an object array.
iterindex : int
    An index which matches the order of iteration.
itersize : int
    Size of the iterator.
itviews
    Structured view(s) of `operands` in memory, matching the reordered
    and optimized iterator access pattern. Valid only before the iterator
    is closed.
multi_index
    When the ``multi_index`` flag was used, this property
    provides access to the index. Raises a ValueError if accessed
    accessed and ``has_multi_index`` is False.
ndim : int
    The dimensions of the iterator.
nop : int
    The number of iterator operands.
operands : tuple of operand(s)
    The array(s) to be iterated over. Valid only before the iterator is
    closed.
shape : tuple of ints
    Shape tuple, the shape of the iterator.
value
    Value of ``operands`` at current iteration. Normally, this is a
    tuple of array scalars, but if the flag ``external_loop`` is used,
    it is a tuple of one dimensional arrays.

Notes
-----
`nditer` supersedes `flatiter`.  The iterator implementation behind
`nditer` is also exposed by the NumPy C API.

The Python exposure supplies two iteration interfaces, one which follows
the Python iterator protocol, and another which mirrors the C-style
do-while pattern.  The native Python approach is better in most cases, but
if you need the coordinates or index of an iterator, use the C-style pattern.

Examples
--------
Here is how we might write an ``iter_add`` function, using the
Python iterator protocol:

>>> def iter_add_py(x, y, out=None):
...     addop = np.add
...     it = np.nditer([x, y, out], [],
...                 [['readonly'], ['readonly'], ['writeonly','allocate']])
...     with it:
...         for (a, b, c) in it:
...             addop(a, b, out=c)
...         return it.operands[2]

Here is the same function, but following the C-style pattern:

>>> def iter_add(x, y, out=None):
...    addop = np.add
...    it = np.nditer([x, y, out], [],
...                [['readonly'], ['readonly'], ['writeonly','allocate']])
...    with it:
...        while not it.finished:
...            addop(it[0], it[1], out=it[2])
...            it.iternext()
...        return it.operands[2]

Here is an example outer product function:

>>> def outer_it(x, y, out=None):
...     mulop = np.multiply
...     it = np.nditer([x, y, out], ['external_loop'],
...             [['readonly'], ['readonly'], ['writeonly', 'allocate']],
...             op_axes=[list(range(x.ndim)) + [-1] * y.ndim,
...                      [-1] * x.ndim + list(range(y.ndim)),
...                      None])
...     with it:
...         for (a, b, c) in it:
...             mulop(a, b, out=c)
...         return it.operands[2]

>>> a = np.arange(2)+1
>>> b = np.arange(3)+1
>>> outer_it(a,b)
array([[1, 2, 3],
       [2, 4, 6]])

Here is an example function which operates like a "lambda" ufunc:

>>> def luf(lamdaexpr, *args, **kwargs):
...    '''luf(lambdaexpr, op1, ..., opn, out=None, order='K', casting='safe', buffersize=0)'''
...    nargs = len(args)
...    op = (kwargs.get('out',None),) + args
...    it = np.nditer(op, ['buffered','external_loop'],
...            [['writeonly','allocate','no_broadcast']] +
...                            [['readonly','nbo','aligned']]*nargs,
...            order=kwargs.get('order','K'),
...            casting=kwargs.get('casting','safe'),
...            buffersize=kwargs.get('buffersize',0))
...    while not it.finished:
...        it[0] = lamdaexpr(*it[1:])
...        it.iternext()
...    return it.operands[0]

>>> a = np.arange(5)
>>> b = np.ones(5)
>>> luf(lambda i,j:i*i + j/2, a, b)
array([  0.5,   1.5,   4.5,   9.5,  16.5])

If operand flags ``"writeonly"`` or ``"readwrite"`` are used the
operands may be views into the original data with the
`WRITEBACKIFCOPY` flag. In this case `nditer` must be used as a
context manager or the `nditer.close` method must be called before
using the result. The temporary data will be written back to the
original data when the `__exit__` function is called but not before:

>>> a = np.arange(6, dtype='i4')[::-2]
>>> with np.nditer(a, [],
...        [['writeonly', 'updateifcopy']],
...        casting='unsafe',
...        op_dtypes=[np.dtype('f4')]) as i:
...    x = i.operands[0]
...    x[:] = [-1, -2, -3]
...    # a still unchanged here
>>> a, x
(array([-1, -2, -3], dtype=int32), array([-1., -2., -3.], dtype=float32))

It is important to note that once the iterator is exited, dangling
references (like `x` in the example) may or may not share data with
the original data `a`. If writeback semantics were active, i.e. if
`x.base.flags.writebackifcopy` is `True`, then exiting the iterator
will sever the connection between `x` and `a`, writing to `x` will
no longer write to `a`. If writeback semantics are not active, then
`x.data` will still point at some part of `a.data`, and writing to
one will affect the other.

Context management and the `close` method appeared in version 1.15.0.

#### Methods

**`reset(...)`**

reset()

Reset the iterator to its initial state.

**`copy(...)`**

copy()

Get a copy of the iterator in its current state.

Examples
--------
>>> x = np.arange(10)
>>> y = x + 1
>>> it = np.nditer([x, y])
>>> next(it)
(array(0), array(1))
>>> it2 = it.copy()
>>> next(it2)
(array(1), array(2))

**`iternext(...)`**

iternext()

Check whether iterations are left, and perform a single internal iteration
without returning the result.  Used in the C-style pattern do-while
pattern.  For an example, see `nditer`.

Returns
-------
iternext : bool
Whether or not there are iterations left.

**`remove_axis(...)`**

remove_axis(i, /)

Removes axis `i` from the iterator. Requires that the flag "multi_index"
be enabled.

**`remove_multi_index(...)`**

remove_multi_index()

When the "multi_index" flag was specified, this removes it, allowing
the internal iteration structure to be optimized further.

**`enable_external_loop(...)`**

enable_external_loop()

When the "external_loop" was not used during construction, but
is desired, this modifies the iterator to behave as if the flag
was specified.

**`debug_print(...)`**

debug_print()

Print the current state of the `nditer` instance and debug info to stdout.

**`close(...)`**

close()

Resolve all writeback semantics in writeable operands.

.. versionadded:: 1.15.0

See Also
--------

:ref:`nditer-context-manager`

### number
Module: `numpy`

Abstract base class of all numeric scalar types.

### object_
Module: `numpy`

Any Python object.

:Character code: ``'O'``

### poly1d
Module: `numpy`

A one-dimensional polynomial class.

.. note::
   This forms part of the old polynomial API. Since version 1.4, the
   new polynomial API defined in `numpy.polynomial` is preferred.
   A summary of the differences can be found in the
   :doc:`transition guide </reference/routines.polynomials>`.

A convenience class, used to encapsulate "natural" operations on
polynomials so that said operations may take on their customary
form in code (see Examples).

Parameters
----------
c_or_r : array_like
    The polynomial's coefficients, in decreasing powers, or if
    the value of the second parameter is True, the polynomial's
    roots (values where the polynomial evaluates to 0).  For example,
    ``poly1d([1, 2, 3])`` returns an object that represents
    :math:`x^2 + 2x + 3`, whereas ``poly1d([1, 2, 3], True)`` returns
    one that represents :math:`(x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x -6`.
r : bool, optional
    If True, `c_or_r` specifies the polynomial's roots; the default
    is False.
variable : str, optional
    Changes the variable used when printing `p` from `x` to `variable`
    (see Examples).

Examples
--------
Construct the polynomial :math:`x^2 + 2x + 3`:

>>> p = np.poly1d([1, 2, 3])
>>> print(np.poly1d(p))
   2
1 x + 2 x + 3

Evaluate the polynomial at :math:`x = 0.5`:

>>> p(0.5)
4.25

Find the roots:

>>> p.r
array([-1.+1.41421356j, -1.-1.41421356j])
>>> p(p.r)
array([ -4.44089210e-16+0.j,  -4.44089210e-16+0.j]) # may vary

These numbers in the previous line represent (0, 0) to machine precision

Show the coefficients:

>>> p.c
array([1, 2, 3])

Display the order (the leading zero-coefficients are removed):

>>> p.order
2

Show the coefficient of the k-th power in the polynomial
(which is equivalent to ``p.c[-(i+1)]``):

>>> p[1]
2

Polynomials can be added, subtracted, multiplied, and divided
(returns quotient and remainder):

>>> p * p
poly1d([ 1,  4, 10, 12,  9])

>>> (p**3 + 4) / p
(poly1d([ 1.,  4., 10., 12.,  9.]), poly1d([4.]))

``asarray(p)`` gives the coefficient array, so polynomials can be
used in all functions that accept arrays:

>>> p**2 # square of polynomial
poly1d([ 1,  4, 10, 12,  9])

>>> np.square(p) # square of individual coefficients
array([1, 4, 9])

The variable used in the string representation of `p` can be modified,
using the `variable` parameter:

>>> p = np.poly1d([1,2,3], variable='z')
>>> print(p)
   2
1 z + 2 z + 3

Construct a polynomial from its roots:

>>> np.poly1d([1, 2], True)
poly1d([ 1., -3.,  2.])

This is the same polynomial as obtained by:

>>> np.poly1d([1, -1]) * np.poly1d([1, -2])
poly1d([ 1, -3,  2])

#### Methods

**`integ(self, m=1, k=0)`**

Return an antiderivative (indefinite integral) of this polynomial.

Refer to `polyint` for full documentation.

See Also
--------
polyint : equivalent function

**`deriv(self, m=1)`**

Return a derivative of this polynomial.

Refer to `polyder` for full documentation.

See Also
--------
polyder : equivalent function

### recarray
Module: `numpy`

Construct an ndarray that allows field access using attributes.

Arrays may have a data-types containing fields, analogous
to columns in a spread sheet.  An example is ``[(x, int), (y, float)]``,
where each entry in the array is a pair of ``(int, float)``.  Normally,
these attributes are accessed using dictionary lookups such as ``arr['x']``
and ``arr['y']``.  Record arrays allow the fields to be accessed as members
of the array, using ``arr.x`` and ``arr.y``.

Parameters
----------
shape : tuple
    Shape of output array.
dtype : data-type, optional
    The desired data-type.  By default, the data-type is determined
    from `formats`, `names`, `titles`, `aligned` and `byteorder`.
formats : list of data-types, optional
    A list containing the data-types for the different columns, e.g.
    ``['i4', 'f8', 'i4']``.  `formats` does *not* support the new
    convention of using types directly, i.e. ``(int, float, int)``.
    Note that `formats` must be a list, not a tuple.
    Given that `formats` is somewhat limited, we recommend specifying
    `dtype` instead.
names : tuple of str, optional
    The name of each column, e.g. ``('x', 'y', 'z')``.
buf : buffer, optional
    By default, a new array is created of the given shape and data-type.
    If `buf` is specified and is an object exposing the buffer interface,
    the array will use the memory from the existing buffer.  In this case,
    the `offset` and `strides` keywords are available.

Other Parameters
----------------
titles : tuple of str, optional
    Aliases for column names.  For example, if `names` were
    ``('x', 'y', 'z')`` and `titles` is
    ``('x_coordinate', 'y_coordinate', 'z_coordinate')``, then
    ``arr['x']`` is equivalent to both ``arr.x`` and ``arr.x_coordinate``.
byteorder : {'<', '>', '='}, optional
    Byte-order for all fields.
aligned : bool, optional
    Align the fields in memory as the C-compiler would.
strides : tuple of ints, optional
    Buffer (`buf`) is interpreted according to these strides (strides
    define how many bytes each array element, row, column, etc.
    occupy in memory).
offset : int, optional
    Start reading buffer (`buf`) from this offset onwards.
order : {'C', 'F'}, optional
    Row-major (C-style) or column-major (Fortran-style) order.

Returns
-------
rec : recarray
    Empty array of the given shape and type.

See Also
--------
core.records.fromrecords : Construct a record array from data.
record : fundamental data-type for `recarray`.
format_parser : determine a data-type from formats, names, titles.

Notes
-----
This constructor can be compared to ``empty``: it creates a new record
array but does not fill it with data.  To create a record array from data,
use one of the following methods:

1. Create a standard ndarray and convert it to a record array,
   using ``arr.view(np.recarray)``
2. Use the `buf` keyword.
3. Use `np.rec.fromrecords`.

Examples
--------
Create an array with two fields, ``x`` and ``y``:

>>> x = np.array([(1.0, 2), (3.0, 4)], dtype=[('x', '<f8'), ('y', '<i8')])
>>> x
array([(1., 2), (3., 4)], dtype=[('x', '<f8'), ('y', '<i8')])

>>> x['x']
array([1., 3.])

View the array as a record array:

>>> x = x.view(np.recarray)

>>> x.x
array([1., 3.])

>>> x.y
array([2, 4])

Create a new, empty record array:

>>> np.recarray((2,),
... dtype=[('x', int), ('y', float), ('z', int)]) #doctest: +SKIP
rec.array([(-1073741821, 1.2249118382103472e-301, 24547520),
       (3471280, 1.2134086255804012e-316, 0)],
      dtype=[('x', '<i4'), ('y', '<f8'), ('z', '<i4')])

#### Methods

**`field(self, attr, val=None)`**

*No documentation available.*

### record
Module: `numpy`

A data-type scalar that allows field access as attribute lookup.
    

#### Methods

**`pprint(self)`**

Pretty-print all fields.

### short
Module: `numpy`

Signed integer type, compatible with C ``short``.

:Character code: ``'h'``
:Canonical name: `numpy.short`
:Alias on this platform (win32 AMD64): `numpy.int16`: 16-bit signed integer (``-32_768`` to ``32_767``).

#### Methods

**`bit_count(...)`**

int16.bit_count() -> int

Computes the number of 1-bits in the absolute value of the input.
Analogous to the builtin `int.bit_count` or ``popcount`` in C++.

Examples
--------
>>> np.int16(127).bit_count()
7
>>> np.int16(-127).bit_count()
7

### signedinteger
Module: `numpy`

Abstract base class of all signed integer scalar types.

### single
Module: `numpy`

Single-precision floating-point number type, compatible with C ``float``.

:Character code: ``'f'``
:Canonical name: `numpy.single`
:Alias on this platform (win32 AMD64): `numpy.float32`: 32-bit-precision floating-point number type: sign bit, 8 bits exponent, 23 bits mantissa.

#### Methods

**`as_integer_ratio(...)`**

single.as_integer_ratio() -> (int, int)

Return a pair of integers, whose ratio is exactly equal to the original
floating point number, and with a positive denominator.
Raise `OverflowError` on infinities and a `ValueError` on NaNs.

>>> np.single(10.0).as_integer_ratio()
(10, 1)
>>> np.single(0.0).as_integer_ratio()
(0, 1)
>>> np.single(-.25).as_integer_ratio()
(-1, 4)

**`is_integer(...)`**

single.is_integer() -> bool

Return ``True`` if the floating point number is finite with integral
value, and ``False`` otherwise.

.. versionadded:: 1.22

Examples
--------
>>> np.single(-2.0).is_integer()
True
>>> np.single(3.2).is_integer()
False

### singlecomplex
Module: `numpy`

Complex number type composed of two single-precision floating-point
numbers.

:Character code: ``'F'``
:Canonical name: `numpy.csingle`
:Alias: `numpy.singlecomplex`
:Alias on this platform (win32 AMD64): `numpy.complex64`: Complex number type composed of 2 32-bit-precision floating-point numbers.

### str_
Module: `numpy`

A unicode string.

This type strips trailing null codepoints.

>>> s = np.str_("abc\x00")
>>> s
'abc'

Unlike the builtin `str`, this supports the :ref:`python:bufferobjects`, exposing its
contents as UCS4:

>>> m = memoryview(np.str_("abc"))
>>> m.format
'3w'
>>> m.tobytes()
b'a\x00\x00\x00b\x00\x00\x00c\x00\x00\x00'

:Character code: ``'U'``
:Alias: `numpy.unicode_`

### string_
Module: `numpy`

A byte string.

When used in arrays, this type strips trailing null bytes.

:Character code: ``'S'``
:Alias: `numpy.string_`

### timedelta64
Module: `numpy`

A timedelta stored as a 64-bit integer.

See :ref:`arrays.datetime` for more information.

:Character code: ``'m'``

### ubyte
Module: `numpy`

Unsigned integer type, compatible with C ``unsigned char``.

:Character code: ``'B'``
:Canonical name: `numpy.ubyte`
:Alias on this platform (win32 AMD64): `numpy.uint8`: 8-bit unsigned integer (``0`` to ``255``).

#### Methods

**`bit_count(...)`**

uint8.bit_count() -> int

Computes the number of 1-bits in the absolute value of the input.
Analogous to the builtin `int.bit_count` or ``popcount`` in C++.

Examples
--------
>>> np.uint8(127).bit_count()
7

### ufunc
Module: `numpy`

Functions that operate element by element on whole arrays.

To see the documentation for a specific ufunc, use `info`.  For
example, ``np.info(np.sin)``.  Because ufuncs are written in C
(for speed) and linked into Python with NumPy's ufunc facility,
Python's help() function finds this page whenever help() is called
on a ufunc.

A detailed explanation of ufuncs can be found in the docs for :ref:`ufuncs`.

**Calling ufuncs:** ``op(*x[, out], where=True, **kwargs)``

Apply `op` to the arguments `*x` elementwise, broadcasting the arguments.

The broadcasting rules are:

* Dimensions of length 1 may be prepended to either array.
* Arrays may be repeated along dimensions of length 1.

Parameters
----------
*x : array_like
    Input arrays.
out : ndarray, None, or tuple of ndarray and None, optional
    Alternate array object(s) in which to put the result; if provided, it
    must have a shape that the inputs broadcast to. A tuple of arrays
    (possible only as a keyword argument) must have length equal to the
    number of outputs; use None for uninitialized outputs to be
    allocated by the ufunc.
where : array_like, optional
    This condition is broadcast over the input. At locations where the
    condition is True, the `out` array will be set to the ufunc result.
    Elsewhere, the `out` array will retain its original value.
    Note that if an uninitialized `out` array is created via the default
    ``out=None``, locations within it where the condition is False will
    remain uninitialized.
**kwargs
    For other keyword-only arguments, see the :ref:`ufunc docs <ufuncs.kwargs>`.

Returns
-------
r : ndarray or tuple of ndarray
    `r` will have the shape that the arrays in `x` broadcast to; if `out` is
    provided, it will be returned. If not, `r` will be allocated and
    may contain uninitialized values. If the function has more than one
    output, then the result will be a tuple of arrays.

#### Methods

**`reduce(...)`**

reduce(array, axis=0, dtype=None, out=None, keepdims=False, initial=<no value>, where=True)

Reduces `array`'s dimension by one, by applying ufunc along one axis.

Let :math:`array.shape = (N_0, ..., N_i, ..., N_{M-1})`.  Then
:math:`ufunc.reduce(array, axis=i)[k_0, ..,k_{i-1}, k_{i+1}, .., k_{M-1}]` =
the result of iterating `j` over :math:`range(N_i)`, cumulatively applying
ufunc to each :math:`array[k_0, ..,k_{i-1}, j, k_{i+1}, .., k_{M-1}]`.
For a one-dimensional array, reduce produces results equivalent to:
::

r = op.identity # op = ufunc
for i in range(len(A)):
r = op(r, A[i])
return r

For example, add.reduce() is equivalent to sum().

Parameters
----------
array : array_like
The array to act on.
axis : None or int or tuple of ints, optional
Axis or axes along which a reduction is performed.
The default (`axis` = 0) is perform a reduction over the first
dimension of the input array. `axis` may be negative, in
which case it counts from the last to the first axis.

.. versionadded:: 1.7.0

If this is None, a reduction is performed over all the axes.
If this is a tuple of ints, a reduction is performed on multiple
axes, instead of a single axis or all the axes as before.

For operations which are either not commutative or not associative,
doing a reduction over multiple axes is not well-defined. The
ufuncs do not currently raise an exception in this case, but will
likely do so in the future.
dtype : data-type code, optional
The type used to represent the intermediate results. Defaults
to the data-type of the output array if this is provided, or
the data-type of the input array if no output array is provided.
out : ndarray, None, or tuple of ndarray and None, optional
A location into which the result is stored. If not provided or None,
a freshly-allocated array is returned. For consistency with
``ufunc.__call__``, if given as a keyword, this may be wrapped in a
1-element tuple.

.. versionchanged:: 1.13.0
Tuples are allowed for keyword argument.
keepdims : bool, optional
If this is set to True, the axes which are reduced are left
in the result as dimensions with size one. With this option,
the result will broadcast correctly against the original `array`.

.. versionadded:: 1.7.0
initial : scalar, optional
The value with which to start the reduction.
If the ufunc has no identity or the dtype is object, this defaults
to None - otherwise it defaults to ufunc.identity.
If ``None`` is given, the first element of the reduction is used,
and an error is thrown if the reduction is empty.

.. versionadded:: 1.15.0

where : array_like of bool, optional
A boolean array which is broadcasted to match the dimensions
of `array`, and selects elements to include in the reduction. Note
that for ufuncs like ``minimum`` that do not have an identity
defined, one has to pass in also ``initial``.

.. versionadded:: 1.17.0

Returns
-------
r : ndarray
The reduced array. If `out` was supplied, `r` is a reference to it.

Examples
--------
>>> np.multiply.reduce([2,3,5])
30

A multi-dimensional array example:

>>> X = np.arange(8).reshape((2,2,2))
>>> X
array([[[0, 1],
[2, 3]],
[[4, 5],
[6, 7]]])
>>> np.add.reduce(X, 0)
array([[ 4,  6],
[ 8, 10]])
>>> np.add.reduce(X) # confirm: default axis value is 0
array([[ 4,  6],
[ 8, 10]])
>>> np.add.reduce(X, 1)
array([[ 2,  4],
[10, 12]])
>>> np.add.reduce(X, 2)
array([[ 1,  5],
[ 9, 13]])

You can use the ``initial`` keyword argument to initialize the reduction
with a different value, and ``where`` to select specific elements to include:

>>> np.add.reduce([10], initial=5)
15
>>> np.add.reduce(np.ones((2, 2, 2)), axis=(0, 2), initial=10)
array([14., 14.])
>>> a = np.array([10., np.nan, 10])
>>> np.add.reduce(a, where=~np.isnan(a))
20.0

Allows reductions of empty arrays where they would normally fail, i.e.
for ufuncs without an identity.

>>> np.minimum.reduce([], initial=np.inf)
inf
>>> np.minimum.reduce([[1., 2.], [3., 4.]], initial=10., where=[True, False])
array([ 1., 10.])
>>> np.minimum.reduce([])
Traceback (most recent call last):
...
ValueError: zero-size array to reduction operation minimum which has no identity

**`accumulate(...)`**

accumulate(array, axis=0, dtype=None, out=None)

Accumulate the result of applying the operator to all elements.

For a one-dimensional array, accumulate produces results equivalent to::

r = np.empty(len(A))
t = op.identity        # op = the ufunc being applied to A's  elements
for i in range(len(A)):
t = op(t, A[i])
r[i] = t
return r

For example, add.accumulate() is equivalent to np.cumsum().

For a multi-dimensional array, accumulate is applied along only one
axis (axis zero by default; see Examples below) so repeated use is
necessary if one wants to accumulate over multiple axes.

Parameters
----------
array : array_like
The array to act on.
axis : int, optional
The axis along which to apply the accumulation; default is zero.
dtype : data-type code, optional
The data-type used to represent the intermediate results. Defaults
to the data-type of the output array if such is provided, or the
data-type of the input array if no output array is provided.
out : ndarray, None, or tuple of ndarray and None, optional
A location into which the result is stored. If not provided or None,
a freshly-allocated array is returned. For consistency with
``ufunc.__call__``, if given as a keyword, this may be wrapped in a
1-element tuple.

.. versionchanged:: 1.13.0
Tuples are allowed for keyword argument.

Returns
-------
r : ndarray
The accumulated values. If `out` was supplied, `r` is a reference to
`out`.

Examples
--------
1-D array examples:

>>> np.add.accumulate([2, 3, 5])
array([ 2,  5, 10])
>>> np.multiply.accumulate([2, 3, 5])
array([ 2,  6, 30])

2-D array examples:

>>> I = np.eye(2)
>>> I
array([[1.,  0.],
[0.,  1.]])

Accumulate along axis 0 (rows), down columns:

>>> np.add.accumulate(I, 0)
array([[1.,  0.],
[1.,  1.]])
>>> np.add.accumulate(I) # no axis specified = axis zero
array([[1.,  0.],
[1.,  1.]])

Accumulate along axis 1 (columns), through rows:

>>> np.add.accumulate(I, 1)
array([[1.,  1.],
[0.,  1.]])

**`reduceat(...)`**

reduceat(array, indices, axis=0, dtype=None, out=None)

Performs a (local) reduce with specified slices over a single axis.

For i in ``range(len(indices))``, `reduceat` computes
``ufunc.reduce(array[indices[i]:indices[i+1]])``, which becomes the i-th
generalized "row" parallel to `axis` in the final result (i.e., in a
2-D array, for example, if `axis = 0`, it becomes the i-th row, but if
`axis = 1`, it becomes the i-th column).  There are three exceptions to this:

* when ``i = len(indices) - 1`` (so for the last index),
``indices[i+1] = array.shape[axis]``.
* if ``indices[i] >= indices[i + 1]``, the i-th generalized "row" is
simply ``array[indices[i]]``.
* if ``indices[i] >= len(array)`` or ``indices[i] < 0``, an error is raised.

The shape of the output depends on the size of `indices`, and may be
larger than `array` (this happens if ``len(indices) > array.shape[axis]``).

Parameters
----------
array : array_like
The array to act on.
indices : array_like
Paired indices, comma separated (not colon), specifying slices to
reduce.
axis : int, optional
The axis along which to apply the reduceat.
dtype : data-type code, optional
The type used to represent the intermediate results. Defaults
to the data type of the output array if this is provided, or
the data type of the input array if no output array is provided.
out : ndarray, None, or tuple of ndarray and None, optional
A location into which the result is stored. If not provided or None,
a freshly-allocated array is returned. For consistency with
``ufunc.__call__``, if given as a keyword, this may be wrapped in a
1-element tuple.

.. versionchanged:: 1.13.0
Tuples are allowed for keyword argument.

Returns
-------
r : ndarray
The reduced values. If `out` was supplied, `r` is a reference to
`out`.

Notes
-----
A descriptive example:

If `array` is 1-D, the function `ufunc.accumulate(array)` is the same as
``ufunc.reduceat(array, indices)[::2]`` where `indices` is
``range(len(array) - 1)`` with a zero placed
in every other element:
``indices = zeros(2 * len(array) - 1)``,
``indices[1::2] = range(1, len(array))``.

Don't be fooled by this attribute's name: `reduceat(array)` is not
necessarily smaller than `array`.

Examples
--------
To take the running sum of four successive values:

>>> np.add.reduceat(np.arange(8),[0,4, 1,5, 2,6, 3,7])[::2]
array([ 6, 10, 14, 18])

A 2-D example:

>>> x = np.linspace(0, 15, 16).reshape(4,4)
>>> x
array([[ 0.,   1.,   2.,   3.],
[ 4.,   5.,   6.,   7.],
[ 8.,   9.,  10.,  11.],
[12.,  13.,  14.,  15.]])

::

# reduce such that the result has the following five rows:
# [row1 + row2 + row3]
# [row4]
# [row2]
# [row3]
# [row1 + row2 + row3 + row4]

>>> np.add.reduceat(x, [0, 3, 1, 2, 0])
array([[12.,  15.,  18.,  21.],
[12.,  13.,  14.,  15.],
[ 4.,   5.,   6.,   7.],
[ 8.,   9.,  10.,  11.],
[24.,  28.,  32.,  36.]])

::

# reduce such that result has the following two columns:
# [col1 * col2 * col3, col4]

>>> np.multiply.reduceat(x, [0, 3], 1)
array([[   0.,     3.],
[ 120.,     7.],
[ 720.,    11.],
[2184.,    15.]])

**`outer(...)`**

outer(A, B, /, **kwargs)

Apply the ufunc `op` to all pairs (a, b) with a in `A` and b in `B`.

Let ``M = A.ndim``, ``N = B.ndim``. Then the result, `C`, of
``op.outer(A, B)`` is an array of dimension M + N such that:

.. math:: C[i_0, ..., i_{M-1}, j_0, ..., j_{N-1}] =
op(A[i_0, ..., i_{M-1}], B[j_0, ..., j_{N-1}])

For `A` and `B` one-dimensional, this is equivalent to::

r = empty(len(A),len(B))
for i in range(len(A)):
for j in range(len(B)):
r[i,j] = op(A[i], B[j])  # op = ufunc in question

Parameters
----------
A : array_like
First array
B : array_like
Second array
kwargs : any
Arguments to pass on to the ufunc. Typically `dtype` or `out`.
See `ufunc` for a comprehensive overview of all available arguments.

Returns
-------
r : ndarray
Output array

See Also
--------
numpy.outer : A less powerful version of ``np.multiply.outer``
that `ravel`\ s all inputs to 1D. This exists
primarily for compatibility with old code.

tensordot : ``np.tensordot(a, b, axes=((), ()))`` and
``np.multiply.outer(a, b)`` behave same for all
dimensions of a and b.

Examples
--------
>>> np.multiply.outer([1, 2, 3], [4, 5, 6])
array([[ 4,  5,  6],
[ 8, 10, 12],
[12, 15, 18]])

A multi-dimensional example:

>>> A = np.array([[1, 2, 3], [4, 5, 6]])
>>> A.shape
(2, 3)
>>> B = np.array([[1, 2, 3, 4]])
>>> B.shape
(1, 4)
>>> C = np.multiply.outer(A, B)
>>> C.shape; C
(2, 3, 1, 4)
array([[[[ 1,  2,  3,  4]],
[[ 2,  4,  6,  8]],
[[ 3,  6,  9, 12]]],
[[[ 4,  8, 12, 16]],
[[ 5, 10, 15, 20]],
[[ 6, 12, 18, 24]]]])

**`at(...)`**

at(a, indices, b=None, /)

Performs unbuffered in place operation on operand 'a' for elements
specified by 'indices'. For addition ufunc, this method is equivalent to
``a[indices] += b``, except that results are accumulated for elements that
are indexed more than once. For example, ``a[[0,0]] += 1`` will only
increment the first element once because of buffering, whereas
``add.at(a, [0,0], 1)`` will increment the first element twice.

.. versionadded:: 1.8.0

Parameters
----------
a : array_like
The array to perform in place operation on.
indices : array_like or tuple
Array like index object or slice object for indexing into first
operand. If first operand has multiple dimensions, indices can be a
tuple of array like index objects or slice objects.
b : array_like
Second operand for ufuncs requiring two operands. Operand must be
broadcastable over first operand after indexing or slicing.

Examples
--------
Set items 0 and 1 to their negative values:

>>> a = np.array([1, 2, 3, 4])
>>> np.negative.at(a, [0, 1])
>>> a
array([-1, -2,  3,  4])

Increment items 0 and 1, and increment item 2 twice:

>>> a = np.array([1, 2, 3, 4])
>>> np.add.at(a, [0, 1, 2, 2], 1)
>>> a
array([2, 3, 5, 4])

Add items 0 and 1 in first array to second array,
and store results in first array:

>>> a = np.array([1, 2, 3, 4])
>>> b = np.array([1, 2])
>>> np.add.at(a, [0, 1], b)
>>> a
array([2, 4, 3, 4])

**`resolve_dtypes(...)`**

resolve_dtypes(dtypes, *, signature=None, casting=None, reduction=False)

Find the dtypes NumPy will use for the operation.  Both input and
output dtypes are returned and may differ from those provided.

.. note::

This function always applies NEP 50 rules since it is not provided
any actual values.  The Python types ``int``, ``float``, and
``complex`` thus behave weak and should be passed for "untyped"
Python input.

Parameters
----------
dtypes : tuple of dtypes, None, or literal int, float, complex
The input dtypes for each operand.  Output operands can be
None, indicating that the dtype must be found.
signature : tuple of DTypes or None, optional
If given, enforces exact DType (classes) of the specific operand.
The ufunc ``dtype`` argument is equivalent to passing a tuple with
only output dtypes set.
casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
The casting mode when casting is necessary.  This is identical to
the ufunc call casting modes.
reduction : boolean
If given, the resolution assumes a reduce operation is happening
which slightly changes the promotion and type resolution rules.
`dtypes` is usually something like ``(None, np.dtype("i2"), None)``
for reductions (first input is also the output).

.. note::

The default casting mode is "same_kind", however, as of
NumPy 1.24, NumPy uses "unsafe" for reductions.

Returns
-------
dtypes : tuple of dtypes
The dtypes which NumPy would use for the calculation.  Note that
dtypes may not match the passed in ones (casting is necessary).

See Also
--------
numpy.ufunc._resolve_dtypes_and_context :
Similar function to this, but returns additional information which
give access to the core C functionality of NumPy.

Examples
--------
This API requires passing dtypes, define them for convenience:

>>> int32 = np.dtype("int32")
>>> float32 = np.dtype("float32")

The typical ufunc call does not pass an output dtype.  `np.add` has two
inputs and one output, so leave the output as ``None`` (not provided):

>>> np.add.resolve_dtypes((int32, float32, None))
(dtype('float64'), dtype('float64'), dtype('float64'))

The loop found uses "float64" for all operands (including the output), the
first input would be cast.

``resolve_dtypes`` supports "weak" handling for Python scalars by passing
``int``, ``float``, or ``complex``:

>>> np.add.resolve_dtypes((float32, float, None))
(dtype('float32'), dtype('float32'), dtype('float32'))

Where the Python ``float`` behaves samilar to a Python value ``0.0``
in a ufunc call.  (See :ref:`NEP 50 <NEP50>` for details.)

### uint
Module: `numpy`

Unsigned integer type, compatible with C ``unsigned long``.

:Character code: ``'L'``
:Canonical name: `numpy.uint`
:Alias on this platform (win32 AMD64): `numpy.uint32`: 32-bit unsigned integer (``0`` to ``4_294_967_295``).

#### Methods

**`bit_count(...)`**

uint32.bit_count() -> int

Computes the number of 1-bits in the absolute value of the input.
Analogous to the builtin `int.bit_count` or ``popcount`` in C++.

Examples
--------
>>> np.uint32(127).bit_count()
7

### uint16
Module: `numpy`

Unsigned integer type, compatible with C ``unsigned short``.

:Character code: ``'H'``
:Canonical name: `numpy.ushort`
:Alias on this platform (win32 AMD64): `numpy.uint16`: 16-bit unsigned integer (``0`` to ``65_535``).

#### Methods

**`bit_count(...)`**

uint16.bit_count() -> int

Computes the number of 1-bits in the absolute value of the input.
Analogous to the builtin `int.bit_count` or ``popcount`` in C++.

Examples
--------
>>> np.uint16(127).bit_count()
7

### uint32
Module: `numpy`

Unsigned integer type, compatible with C ``unsigned long``.

:Character code: ``'L'``
:Canonical name: `numpy.uint`
:Alias on this platform (win32 AMD64): `numpy.uint32`: 32-bit unsigned integer (``0`` to ``4_294_967_295``).

#### Methods

**`bit_count(...)`**

uint32.bit_count() -> int

Computes the number of 1-bits in the absolute value of the input.
Analogous to the builtin `int.bit_count` or ``popcount`` in C++.

Examples
--------
>>> np.uint32(127).bit_count()
7

### uint64
Module: `numpy`

Signed integer type, compatible with C ``unsigned long long``.

:Character code: ``'Q'``
:Canonical name: `numpy.ulonglong`
:Alias on this platform (win32 AMD64): `numpy.uint64`: 64-bit unsigned integer (``0`` to ``18_446_744_073_709_551_615``).
:Alias on this platform (win32 AMD64): `numpy.uintp`: Unsigned integer large enough to fit pointer, compatible with C ``uintptr_t``.

#### Methods

**`bit_count(...)`**

uint64.bit_count() -> int

Computes the number of 1-bits in the absolute value of the input.
Analogous to the builtin `int.bit_count` or ``popcount`` in C++.

Examples
--------
>>> np.uint64(127).bit_count()
7

### uint8
Module: `numpy`

Unsigned integer type, compatible with C ``unsigned char``.

:Character code: ``'B'``
:Canonical name: `numpy.ubyte`
:Alias on this platform (win32 AMD64): `numpy.uint8`: 8-bit unsigned integer (``0`` to ``255``).

#### Methods

**`bit_count(...)`**

uint8.bit_count() -> int

Computes the number of 1-bits in the absolute value of the input.
Analogous to the builtin `int.bit_count` or ``popcount`` in C++.

Examples
--------
>>> np.uint8(127).bit_count()
7

### uintc
Module: `numpy`

Unsigned integer type, compatible with C ``unsigned int``.

:Character code: ``'I'``

#### Methods

**`bit_count(...)`**

*No documentation available.*

### uintp
Module: `numpy`

Signed integer type, compatible with C ``unsigned long long``.

:Character code: ``'Q'``
:Canonical name: `numpy.ulonglong`
:Alias on this platform (win32 AMD64): `numpy.uint64`: 64-bit unsigned integer (``0`` to ``18_446_744_073_709_551_615``).
:Alias on this platform (win32 AMD64): `numpy.uintp`: Unsigned integer large enough to fit pointer, compatible with C ``uintptr_t``.

#### Methods

**`bit_count(...)`**

uint64.bit_count() -> int

Computes the number of 1-bits in the absolute value of the input.
Analogous to the builtin `int.bit_count` or ``popcount`` in C++.

Examples
--------
>>> np.uint64(127).bit_count()
7

### ulonglong
Module: `numpy`

Signed integer type, compatible with C ``unsigned long long``.

:Character code: ``'Q'``
:Canonical name: `numpy.ulonglong`
:Alias on this platform (win32 AMD64): `numpy.uint64`: 64-bit unsigned integer (``0`` to ``18_446_744_073_709_551_615``).
:Alias on this platform (win32 AMD64): `numpy.uintp`: Unsigned integer large enough to fit pointer, compatible with C ``uintptr_t``.

#### Methods

**`bit_count(...)`**

uint64.bit_count() -> int

Computes the number of 1-bits in the absolute value of the input.
Analogous to the builtin `int.bit_count` or ``popcount`` in C++.

Examples
--------
>>> np.uint64(127).bit_count()
7

### unicode_
Module: `numpy`

A unicode string.

This type strips trailing null codepoints.

>>> s = np.str_("abc\x00")
>>> s
'abc'

Unlike the builtin `str`, this supports the :ref:`python:bufferobjects`, exposing its
contents as UCS4:

>>> m = memoryview(np.str_("abc"))
>>> m.format
'3w'
>>> m.tobytes()
b'a\x00\x00\x00b\x00\x00\x00c\x00\x00\x00'

:Character code: ``'U'``
:Alias: `numpy.unicode_`

### unsignedinteger
Module: `numpy`

Abstract base class of all unsigned integer scalar types.

### ushort
Module: `numpy`

Unsigned integer type, compatible with C ``unsigned short``.

:Character code: ``'H'``
:Canonical name: `numpy.ushort`
:Alias on this platform (win32 AMD64): `numpy.uint16`: 16-bit unsigned integer (``0`` to ``65_535``).

#### Methods

**`bit_count(...)`**

uint16.bit_count() -> int

Computes the number of 1-bits in the absolute value of the input.
Analogous to the builtin `int.bit_count` or ``popcount`` in C++.

Examples
--------
>>> np.uint16(127).bit_count()
7

### vectorize
Module: `numpy`

vectorize(pyfunc=np._NoValue, otypes=None, doc=None, excluded=None,
cache=False, signature=None)

Returns an object that acts like pyfunc, but takes arrays as input.

Define a vectorized function which takes a nested sequence of objects or
numpy arrays as inputs and returns a single numpy array or a tuple of numpy
arrays. The vectorized function evaluates `pyfunc` over successive tuples
of the input arrays like the python map function, except it uses the
broadcasting rules of numpy.

The data type of the output of `vectorized` is determined by calling
the function with the first element of the input.  This can be avoided
by specifying the `otypes` argument.

Parameters
----------
pyfunc : callable, optional
    A python function or method.
    Can be omitted to produce a decorator with keyword arguments.
otypes : str or list of dtypes, optional
    The output data type. It must be specified as either a string of
    typecode characters or a list of data type specifiers. There should
    be one data type specifier for each output.
doc : str, optional
    The docstring for the function. If None, the docstring will be the
    ``pyfunc.__doc__``.
excluded : set, optional
    Set of strings or integers representing the positional or keyword
    arguments for which the function will not be vectorized.  These will be
    passed directly to `pyfunc` unmodified.

    .. versionadded:: 1.7.0

cache : bool, optional
    If `True`, then cache the first function call that determines the number
    of outputs if `otypes` is not provided.

    .. versionadded:: 1.7.0

signature : string, optional
    Generalized universal function signature, e.g., ``(m,n),(n)->(m)`` for
    vectorized matrix-vector multiplication. If provided, ``pyfunc`` will
    be called with (and expected to return) arrays with shapes given by the
    size of corresponding core dimensions. By default, ``pyfunc`` is
    assumed to take scalars as input and output.

    .. versionadded:: 1.12.0

Returns
-------
out : callable
    A vectorized function if ``pyfunc`` was provided,
    a decorator otherwise.

See Also
--------
frompyfunc : Takes an arbitrary Python function and returns a ufunc

Notes
-----
The `vectorize` function is provided primarily for convenience, not for
performance. The implementation is essentially a for loop.

If `otypes` is not specified, then a call to the function with the
first argument will be used to determine the number of outputs.  The
results of this call will be cached if `cache` is `True` to prevent
calling the function twice.  However, to implement the cache, the
original function must be wrapped which will slow down subsequent
calls, so only do this if your function is expensive.

The new keyword argument interface and `excluded` argument support
further degrades performance.

References
----------
.. [1] :doc:`/reference/c-api/generalized-ufuncs`

Examples
--------
>>> def myfunc(a, b):
...     "Return a-b if a>b, otherwise return a+b"
...     if a > b:
...         return a - b
...     else:
...         return a + b

>>> vfunc = np.vectorize(myfunc)
>>> vfunc([1, 2, 3, 4], 2)
array([3, 4, 1, 2])

The docstring is taken from the input function to `vectorize` unless it
is specified:

>>> vfunc.__doc__
'Return a-b if a>b, otherwise return a+b'
>>> vfunc = np.vectorize(myfunc, doc='Vectorized `myfunc`')
>>> vfunc.__doc__
'Vectorized `myfunc`'

The output type is determined by evaluating the first element of the input,
unless it is specified:

>>> out = vfunc([1, 2, 3, 4], 2)
>>> type(out[0])
<class 'numpy.int64'>
>>> vfunc = np.vectorize(myfunc, otypes=[float])
>>> out = vfunc([1, 2, 3, 4], 2)
>>> type(out[0])
<class 'numpy.float64'>

The `excluded` argument can be used to prevent vectorizing over certain
arguments.  This can be useful for array-like arguments of a fixed length
such as the coefficients for a polynomial as in `polyval`:

>>> def mypolyval(p, x):
...     _p = list(p)
...     res = _p.pop(0)
...     while _p:
...         res = res*x + _p.pop(0)
...     return res
>>> vpolyval = np.vectorize(mypolyval, excluded=['p'])
>>> vpolyval(p=[1, 2, 3], x=[0, 1])
array([3, 6])

Positional arguments may also be excluded by specifying their position:

>>> vpolyval.excluded.add(0)
>>> vpolyval([1, 2, 3], x=[0, 1])
array([3, 6])

The `signature` argument allows for vectorizing functions that act on
non-scalar arrays of fixed length. For example, you can use it for a
vectorized calculation of Pearson correlation coefficient and its p-value:

>>> import scipy.stats
>>> pearsonr = np.vectorize(scipy.stats.pearsonr,
...                 signature='(n),(n)->(),()')
>>> pearsonr([[0, 1, 2, 3]], [[1, 2, 3, 4], [4, 3, 2, 1]])
(array([ 1., -1.]), array([ 0.,  0.]))

Or for a vectorized convolution:

>>> convolve = np.vectorize(np.convolve, signature='(n),(m)->(k)')
>>> convolve(np.eye(4), [1, 2, 1])
array([[1., 2., 1., 0., 0., 0.],
       [0., 1., 2., 1., 0., 0.],
       [0., 0., 1., 2., 1., 0.],
       [0., 0., 0., 1., 2., 1.]])

Decorator syntax is supported.  The decorator can be called as
a function to provide keyword arguments.
>>>@np.vectorize
...def identity(x):
...    return x
...
>>>identity([0, 1, 2])
array([0, 1, 2])
>>>@np.vectorize(otypes=[float])
...def as_float(x):
...    return x
...
>>>as_float([0, 1, 2])
array([0., 1., 2.])

### void
Module: `numpy`

np.void(length_or_data, /, dtype=None)

Create a new structured or unstructured void scalar.

Parameters
----------
length_or_data : int, array-like, bytes-like, object
   One of multiple meanings (see notes).  The length or
   bytes data of an unstructured void.  Or alternatively,
   the data to be stored in the new scalar when `dtype`
   is provided.
   This can be an array-like, in which case an array may
   be returned.
dtype : dtype, optional
    If provided the dtype of the new scalar.  This dtype must
    be "void" dtype (i.e. a structured or unstructured void,
    see also :ref:`defining-structured-types`).

   ..versionadded:: 1.24

Notes
-----
For historical reasons and because void scalars can represent both
arbitrary byte data and structured dtypes, the void constructor
has three calling conventions:

1. ``np.void(5)`` creates a ``dtype="V5"`` scalar filled with five
   ``\0`` bytes.  The 5 can be a Python or NumPy integer.
2. ``np.void(b"bytes-like")`` creates a void scalar from the byte string.
   The dtype itemsize will match the byte string length, here ``"V10"``.
3. When a ``dtype=`` is passed the call is roughly the same as an
   array creation.  However, a void scalar rather than array is returned.

Please see the examples which show all three different conventions.

Examples
--------
>>> np.void(5)
void(b'\x00\x00\x00\x00\x00')
>>> np.void(b'abcd')
void(b'\x61\x62\x63\x64')
>>> np.void((5, 3.2, "eggs"), dtype="i,d,S5")
(5, 3.2, b'eggs')  # looks like a tuple, but is `np.void`
>>> np.void(3, dtype=[('x', np.int8), ('y', np.int8)])
(3, 3)  # looks like a tuple, but is `np.void`

:Character code: ``'V'``

#### Methods

**`getfield(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.getfield`.

**`setfield(...)`**

Scalar method identical to the corresponding array attribute.

Please see `ndarray.setfield`.
