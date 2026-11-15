"""The two arithmetic paths used by the Ising formulas and spectrum searches."""

import math
import operator

import numpy as np


class Arithmetic:
    """NumPy doubles by default; an isolated mpmath context when dps is supplied.

    Never change mpmath's global context. Convert incoming numbers to this context
    before arithmetic, including numbers returned by a previous higher-precision call.
    Arrays remain NumPy arrays in both paths; arbitrary-precision values use object dtype.
    """

    def __init__(self, dps=None):
        if dps is not None:
            if isinstance(dps, (bool, np.bool_)):
                raise ValueError("dps must be an integer of at least 2, or None")
            try:
                dps = operator.index(dps)
            except TypeError:
                raise ValueError("dps must be an integer of at least 2, or None") from None
            if dps < 2:
                raise ValueError("dps must be an integer of at least 2, or None")
        self.dps = dps
        self.dtype = float if dps is None else object
        self.context = None
        if dps is None:
            self._number = float
            self.pi = np.pi
            self.inf = np.inf
            self.fsum = math.fsum
        else:
            try:
                from mpmath import mp
            except ImportError:
                raise ImportError('dps requires mpmath; install "exact-solutions[precision]"') from None
            self.context = mp.clone()
            self.context.dps = dps
            self._number = self.context.mpf
            self.pi = +self.context.pi
            self.inf = self.context.inf
            self.fsum = self.context.fsum

    def number(self, value):
        if isinstance(value, np.generic):
            value = value.item()
        return self._number(value)

    def values(self, value):
        """Convert a scalar or array without discarding its shape."""
        if np.ndim(value) == 0:
            if isinstance(value, np.ndarray):
                value = value.item()
            return self.number(value)
        return self.array(value)

    def array(self, values):
        # Object dtype must be selected before NumPy infers a common scalar type:
        # a mixed list of Python floats and large integers must not round to double.
        values = np.asarray(values, dtype=self.dtype)
        if self.context is None:
            return values
        return np.array([self.number(v) for v in values.flat], dtype=object).reshape(values.shape)

    def _unary(self, numpy_name, mp_name, value):
        if self.context is None:
            return getattr(np, numpy_name)(value)
        function = getattr(self.context, mp_name)
        if np.ndim(value) == 0:
            if isinstance(value, np.ndarray):
                value = value.item()
            return function(value)
        values = np.asarray(value)
        dtype = bool if mp_name == "isfinite" else object
        return np.array([function(v) for v in values.flat], dtype=dtype).reshape(values.shape)

    def exp(self, value):
        return self._unary("exp", "exp", value)

    def expm1(self, value):
        return self._unary("expm1", "expm1", value)

    def log(self, value):
        return self._unary("log", "log", value)

    def log1p(self, value):
        return self._unary("log1p", "log1p", value)

    def sqrt(self, value):
        return self._unary("sqrt", "sqrt", value)

    def sin(self, value):
        return self._unary("sin", "sin", value)

    def sinh(self, value):
        return self._unary("sinh", "sinh", value)

    def tanh(self, value):
        return self._unary("tanh", "tanh", value)

    def asinh(self, value):
        return self._unary("arcsinh", "asinh", value)

    def isfinite(self, value):
        return self._unary("isfinite", "isfinite", value)
