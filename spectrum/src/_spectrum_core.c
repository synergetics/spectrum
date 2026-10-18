/* _spectrum_core.c - CPython C-API binding for the spectrum C core.
 *
 * One flat function per estimator; kernels live in spectrum/src/*.c and are
 * called with the GIL released. All arrays are contiguous numpy arrays:
 * complex estimators operate on complex128 (real float64 input is cast up),
 * real-only estimators on float64.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <complex.h>
#include <pthread.h>

#include "common.h"
#include "fft/fft_iface.h"
#include "estimators_cum.h"
#include "estimators_bispec.h"
#include "wigner.h"
#include "estimators_ext.h"

/* module-level one-time init flag for the fft engine (nothing needed yet) */

static PyObject *g_selftest_err = NULL;

/* ---- helpers ------------------------------------------------------- */

static PyArrayObject *py_to_c128(PyObject *obj) {
    return (PyArrayObject *)PyArray_FROM_OTF(obj, NPY_CDOUBLE,
                                             NPY_ARRAY_IN_ARRAY);
}

static PyArrayObject *py_to_f64(PyObject *obj) {
    return (PyArrayObject *)PyArray_FROM_OTF(obj, NPY_DOUBLE,
                                             NPY_ARRAY_IN_ARRAY);
}

static PyArrayObject *py_new_c128(int ndim, const npy_intp *dims) {
    return (PyArrayObject *)PyArray_SimpleNew(ndim, dims, NPY_CDOUBLE);
}

static PyArrayObject *py_new_f64(int ndim, const npy_intp *dims) {
    return (PyArrayObject *)PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
}

/* ---- cumulant bindings --------------------------------------------- */

/* Coerce to a Fortran-order complex128 array (== numpy ravel(order="F")
 * view of the 2-D input). */
static PyArrayObject *py_to_c128_f(PyObject *obj) {
    return (PyArrayObject *)PyArray_FROM_OTF(
        obj, NPY_CDOUBLE,
        NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_ALIGNED |
            NPY_ARRAY_FORCECAST);
}

static int flag_biased(const char *flag) {
    /* low-level estimators: anything != "biased" is unbiased (parity with
     * the Python reference); cumest normalizes flags before dispatch. */
    return (flag && strcmp(flag, "biased") == 0) ? 1 : 0;
}

static PyObject *py_cum2est(PyObject *self, PyObject *args) {
    PyObject *yo;
    int maxlag, overlap;
    Py_ssize_t nsamp;
    const char *flag;
    PyArrayObject *ya, *outa;
    npy_intp out_dims[1];
    sp_status st;
    (void)self;
    if (!PyArg_ParseTuple(args, "Oinis", &yo, &maxlag, &nsamp, &overlap, &flag))
        return NULL;
    ya = py_to_c128_f(yo);
    if (!ya) return NULL;
    out_dims[0] = (npy_intp)(2 * maxlag + 1);
    outa = py_new_c128(2, (npy_intp[]){out_dims[0], 1});
    if (!outa) {
        Py_DECREF(ya);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    st = sp_cum2est((const sp_c128 *)PyArray_DATA(ya),
                   (size_t)PyArray_SIZE(ya), maxlag, (size_t)nsamp, overlap,
                   flag_biased(flag), (sp_c128 *)PyArray_DATA(outa));
    Py_END_ALLOW_THREADS
    Py_DECREF(ya);
    if (st != SP_OK) {
        Py_DECREF(outa);
        return PyErr_Format(PyExc_ValueError, "cum2est kernel error %d", (int)st);
    }
    return (PyObject *)outa;
}

static PyObject *py_cum2x(PyObject *self, PyObject *args) {
    PyObject *xo, *yo;
    int maxlag, overlap;
    Py_ssize_t nsamp;
    const char *flag;
    PyArrayObject *xa, *ya, *outa;
    npy_intp cols;
    sp_status st;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOinis", &xo, &yo, &maxlag, &nsamp, &overlap,
                          &flag))
        return NULL;
    xa = py_to_c128_f(xo);
    ya = py_to_c128_f(yo);
    if (!xa || !ya) {
        Py_XDECREF(xa);
        Py_XDECREF(ya);
        return NULL;
    }
    cols = (PyArray_NDIM(xa) == 2) ? PyArray_DIM(xa, 1) : 1;
    outa = py_new_c128(2, (npy_intp[]){(npy_intp)(2 * maxlag + 1), 1});
    if (!outa) {
        Py_DECREF(xa);
        Py_DECREF(ya);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    st = sp_cum2x((const sp_c128 *)PyArray_DATA(xa),
                  (const sp_c128 *)PyArray_DATA(ya),
                  (size_t)PyArray_SIZE(xa), (size_t)cols, maxlag,
                  (size_t)nsamp, overlap, flag_biased(flag),
                  (sp_c128 *)PyArray_DATA(outa));
    Py_END_ALLOW_THREADS
    Py_DECREF(xa);
    Py_DECREF(ya);
    if (st != SP_OK) {
        Py_DECREF(outa);
        return PyErr_Format(PyExc_ValueError, "cum2x kernel error %d", (int)st);
    }
    return (PyObject *)outa;
}

static PyObject *py_cum3est(PyObject *self, PyObject *args) {
    PyObject *yo;
    int maxlag, overlap, k1;
    Py_ssize_t nsamp;
    const char *flag;
    PyArrayObject *ya, *outa;
    sp_status st;
    (void)self;
    if (!PyArg_ParseTuple(args, "Oinisi", &yo, &maxlag, &nsamp, &overlap,
                          &flag, &k1))
        return NULL;
    ya = py_to_c128_f(yo);
    if (!ya) return NULL;
    outa = py_new_c128(2, (npy_intp[]){(npy_intp)(2 * maxlag + 1), 1});
    if (!outa) {
        Py_DECREF(ya);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    st = sp_cum3est((const sp_c128 *)PyArray_DATA(ya),
                    (size_t)PyArray_SIZE(ya), maxlag, (size_t)nsamp, overlap,
                    flag_biased(flag), k1, (sp_c128 *)PyArray_DATA(outa));
    Py_END_ALLOW_THREADS
    Py_DECREF(ya);
    if (st != SP_OK) {
        Py_DECREF(outa);
        return PyErr_Format(PyExc_ValueError, "cum3est kernel error %d", (int)st);
    }
    return (PyObject *)outa;
}

static PyObject *py_cum3x(PyObject *self, PyObject *args) {
    PyObject *xo, *yo, *zo;
    int maxlag, overlap, k1;
    Py_ssize_t nsamp;
    const char *flag;
    PyArrayObject *xa, *ya, *za, *outa;
    npy_intp cols;
    sp_status st;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOiiisi", &xo, &yo, &zo, &maxlag, &nsamp,
                          &overlap, &flag, &k1))
        return NULL;
    xa = py_to_c128_f(xo);
    ya = py_to_c128_f(yo);
    za = py_to_c128_f(zo);
    if (!xa || !ya || !za) {
        Py_XDECREF(xa);
        Py_XDECREF(ya);
        Py_XDECREF(za);
        return NULL;
    }
    cols = (PyArray_NDIM(xa) == 2) ? PyArray_DIM(xa, 1) : 1;
    outa = py_new_c128(2, (npy_intp[]){(npy_intp)(2 * maxlag + 1), 1});
    if (!outa) {
        Py_DECREF(xa);
        Py_DECREF(ya);
        Py_DECREF(za);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    st = sp_cum3x((const sp_c128 *)PyArray_DATA(xa),
                  (const sp_c128 *)PyArray_DATA(ya),
                  (const sp_c128 *)PyArray_DATA(za), (size_t)PyArray_SIZE(xa),
                  (size_t)cols, maxlag, (size_t)nsamp, overlap,
                  flag_biased(flag), k1, (sp_c128 *)PyArray_DATA(outa));
    Py_END_ALLOW_THREADS
    Py_DECREF(xa);
    Py_DECREF(ya);
    Py_DECREF(za);
    if (st != SP_OK) {
        Py_DECREF(outa);
        return PyErr_Format(PyExc_ValueError, "cum3x kernel error %d", (int)st);
    }
    return (PyObject *)outa;
}

static PyObject *py_cum4est(PyObject *self, PyObject *args) {
    PyObject *yo;
    int maxlag, overlap, k1, k2;
    Py_ssize_t nsamp;
    const char *flag;
    PyArrayObject *ya, *outa;
    sp_status st;
    (void)self;
    if (!PyArg_ParseTuple(args, "Oinisii", &yo, &maxlag, &nsamp, &overlap,
                          &flag, &k1, &k2))
        return NULL;
    ya = py_to_c128_f(yo);
    if (!ya) return NULL;
    outa = py_new_c128(2, (npy_intp[]){(npy_intp)(2 * maxlag + 1), 1});
    if (!outa) {
        Py_DECREF(ya);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    st = sp_cum4est((const sp_c128 *)PyArray_DATA(ya),
                    (size_t)PyArray_SIZE(ya), maxlag, (size_t)nsamp, overlap,
                    flag_biased(flag), k1, k2, (sp_c128 *)PyArray_DATA(outa));
    Py_END_ALLOW_THREADS
    Py_DECREF(ya);
    if (st != SP_OK) {
        Py_DECREF(outa);
        return PyErr_Format(PyExc_ValueError, "cum4est kernel error %d", (int)st);
    }
    return (PyObject *)outa;
}

static PyObject *py_cum4x(PyObject *self, PyObject *args) {
    PyObject *wo, *xo, *yo, *zo;
    int maxlag, overlap, k1, k2;
    Py_ssize_t nsamp;
    const char *flag;
    PyArrayObject *wa, *xa, *ya, *za, *outa;
    npy_intp cols;
    sp_status st;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOinisii", &wo, &xo, &yo, &zo, &maxlag,
                          &nsamp, &overlap, &flag, &k1, &k2))
        return NULL;
    wa = py_to_c128_f(wo);
    xa = py_to_c128_f(xo);
    ya = py_to_c128_f(yo);
    za = py_to_c128_f(zo);
    if (!wa || !xa || !ya || !za) {
        Py_XDECREF(wa);
        Py_XDECREF(xa);
        Py_XDECREF(ya);
        Py_XDECREF(za);
        return NULL;
    }
    cols = (PyArray_NDIM(wa) == 2) ? PyArray_DIM(wa, 1) : 1;
    outa = py_new_c128(2, (npy_intp[]){(npy_intp)(2 * maxlag + 1), 1});
    if (!outa) {
        Py_DECREF(wa);
        Py_DECREF(xa);
        Py_DECREF(ya);
        Py_DECREF(za);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    st = sp_cum4x((const sp_c128 *)PyArray_DATA(wa),
                  (const sp_c128 *)PyArray_DATA(xa),
                  (const sp_c128 *)PyArray_DATA(ya),
                  (const sp_c128 *)PyArray_DATA(za), (size_t)PyArray_SIZE(wa),
                  (size_t)cols, maxlag, (size_t)nsamp, overlap,
                  flag_biased(flag), k1, k2, (sp_c128 *)PyArray_DATA(outa));
    Py_END_ALLOW_THREADS
    Py_DECREF(wa);
    Py_DECREF(xa);
    Py_DECREF(ya);
    Py_DECREF(za);
    if (st != SP_OK) {
        Py_DECREF(outa);
        return PyErr_Format(PyExc_ValueError, "cum4x kernel error %d", (int)st);
    }
    return (PyObject *)outa;
}

/* ---- bispectrum-family bindings ------------------------------------ */

static int parse_window_name(PyObject *wind, const char *func_name) {
    /* returns window_type for trispectrum-family, -1 on error */
    if (wind == Py_None) return 0;
    const char *s = PyUnicode_AsUTF8(wind);
    if (!s) {
        PyErr_SetString(PyExc_TypeError, "window must be a string or None");
        return -1;
    }
    if (strcmp(s, "rect") == 0) return 0;
    if (strcmp(s, "hanning") == 0) return 1;
    if (strcmp(s, "hamming") == 0) return 2;
    if (strcmp(s, "blackman") == 0) return 3;
    PyErr_Format(PyExc_ValueError, "Unknown window type: %s", s);
    if (func_name) {
    }
    return -1;
}

/* Coerce an optional 1-D real window to a contiguous double array. */
static PyArrayObject *py_wind_to_f64(PyObject *obj) {
    return (PyArrayObject *)PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
}

static PyObject *py_bicoher_generic(PyObject *self, PyObject *args, int cross) {
    PyObject *yo = NULL, *wo = NULL, *xo = NULL, *zo = NULL;
    long nfft_in;
    PyObject *wind_obj;
    long nsamp_in;
    int overlap;
    PyArrayObject *wa = NULL, *xa = NULL, *ya = NULL, *za = NULL;
    PyArrayObject *winda = NULL, *outa = NULL, *waxa = NULL;
    size_t nsamp, nfft, N, cols;
    sp_status st;
    (void)self;

    if (!cross) {
        if (!PyArg_ParseTuple(args, "OlOli", &yo, &nfft_in, &wind_obj,
                              &nsamp_in, &overlap))
            return NULL;
        wo = yo;
        xo = yo;
        zo = yo;
    } else {
        if (!PyArg_ParseTuple(args, "OOOlOli", &wo, &xo, &yo, &nfft_in,
                              &wind_obj, &nsamp_in, &overlap))
            return NULL;
        zo = yo;
    }

    wa = py_to_c128_f(wo);
    xa = py_to_c128_f(xo);
    ya = py_to_c128_f(yo);
    za = py_to_c128_f(zo);
    if (!wa || !xa || !ya || !za) {
        Py_XDECREF(wa);
        Py_XDECREF(xa);
        Py_XDECREF(ya);
        Py_XDECREF(za);
        return NULL;
    }
    N = (size_t)PyArray_SIZE(wa);
    cols = (PyArray_NDIM(wa) == 2) ? (size_t)PyArray_DIM(wa, 1) : 1;
    if (overlap < 0) overlap = 0;
    if (overlap > 99) overlap = 99;
    st = sp_bispec_meta(N / (cols ? cols : 1), cols, nsamp_in, overlap,
                        nfft_in, &nsamp, &nfft);
    if (st != SP_OK) {
        Py_DECREF(wa);
        Py_DECREF(xa);
        Py_DECREF(ya);
        Py_DECREF(za);
        return PyErr_Format(PyExc_ValueError, "invalid segmentation (status %d)",
                            (int)st);
    }
    if (wind_obj != Py_None) {
        winda = py_wind_to_f64(wind_obj);
        if (!winda) {
            Py_DECREF(wa);
            Py_DECREF(xa);
            Py_DECREF(ya);
            Py_DECREF(za);
            return NULL;
        }
        if ((size_t)PyArray_SIZE(winda) != nsamp) {
            /* wrong-size window: fall back to the default hanning */
            Py_DECREF(winda);
            winda = NULL;
        }
    }
    outa = py_new_c128(2, (npy_intp[]){(npy_intp)nfft, (npy_intp)nfft});
    waxa = py_new_f64(1, (npy_intp[]){(npy_intp)nfft});
    if (!outa || !waxa) {
        Py_XDECREF(outa);
        Py_XDECREF(waxa);
        Py_XDECREF(winda);
        Py_DECREF(wa);
        Py_DECREF(xa);
        Py_DECREF(ya);
        Py_DECREF(za);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    st = sp_bicoherx((const sp_c128 *)PyArray_DATA(wa),
                     (const sp_c128 *)PyArray_DATA(xa),
                     (const sp_c128 *)PyArray_DATA(ya), N, cols, nsamp, nfft,
                     winda ? (const double *)PyArray_DATA(winda) : NULL,
                     overlap, cross ? 0 : 1, (sp_c128 *)PyArray_DATA(outa),
                     (double *)PyArray_DATA(waxa));
    Py_END_ALLOW_THREADS
    Py_XDECREF(winda);
    Py_DECREF(wa);
    Py_DECREF(xa);
    Py_DECREF(ya);
    Py_DECREF(za);
    if (st != SP_OK) {
        Py_DECREF(outa);
        Py_DECREF(waxa);
        return PyErr_Format(PyExc_ValueError, "bicoherence kernel error %d",
                            (int)st);
    }
    return Py_BuildValue("NN", (PyObject *)outa, (PyObject *)waxa);
}

static PyObject *py_bicoherence(PyObject *self, PyObject *args) {
    return py_bicoher_generic(self, args, 0);
}

static PyObject *py_bicoherencex(PyObject *self, PyObject *args) {
    return py_bicoher_generic(self, args, 1);
}

static PyObject *py_bispecd_generic(PyObject *self, PyObject *args, int cross) {
    PyObject *xo = NULL, *yo = NULL, *zo = NULL;
    long nfft_in;
    PyObject *wind_obj;
    long nsamp_in;
    int overlap;
    PyArrayObject *xa = NULL, *ya = NULL, *za = NULL, *wa = NULL;
    PyArrayObject *winda = NULL, *mata = NULL, *outa = NULL, *waxa = NULL;
    size_t nsamp, nfft, N, cols;
    sp_wind_spec spec;
    sp_status st;
    (void)self;

    if (!cross) {
        if (!PyArg_ParseTuple(args, "OlOli", &xo, &nfft_in, &wind_obj,
                              &nsamp_in, &overlap))
            return NULL;
        yo = xo;
        zo = xo;
    } else {
        if (!PyArg_ParseTuple(args, "OOOlOli", &xo, &yo, &zo, &nfft_in,
                              &wind_obj, &nsamp_in, &overlap))
            return NULL;
    }

    xa = py_to_c128_f(xo);
    ya = py_to_c128_f(yo);
    za = py_to_c128_f(zo);
    if (!xa || !ya || !za) {
        Py_XDECREF(xa);
        Py_XDECREF(ya);
        Py_XDECREF(za);
        return NULL;
    }
    N = (size_t)PyArray_SIZE(xa);
    cols = (PyArray_NDIM(xa) == 2) ? (size_t)PyArray_DIM(xa, 1) : 1;
    if (overlap < 0) overlap = 0;
    if (overlap > 99) overlap = 99;
    st = sp_bispec_meta(N / (cols ? cols : 1), cols, nsamp_in, overlap,
                        nfft_in, &nsamp, &nfft);
    if (st != SP_OK) {
        Py_DECREF(xa);
        Py_DECREF(ya);
        Py_DECREF(za);
        return PyErr_Format(PyExc_ValueError, "invalid segmentation (status %d)",
                            (int)st);
    }

    memset(&spec, 0, sizeof(spec));
    spec.kind = 0;
    spec.opt_size = 5;
    if (wind_obj != Py_None) {
        if (PyLong_Check(wind_obj)) {
            spec.kind = 0;
            spec.opt_size = (int)PyLong_AsLong(wind_obj);
        } else if (PyArray_Check(wind_obj)) {
            PyArrayObject *wa2 = (PyArrayObject *)wind_obj;
            if (PyArray_NDIM(wa2) == 1) {
                spec.kind = 1;
                winda = py_wind_to_f64(wind_obj);
                if (!winda) {
                    Py_DECREF(xa);
                    Py_DECREF(ya);
                    Py_DECREF(za);
                    return NULL;
                }
                spec.vec = (const double *)PyArray_DATA(winda);
                spec.vec_len = (long)PyArray_SIZE(winda);
            } else if (PyArray_NDIM(wa2) == 2 &&
                       PyArray_DIM(wa2, 0) == PyArray_DIM(wa2, 1) &&
                       PyArray_DIM(wa2, 0) % 2 == 1) {
                spec.kind = 2;
                mata = py_wind_to_f64(wind_obj);
                if (!mata) {
                    Py_DECREF(xa);
                    Py_DECREF(ya);
                    Py_DECREF(za);
                    return NULL;
                }
                spec.mat = (const double *)PyArray_DATA(mata);
                spec.mat_dim = (long)PyArray_DIM(mata, 0);
            } else if (PyArray_NDIM(wa2) == 2) {
                /* invalid 2-D window: warn + skip smoothing */
                spec.kind = 0;
                spec.opt_size = 1;
            }
        } else {
            Py_DECREF(xa);
            Py_DECREF(ya);
            Py_DECREF(za);
            PyErr_SetString(PyExc_TypeError, "wind must be an int or array");
            return NULL;
        }
    }

    outa = py_new_c128(2, (npy_intp[]){(npy_intp)nfft, (npy_intp)nfft});
    waxa = py_new_f64(1, (npy_intp[]){(npy_intp)nfft});
    if (!outa || !waxa) {
        Py_XDECREF(outa);
        Py_XDECREF(waxa);
        Py_XDECREF(winda);
        Py_XDECREF(mata);
        Py_DECREF(xa);
        Py_DECREF(ya);
        Py_DECREF(za);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    st = sp_bispecdx((const sp_c128 *)PyArray_DATA(xa),
                     (const sp_c128 *)PyArray_DATA(ya),
                     (const sp_c128 *)PyArray_DATA(za), N, cols, nsamp, nfft,
                     &spec, overlap, cross ? 0 : 1,
                     (sp_c128 *)PyArray_DATA(outa),
                     (double *)PyArray_DATA(waxa));
    Py_END_ALLOW_THREADS
    Py_XDECREF(winda);
    Py_XDECREF(mata);
    Py_DECREF(xa);
    Py_DECREF(ya);
    Py_DECREF(za);
    Py_XDECREF(wa);
    if (st != SP_OK) {
        Py_DECREF(outa);
        Py_DECREF(waxa);
        return PyErr_Format(PyExc_ValueError, "bispectrum kernel error %d",
                            (int)st);
    }
    return Py_BuildValue("NN", (PyObject *)outa, (PyObject *)waxa);
}

static PyObject *py_bispectrumd(PyObject *self, PyObject *args) {
    return py_bispecd_generic(self, args, 0);
}

static PyObject *py_bispectrumdx(PyObject *self, PyObject *args) {
    return py_bispecd_generic(self, args, 1);
}

static PyObject *py_bispectrumi(PyObject *self, PyObject *args) {
    PyObject *yo;
    long nlag, nsamp_in, nfft_in;
    int overlap, wind_mode;
    const char *flag;
    PyObject *wind_obj;
    PyArrayObject *ya = NULL, *outa = NULL, *waxa = NULL;
    size_t nsamp, nfft, N;
    int biased;
    sp_status st;
    (void)self;

    if (!PyArg_ParseTuple(args, "OllislO", &yo, &nlag, &nsamp_in, &overlap,
                          &flag, &nfft_in, &wind_obj))
        return NULL;
    ya = py_to_c128_f(yo);
    if (!ya) return NULL;
    N = (size_t)PyArray_SIZE(ya);
    nsamp = (nsamp_in <= 0 || (size_t)nsamp_in > N) ? N : (size_t)nsamp_in;
    nfft = (nfft_in <= 0) ? 128 : (size_t)nfft_in;
    biased = (strcmp(flag, "biased") == 0);
    if (wind_obj == Py_None) {
        wind_mode = 0; /* Parzen */
    } else if (PyLong_Check(wind_obj)) {
        long v = PyLong_AsLong(wind_obj);
        wind_mode = (v == 0) ? 0 : ((v == -1) ? -1 : 1);
    } else {
        wind_mode = 1; /* hexagonal unity */
    }
    outa = py_new_c128(2, (npy_intp[]){(npy_intp)nfft, (npy_intp)nfft});
    waxa = py_new_f64(1, (npy_intp[]){(npy_intp)nfft});
    if (!outa || !waxa) {
        Py_XDECREF(outa);
        Py_XDECREF(waxa);
        Py_DECREF(ya);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    st = sp_bispeci((const sp_c128 *)PyArray_DATA(ya), N, nlag, nsamp, overlap,
                    biased, nfft, wind_mode, (sp_c128 *)PyArray_DATA(outa),
                    (double *)PyArray_DATA(waxa));
    Py_END_ALLOW_THREADS
    Py_DECREF(ya);
    if (st != SP_OK) {
        Py_DECREF(outa);
        Py_DECREF(waxa);
        return PyErr_Format(PyExc_ValueError, "bispectrumi kernel error %d",
                            (int)st);
    }
    return Py_BuildValue("NN", (PyObject *)outa, (PyObject *)waxa);
}

static PyObject *py_trispectrum(PyObject *self, PyObject *args) {
    PyObject *yo;
    long nfft_in, nsamp_in;
    int overlap, wtype;
    PyArrayObject *ya = NULL, *outa = NULL, *waxa = NULL;
    size_t nsamp, nfft, N, mfi;
    sp_status st;
    (void)self;

    if (!PyArg_ParseTuple(args, "Olili", &yo, &nfft_in, &wtype, &nsamp_in,
                          &overlap))
        return NULL;
    ya = py_to_c128_f(yo);
    if (!ya) return NULL;
    N = (size_t)PyArray_SIZE(ya);
    nsamp = (nsamp_in <= 0 || (size_t)nsamp_in > N) ? N : (size_t)nsamp_in;
    nfft = (size_t)(nfft_in > 0 ? nfft_in : 1);
    mfi = (nfft / 2 + 1) < (nfft / 8) ? (nfft / 2 + 1) : (nfft / 8);
    if (mfi == 0) {
        Py_DECREF(ya);
        PyErr_SetString(PyExc_ValueError, "nfft too small for trispectrum");
        return NULL;
    }
    outa = py_new_c128(3, (npy_intp[]){(npy_intp)mfi, (npy_intp)mfi, (npy_intp)mfi});
    waxa = py_new_f64(1, (npy_intp[]){(npy_intp)mfi});
    if (!outa || !waxa) {
        Py_XDECREF(outa);
        Py_XDECREF(waxa);
        Py_DECREF(ya);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    st = sp_trispectrum((const sp_c128 *)PyArray_DATA(ya), N, nsamp, nfft,
                        wtype, overlap, (sp_c128 *)PyArray_DATA(outa),
                        (double *)PyArray_DATA(waxa), &mfi);
    Py_END_ALLOW_THREADS
    Py_DECREF(ya);
    if (st != SP_OK) {
        Py_DECREF(outa);
        Py_DECREF(waxa);
        return PyErr_Format(PyExc_ValueError, "trispectrum kernel error %d",
                            (int)st);
    }
    return Py_BuildValue("NN", (PyObject *)outa, (PyObject *)waxa);
}

static PyObject *py_tricoherence(PyObject *self, PyObject *args) {
    PyObject *yo;
    long nfft_in, nsamp_in;
    int overlap, wtype;
    PyArrayObject *ya = NULL, *outa = NULL, *waxa = NULL;
    size_t nsamp, nfft, N, mfi;
    sp_status st;
    (void)self;

    if (!PyArg_ParseTuple(args, "Olili", &yo, &nfft_in, &wtype, &nsamp_in,
                          &overlap))
        return NULL;
    ya = py_to_c128_f(yo);
    if (!ya) return NULL;
    N = (size_t)PyArray_SIZE(ya);
    nsamp = (nsamp_in <= 0 || (size_t)nsamp_in > N) ? N : (size_t)nsamp_in;
    nfft = (size_t)(nfft_in > 0 ? nfft_in : 1);
    mfi = (nfft / 2 + 1) < (nfft / 8) ? (nfft / 2 + 1) : (nfft / 8);
    if (mfi == 0) {
        Py_DECREF(ya);
        PyErr_SetString(PyExc_ValueError, "nfft too small for tricoherence");
        return NULL;
    }
    outa = py_new_f64(3, (npy_intp[]){(npy_intp)mfi, (npy_intp)mfi, (npy_intp)mfi});
    waxa = py_new_f64(1, (npy_intp[]){(npy_intp)mfi});
    if (!outa || !waxa) {
        Py_XDECREF(outa);
        Py_XDECREF(waxa);
        Py_DECREF(ya);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    st = sp_tricoherence((const sp_c128 *)PyArray_DATA(ya), N, nsamp, nfft,
                          wtype, overlap, (double *)PyArray_DATA(outa),
                          (double *)PyArray_DATA(waxa), &mfi);
    Py_END_ALLOW_THREADS
    Py_DECREF(ya);
    if (st != SP_OK) {
        Py_DECREF(outa);
        Py_DECREF(waxa);
        return PyErr_Format(PyExc_ValueError, "tricoherence kernel error %d",
                            (int)st);
    }
    return Py_BuildValue("NN", (PyObject *)outa, (PyObject *)waxa);
}

static PyObject *py_cumtrue(PyObject *self, PyObject *args) {
    PyObject *mao, *aro;
    int norder, k;
    long nlags_in;
    PyArrayObject *maa = NULL, *ara = NULL, *outa = NULL;
    long q1, p1, nlags;
    sp_status st;
    (void)self;

    if (!PyArg_ParseTuple(args, "OOili", &mao, &aro, &norder, &nlags_in, &k))
        return NULL;
    if (norder < 2 || norder > 4) {
        PyErr_SetString(PyExc_ValueError, "norder should be 2, 3 or 4");
        return NULL;
    }
    maa = py_wind_to_f64(mao); /* reuse: contiguous float64 coercion */
    ara = py_wind_to_f64(aro);
    if (!maa || !ara) {
        Py_XDECREF(maa);
        Py_XDECREF(ara);
        return NULL;
    }
    q1 = (long)PyArray_SIZE(maa);
    p1 = (long)PyArray_SIZE(ara);
    nlags = (nlags_in < 0) ? (p1 - 1 + q1 - 1) : nlags_in;
    if (nlags < 0) nlags = 0;
    if (norder == 2) {
        outa = py_new_f64(2, (npy_intp[]){(npy_intp)(2 * nlags + 1), 1});
    } else {
        outa = py_new_f64(2, (npy_intp[]){(npy_intp)(2 * nlags + 1),
                                          (npy_intp)(2 * nlags + 1)});
    }
    if (!outa) {
        Py_DECREF(maa);
        Py_DECREF(ara);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    st = sp_cumtrue((const double *)PyArray_DATA(maa), q1,
                    (const double *)PyArray_DATA(ara), p1, norder, nlags, k,
                    (double *)PyArray_DATA(outa));
    Py_END_ALLOW_THREADS
    Py_DECREF(maa);
    Py_DECREF(ara);
    if (st != SP_OK) {
        Py_DECREF(outa);
        if (st == SP_ERR_ARG)
            PyErr_SetString(PyExc_ValueError, "unstable AR polynomial passed");
        else
            PyErr_Format(PyExc_RuntimeError, "cumtrue kernel error %d", (int)st);
        return NULL;
    }
    return (PyObject *)outa;
}

static PyObject *py_trispec(PyObject *self, PyObject *args) {
    PyObject *yo;
    long nfft_in, nsamp_in;
    int overlap;
    PyArrayObject *ya = NULL, *outa = NULL, *waxa = NULL;
    size_t nsamp, nfft, N, cols;
    sp_status st;
    (void)self;

    if (!PyArg_ParseTuple(args, "Olli", &yo, &nfft_in, &nsamp_in, &overlap))
        return NULL;
    ya = py_to_c128_f(yo);
    if (!ya) return NULL;
    N = (size_t)PyArray_SIZE(ya);
    cols = (PyArray_NDIM(ya) == 2) ? (size_t)PyArray_DIM(ya, 1) : 1;
    if (overlap < 0) overlap = 0;
    if (overlap > 99) overlap = 99;
    st = sp_bispec_meta(N / (cols ? cols : 1), cols, nsamp_in, overlap,
                        nfft_in, &nsamp, &nfft);
    if (st != SP_OK) {
        Py_DECREF(ya);
        return PyErr_Format(PyExc_ValueError, "invalid segmentation (status %d)", (int)st);
    }
    if (nfft > 512) {
        Py_DECREF(ya);
        PyErr_SetString(PyExc_ValueError,
                        "nfft too large for the full trispectrum cube (use spectrum.trispectrum)");
        return NULL;
    }
    outa = py_new_c128(3, (npy_intp[]){(npy_intp)nfft, (npy_intp)nfft, (npy_intp)nfft});
    waxa = py_new_f64(1, (npy_intp[]){(npy_intp)nfft});
    if (!outa || !waxa) {
        Py_XDECREF(outa);
        Py_XDECREF(waxa);
        Py_DECREF(ya);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    st = sp_trispec((const sp_c128 *)PyArray_DATA(ya), N, cols, nsamp, nfft,
                    overlap, (sp_c128 *)PyArray_DATA(outa),
                    (double *)PyArray_DATA(waxa));
    Py_END_ALLOW_THREADS
    Py_DECREF(ya);
    if (st != SP_OK) {
        Py_DECREF(outa);
        Py_DECREF(waxa);
        return PyErr_Format(PyExc_ValueError, "trispec kernel error %d", (int)st);
    }
    return Py_BuildValue("NN", (PyObject *)outa, (PyObject *)waxa);
}

static PyObject *py_wig_generic(PyObject *self, PyObject *args, int which) {
    /* which: 0 = wig2(real), 1 = wig2c, 2 = wig4(real), 3 = wig4c */
    PyObject *xo;
    long nfft_in, inc_in;
    PyArrayObject *xa = NULL, *outa = NULL;
    size_t N, n_time, nfft, inc;
    sp_status st;
    npy_intp dims[3];
    (void)self;

    if (!PyArg_ParseTuple(args, "Oll", &xo, &nfft_in, &inc_in)) return NULL;
    xa = py_to_c128_f(xo);
    if (!xa) return NULL;
    N = (size_t)PyArray_SIZE(xa);
    nfft = (size_t)(nfft_in > 0 ? nfft_in : 128);
    inc = (size_t)(inc_in > 0 ? inc_in : 1);
    n_time = (N + inc - 1) / inc;
    if (n_time == 0) n_time = 1;

    if (which <= 1) {
        outa = py_new_c128(2, (npy_intp[]){(npy_intp)n_time, (npy_intp)nfft});
    } else {
        dims[0] = (npy_intp)n_time;
        dims[1] = (npy_intp)nfft;
        dims[2] = (npy_intp)nfft;
        outa = py_new_c128(3, dims);
    }
    if (!outa) {
        Py_DECREF(xa);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    if (which <= 1)
        st = sp_wig2((const sp_c128 *)PyArray_DATA(xa), N, nfft, inc,
                     which == 1, (sp_c128 *)PyArray_DATA(outa));
    else
        st = sp_wig4((const sp_c128 *)PyArray_DATA(xa), N, nfft, inc,
                     which == 3, (sp_c128 *)PyArray_DATA(outa));
    Py_END_ALLOW_THREADS
    Py_DECREF(xa);
    if (st != SP_OK) {
        Py_DECREF(outa);
        return PyErr_Format(PyExc_ValueError, "wigner kernel error %d", (int)st);
    }
    /* the real-valued variants store real numbers in a complex container;
     * the python wrapper takes the real part */
    return (PyObject *)outa;
}

static PyObject *py_wig2(PyObject *self, PyObject *args) {
    return py_wig_generic(self, args, 0);
}
static PyObject *py_wig2c(PyObject *self, PyObject *args) {
    return py_wig_generic(self, args, 1);
}
static PyObject *py_wig4(PyObject *self, PyObject *args) {
    return py_wig_generic(self, args, 2);
}
static PyObject *py_wig4c(PyObject *self, PyObject *args) {
    return py_wig_generic(self, args, 3);
}
static PyObject *py_tfrwv(PyObject *self, PyObject *args) {
    PyObject *yo;
    long nfft_in, inc_in;
    PyArrayObject *ya = NULL, *outa = NULL, *ta = NULL;
    size_t N, n_time, nfft, inc;
    sp_status st;
    (void)self;

    if (!PyArg_ParseTuple(args, "Oll", &yo, &nfft_in, &inc_in)) return NULL;
    ya = py_to_c128_f(yo);
    if (!ya) return NULL;
    N = (size_t)PyArray_SIZE(ya);
    nfft = (size_t)(nfft_in > 0 ? nfft_in : 128);
    inc = (size_t)(inc_in > 0 ? inc_in : 1);
    n_time = (N + inc - 1) / inc;
    if (n_time == 0) n_time = 1;
    outa = py_new_c128(2, (npy_intp[]){(npy_intp)n_time, (npy_intp)nfft});
    ta = py_new_f64(1, (npy_intp[]){(npy_intp)n_time});
    if (!outa || !ta) {
        Py_XDECREF(outa);
        Py_XDECREF(ta);
        Py_DECREF(ya);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    st = sp_wig2((const sp_c128 *)PyArray_DATA(ya), N, nfft, inc, 0,
                 (sp_c128 *)PyArray_DATA(outa));
    Py_END_ALLOW_THREADS
    Py_DECREF(ya);
    if (st != SP_OK) {
        Py_DECREF(outa);
        Py_DECREF(ta);
        return PyErr_Format(PyExc_ValueError, "tfrwv kernel error %d", (int)st);
    }
    for (size_t i = 0; i < n_time; i++)
        ((double *)PyArray_DATA(ta))[i] = (double)(i * inc);
    /* materialize the real part as float64 */
    {
        npy_intp rdims[2] = {(npy_intp)n_time, (npy_intp)nfft};
        PyArrayObject *reala = py_new_f64(2, rdims);
        if (!reala) {
            Py_DECREF(outa);
            Py_DECREF(ta);
            return NULL;
        }
        const sp_c128 *src = (const sp_c128 *)PyArray_DATA(outa);
        double *dst = (double *)PyArray_DATA(reala);
        for (size_t i = 0; i < n_time * nfft; i++) dst[i] = creal(src[i]);
        Py_DECREF(outa);
        return Py_BuildValue("NN", (PyObject *)reala, (PyObject *)ta);
    }
}

static PyObject *py_doa_cum(PyObject *self, PyObject *args) {
    PyObject *xo;
    PyArrayObject *xa = NULL, *outa = NULL;
    size_t N, M;
    sp_status st;
    (void)self;

    if (!PyArg_ParseTuple(args, "O", &xo)) return NULL;
    xa = (PyArrayObject *)PyArray_FROM_OTF(xo, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!xa) return NULL;
    if (PyArray_NDIM(xa) != 2) {
        Py_DECREF(xa);
        PyErr_SetString(PyExc_ValueError, "y must be 2-D with shape (snapshots, sensors)");
        return NULL;
    }
    N = (size_t)PyArray_DIM(xa, 0);
    M = (size_t)PyArray_DIM(xa, 1);
    outa = py_new_c128(2, (npy_intp[]){(npy_intp)M, (npy_intp)M});
    if (!outa) {
        Py_DECREF(xa);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    st = sp_doa_cum((const sp_c128 *)PyArray_DATA(xa), N, M,
                    (sp_c128 *)PyArray_DATA(outa));
    Py_END_ALLOW_THREADS
    Py_DECREF(xa);
    if (st != SP_OK) {
        Py_DECREF(outa);
        return PyErr_Format(PyExc_ValueError, "doa kernel error %d", (int)st);
    }
    return (PyObject *)outa;
}

/* ---- selftest ------------------------------------------------------ */

static PyObject *py_selftest(PyObject *self, PyObject *args) {    double complex buf[16];
    (void)self;
    (void)args;
    for (int i = 0; i < 16; i++) buf[i] = (i == 1) ? 1.0 : 0.0;
    if (sp_fft_forward(buf, 16) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "fft engine failure");
        return NULL;
    }
    if (cabs(buf[1] - (cos(-M_PI / 8.0) + I * sin(-M_PI / 8.0))) > 1e-10) {
        PyErr_SetString(PyExc_RuntimeError, "fft engine produced wrong values");
        return NULL;
    }
    return PyLong_FromLong(42);
}

static PyMethodDef module_methods[] = {
    {"_selftest", py_selftest, METH_NOARGS, "binding self-test"},
    {"cum2est", py_cum2est, METH_VARARGS, "2nd-order cumulant kernel"},
    {"cum2x", py_cum2x, METH_VARARGS, "2nd-order cross-cumulant kernel"},
    {"cum3est", py_cum3est, METH_VARARGS, "3rd-order cumulant kernel"},
    {"cum3x", py_cum3x, METH_VARARGS, "3rd-order cross-cumulant kernel"},
    {"cum4est", py_cum4est, METH_VARARGS, "4th-order cumulant kernel"},
    {"cum4x", py_cum4x, METH_VARARGS, "4th-order cross-cumulant kernel"},
    {"bicoherence", py_bicoherence, METH_VARARGS, "bicoherence (direct method)"},
    {"bicoherencex", py_bicoherencex, METH_VARARGS, "cross bicoherence"},
    {"bispectrumd", py_bispectrumd, METH_VARARGS, "bispectrum (direct method)"},
    {"bispectrumdx", py_bispectrumdx, METH_VARARGS, "cross bispectrum (direct)"},
    {"bispectrumi", py_bispectrumi, METH_VARARGS, "bispectrum (indirect)"},
    {"trispectrum", py_trispectrum, METH_VARARGS, "trispectrum (reduced grid)"},
    {"tricoherence", py_tricoherence, METH_VARARGS, "tricoherence"},
    {"cumtrue", py_cumtrue, METH_VARARGS, "theoretical ARMA cumulants"},
    {"trispec", py_trispec, METH_VARARGS, "HOSA-style full trispectrum"},
    {"wig2", py_wig2, METH_VARARGS, "Wigner spectrum (real)"},
    {"wig2c", py_wig2c, METH_VARARGS, "Wigner spectrum (complex)"},
    {"wig4", py_wig4, METH_VARARGS, "Wigner bispectrum (real)"},
    {"wig4c", py_wig4c, METH_VARARGS, "Wigner bispectrum (complex)"},
    {"tfrwv", py_tfrwv, METH_VARARGS, "pseudo Wigner-Ville time-freq map"},
    {"doa_cum", py_doa_cum, METH_VARARGS, "doa 4th-order cumulant matrix"},
    {NULL, NULL, 0, NULL} /* sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_spectrum_core",
    "Native C core for the spectrum package (HOSA estimators).",
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit__spectrum_core(void) {
    import_array();
    PyObject *m = PyModule_Create(&moduledef);
    return m;
}
