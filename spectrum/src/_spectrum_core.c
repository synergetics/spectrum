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
