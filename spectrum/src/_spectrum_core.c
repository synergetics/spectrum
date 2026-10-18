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

/* ---- selftest ------------------------------------------------------ */

static PyObject *py_selftest(PyObject *self, PyObject *args) {
    double complex buf[16];
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
