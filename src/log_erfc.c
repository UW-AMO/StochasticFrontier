#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

/*
 * log_erfc.c
 * This is the C code for creating your own
 * NumPy ufunc for a log_erfc function.
 *
 * Each function of the form type_log_erfc defines the
 * log_erfc function for a different numpy dtype. Each
 * of these functions must be modified when you
 * create your own ufunc. The computations that must
 * be replaced to create a ufunc for
 * a different function are marked with BEGIN
 * and END.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 *
 */


static PyMethodDef LogErfcMethods[] = {
        {NULL, NULL, 0, NULL}
};

/* The loop definitions must precede the PyMODINIT_FUNC. */

static void long_double_log_erfc(char **args, npy_intp *dimensions,
                              npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out=args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    long double tmp;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        tmp = *(long double *)in;
        if (tmp > 25.0l) {
            *((long double *)out) = -tmp*tmp + \
                logl((1.0l - 0.5l/(tmp*tmp))/(tmp*1.7724538509055159l));
        } else {
            *((long double *)out) = logl(erfcl(tmp));
        }
        /*END main ufunc computation*/

        in += in_step;
        out += out_step;
    }
}

static void double_log_erfc(char **args, npy_intp *dimensions,
                         npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    double tmp;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        tmp = *(double *)in;
        if (tmp > 25.0) {
            *((double *)out) = -tmp*tmp + \
                log((1.0 - 0.5/(tmp*tmp))/(tmp*1.7724538509055159));
        } else {
            *((double *)out) = log(erfc(tmp));
        }
        /*END main ufunc computation*/

        in += in_step;
        out += out_step;
    }
}

static void float_log_erfc(char **args, npy_intp *dimensions,
                        npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in=args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    float tmp;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        tmp = *(float *)in;
        if (tmp > 25.0f) {
            *((float *)out) = -tmp*tmp + \
                logf((1.0f - 0.5f/(tmp*tmp))/(tmp*1.7724538509055159f));
        } else {
            *((float *)out) = logf(erfcf(tmp));
        }
        /*END main ufunc computation*/

        in += in_step;
        out += out_step;
    }
}


/*This gives pointers to the above functions*/
PyUFuncGenericFunction funcs[3] = {&float_log_erfc,
                                   &double_log_erfc,
                                   &long_double_log_erfc};

static char types[6] = {NPY_FLOAT, NPY_FLOAT,
                NPY_DOUBLE,NPY_DOUBLE,
                NPY_LONGDOUBLE, NPY_LONGDOUBLE};
static void *data[3] = {NULL, NULL, NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "npufunc",
    NULL,
    -1,
    LogErfcMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_npufunc(void)
{
    PyObject *m, *log_erfc, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    log_erfc = PyUFunc_FromFuncAndData(funcs, data, types, 4, 1, 1,
                                    PyUFunc_None, "log_erfc",
                                    "log_erfc_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "log_erfc", log_erfc);
    Py_DECREF(log_erfc);

    return m;
}
#else
PyMODINIT_FUNC initnpufunc(void)
{
    PyObject *m, *log_erfc, *d;


    m = Py_InitModule("npufunc", LogErfcMethods);
    if (m == NULL) {
        return;
    }

    import_array();
    import_umath();

    log_erfc = PyUFunc_FromFuncAndData(funcs, data, types, 4, 1, 1,
                                    PyUFunc_None, "log_erfc",
                                    "log_erfc_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "log_erfc", log_erfc);
    Py_DECREF(log_erfc);
}
#endif