#include "filters.h"

#define STANDARD_DECL( WRAPPER)                                                \
    static PyObject * WRAPPER (PyObject *self, PyObject *args)

#define STANDARD_MAPPING(FUNC, WRAPPER)                                        \
    {FUNC, WRAPPER, METH_VARARGS, NULL}

#define STANDARD_IMAGEFILTER_WRAPPER(FUNC, WRAPPER)                            \
    static PyObject * WRAPPER (PyObject *self, PyObject *args){                \
        PyObject *obj;                                                         \
        uint8_t decode;                                                        \
                                                                               \
        if(!PyArg_ParseTuple(args, "O!b", &PyArray_Type, &obj, &decode)){      \
            return NULL;                                                       \
        }                                                                      \
                                                                               \
        PyArrayObject *image_arr = (PyArrayObject *)                           \
            PyArray_FROM_OTF(obj, NPY_UBYTE, NPY_ARRAY_INOUT_ARRAY);           \
                                                                               \
        if(image_arr == NULL){                                                 \
            Py_XDECREF(obj);                                                   \
            PyArray_XDECREF_ERR(image_arr);                                    \
            return NULL;                                                       \
        }                                                                      \
                                                                               \
        FUNC(image_arr, decode);                                               \
                                                                               \
        Py_DECREF(image_arr);                                                  \
                                                                               \
        Py_INCREF(Py_None);                                                    \
        return Py_None;                                                        \
    }

// PNG-style image filter decls
STANDARD_DECL(ppgfilter_SubImageFilter);
STANDARD_DECL(ppgfilter_UpImageFilter);
STANDARD_DECL(ppgfilter_AverageImageFilter);
STANDARD_DECL(ppgfilter_PaethImageFilter);
STANDARD_DECL(ppgfilter_BrokenPaethImageFilter);
STANDARD_DECL(ppgfilter_BrokenAverageImageFilter);

// Other image filter decls
STANDARD_DECL(ppgfilter_RandomLineImageFilter);
STANDARD_DECL(ppgfilter_UniformGlitchImageFilter);

static PyMethodDef module_methods[] = {
    
    // PNG-style image filter mappings
    STANDARD_MAPPING("SubImageFilter",             ppgfilter_SubImageFilter),
    STANDARD_MAPPING("UpImageFilter",              ppgfilter_UpImageFilter),
    STANDARD_MAPPING("PaethImageFilter",           ppgfilter_PaethImageFilter),
    STANDARD_MAPPING("AverageImageFilter",         ppgfilter_AverageImageFilter),
    STANDARD_MAPPING("BrokenPaethImageFilter",     ppgfilter_BrokenPaethImageFilter),
    STANDARD_MAPPING("BrokenAverageImageFilter",   ppgfilter_BrokenAverageImageFilter),

    // Other image filter mappings
    STANDARD_MAPPING("RandomLineImageFilter",      ppgfilter_RandomLineImageFilter),
    STANDARD_MAPPING("UniformGlitchImageFilter",   ppgfilter_UniformGlitchImageFilter),
    
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initppgfilter(void){
    if(Py_InitModule("ppgfilter", module_methods) == NULL){
        return;
    }
    import_array();
}

STANDARD_IMAGEFILTER_WRAPPER(SubImageFilter, ppgfilter_SubImageFilter)
STANDARD_IMAGEFILTER_WRAPPER(UpImageFilter, ppgfilter_UpImageFilter)
STANDARD_IMAGEFILTER_WRAPPER(PaethImageFilter, ppgfilter_PaethImageFilter)
STANDARD_IMAGEFILTER_WRAPPER(AverageImageFilter, ppgfilter_AverageImageFilter)
STANDARD_IMAGEFILTER_WRAPPER(BrokenPaethImageFilter, ppgfilter_BrokenPaethImageFilter)
STANDARD_IMAGEFILTER_WRAPPER(BrokenAverageImageFilter, ppgfilter_BrokenAverageImageFilter)

static PyObject *ppgfilter_RandomLineImageFilter(PyObject *self, PyObject *args){
    PyObject *image_obj, *candidates_obj;
    uint8_t decode;
    uint32_t seed, corr;
    
    if(!PyArg_ParseTuple(args, "O!O!IIb", &PyArray_Type, &image_obj, &PyList_Type, &candidates_obj, &seed, &corr, &decode)){
        return NULL;
    }
    
    PyArrayObject *image_arr = (PyArrayObject *) PyArray_FROM_OTF(image_obj, NPY_UBYTE, NPY_ARRAY_INOUT_ARRAY);

    if(image_arr == NULL){
        Py_XDECREF(image_obj);
        PyArray_XDECREF_ERR(image_arr);
        return NULL;
    }
    
    RandomLineImageFilter(image_arr, (PyListObject *) candidates_obj, seed, corr, decode);
    
    Py_DECREF(image_arr);
    
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *ppgfilter_UniformGlitchImageFilter(PyObject *self, PyObject *args){
    PyObject *image_obj;
    uint32_t seed, prob;
    
    if(!PyArg_ParseTuple(args, "O!II", &PyArray_Type, &image_obj, &seed, &prob)){
        return NULL;
    }
    
    PyArrayObject *image_arr = (PyArrayObject *) PyArray_FROM_OTF(image_obj, NPY_UBYTE, NPY_ARRAY_INOUT_ARRAY);

    if(image_arr == NULL){
        Py_XDECREF(image_obj);
        PyArray_XDECREF_ERR(image_arr);
        return NULL;
    }
    
    UniformGlitchImageFilter(image_arr, seed, prob);
    
    Py_DECREF(image_arr);
    
    Py_INCREF(Py_None);
    return Py_None;
}
