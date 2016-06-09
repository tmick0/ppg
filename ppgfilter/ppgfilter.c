#include "filters.h"


static PyObject *ppgfilter_SubImageFilter            (PyObject *self, PyObject *args);
static PyObject *ppgfilter_UpImageFilter             (PyObject *self, PyObject *args);
static PyObject *ppgfilter_PaethImageFilter          (PyObject *self, PyObject *args);
static PyObject *ppgfilter_AverageImageFilter        (PyObject *self, PyObject *args);
static PyObject *ppgfilter_RandomLineImageFilter     (PyObject *self, PyObject *args);
static PyObject *ppgfilter_UniformGlitchImageFilter  (PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"SubImageFilter",             ppgfilter_SubImageFilter,             METH_VARARGS, NULL},
    {"UpImageFilter",              ppgfilter_UpImageFilter,              METH_VARARGS, NULL},
    {"PaethImageFilter",           ppgfilter_PaethImageFilter,           METH_VARARGS, NULL},
    {"AverageImageFilter",         ppgfilter_AverageImageFilter,         METH_VARARGS, NULL},
    {"RandomLineImageFilter",      ppgfilter_RandomLineImageFilter,      METH_VARARGS, NULL},
    {"UniformGlitchImageFilter",   ppgfilter_UniformGlitchImageFilter,   METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initppgfilter(void){
    if(Py_InitModule("ppgfilter", module_methods) == NULL){
        return;
    }
    import_array();
}

static PyObject *ppgfilter_SubImageFilter(PyObject *self, PyObject *args){
    PyObject *image_obj;
    uint8_t decode;
    
    if(!PyArg_ParseTuple(args, "O!b", &PyArray_Type, &image_obj, &decode)){
        return NULL;
    }
    
    PyArrayObject *image_arr = (PyArrayObject *) PyArray_FROM_OTF(image_obj, NPY_UBYTE, NPY_ARRAY_INOUT_ARRAY);
    
    if(image_arr == NULL){
        Py_XDECREF(image_obj);
        PyArray_XDECREF_ERR(image_arr);
        return NULL;
    }
    
    SubImageFilter(image_arr, decode);
    
    Py_DECREF(image_arr);
    
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *ppgfilter_UpImageFilter(PyObject *self, PyObject *args){
    PyObject *image_obj;
    uint8_t decode;
    
    if(!PyArg_ParseTuple(args, "O!b", &PyArray_Type, &image_obj, &decode)){
        return NULL;
    }
    
    PyArrayObject *image_arr = (PyArrayObject *) PyArray_FROM_OTF(image_obj, NPY_UBYTE, NPY_ARRAY_INOUT_ARRAY);
    
    if(image_arr == NULL){
        Py_XDECREF(image_obj);
        PyArray_XDECREF_ERR(image_arr);
        return NULL;
    }
    
    UpImageFilter(image_arr, decode);
    
    Py_DECREF(image_arr);
    
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *ppgfilter_PaethImageFilter(PyObject *self, PyObject *args){
    PyObject *image_obj;
    uint8_t decode;
    
    if(!PyArg_ParseTuple(args, "O!b", &PyArray_Type, &image_obj, &decode)){
        return NULL;
    }
    
    PyArrayObject *image_arr = (PyArrayObject *) PyArray_FROM_OTF(image_obj, NPY_UBYTE, NPY_ARRAY_INOUT_ARRAY);
    
    if(image_arr == NULL){
        Py_XDECREF(image_obj);
        PyArray_XDECREF_ERR(image_arr);
        return NULL;
    }
    
    PaethImageFilter(image_arr, decode);
    
    Py_DECREF(image_arr);
    
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *ppgfilter_AverageImageFilter(PyObject *self, PyObject *args){
    PyObject *image_obj;
    uint8_t decode;
    
    if(!PyArg_ParseTuple(args, "O!b", &PyArray_Type, &image_obj, &decode)){
        return NULL;
    }
    
    PyArrayObject *image_arr = (PyArrayObject *) PyArray_FROM_OTF(image_obj, NPY_UBYTE, NPY_ARRAY_INOUT_ARRAY);
    
    if(image_arr == NULL){
        Py_XDECREF(image_obj);
        PyArray_XDECREF_ERR(image_arr);
        return NULL;
    }
    
    AverageImageFilter(image_arr, decode);
    
    Py_DECREF(image_arr);
    
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *ppgfilter_RandomLineImageFilter(PyObject *self, PyObject *args){
    PyObject *image_obj;
    uint8_t decode;
    uint32_t seed = 0;
    
    if(!PyArg_ParseTuple(args, "O!Ib", &PyArray_Type, &image_obj, &seed, &decode)){
        return NULL;
    }
    
    PyArrayObject *image_arr = (PyArrayObject *) PyArray_FROM_OTF(image_obj, NPY_UBYTE, NPY_ARRAY_INOUT_ARRAY);

    if(image_arr == NULL){
        Py_XDECREF(image_obj);
        PyArray_XDECREF_ERR(image_arr);
        return NULL;
    }
    
    RandomLineImageFilter(image_arr, seed, decode);
    
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
