#include "filters.h"

//static char module_docstring[] = "";
static char SubImageFilter_docstring[] = "";

static PyObject *ppgfilter_SubImageFilter(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"SubImageFilter", ppgfilter_SubImageFilter, METH_VARARGS, SubImageFilter_docstring},
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
