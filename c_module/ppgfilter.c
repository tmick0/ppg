#include "filters.h"

#define MODULE_DOCSTRING "Low-level implementations of the PPG image filters"

// Macro for defining the prototype of a Python function
#define STANDARD_DECL( WRAPPER)                                                \
    static PyObject * WRAPPER (PyObject *self, PyObject *args)

// Macro for defining the mapping of a function into the Python module
#define STANDARD_MAPPING(FUNC, WRAPPER, DOCSTRING)                             \
    {FUNC, WRAPPER, METH_VARARGS, DOCSTRING}

// Macro for defining the most basic PNG-style image filters
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

// Define mapping of functions
static PyMethodDef module_methods[] = {
    
    // PNG-style image filter mappings
    
    STANDARD_MAPPING(
        "SubImageFilter",
        ppgfilter_SubImageFilter,
        "Low-level implementation of the SubFilter. Modifies the input image in-place.\n"
        "\n"
        "Parameters:\n"
        "\n"
        "    - data: Numpy array representing the image\n"
        "    - decode: boolean -- true = decode, false = encode"
    ),
    
    STANDARD_MAPPING(
        "UpImageFilter",
        ppgfilter_UpImageFilter,
        "Low-level implementation of the UpFilter. Modifies the input image in-place.\n"
        "\n"
        "Parameters:\n"
        "\n"
        "    - data: Numpy array representing the image\n"
        "    - decode: boolean -- true = decode, false = encode"
    ),
    
    STANDARD_MAPPING(
        "PaethImageFilter",
        ppgfilter_PaethImageFilter,
        "Low-level implementation of the PaethFilter. Modifies the input image in-place.\n"
        "\n"
        "Parameters:\n"
        "\n"
        "    - data: Numpy array representing the image\n"
        "    - decode: boolean -- true = decode, false = encode"
    ),
    
    STANDARD_MAPPING(
        "AverageImageFilter",
        ppgfilter_AverageImageFilter,
        "Low-level implementation of the AverageFilter. Modifies the input image in-place.\n"
        "\n"
        "Parameters:\n"
        "\n"
        "    - data: Numpy array representing the image\n"
        "    - decode: boolean -- true = decode, false = encode"
    ),
    
    STANDARD_MAPPING(
        "BrokenPaethImageFilter",
        ppgfilter_BrokenPaethImageFilter,
        "Low-level implementation of the BrokenPaethFilter. Modifies the input image in-place.\n"
        "\n"
        "Parameters:\n"
        "\n"
        "    - data: Numpy array representing the image\n"
        "    - decode: boolean -- true = decode, false = encode"
    ),
    
    STANDARD_MAPPING(
        "BrokenAverageImageFilter",
        ppgfilter_BrokenAverageImageFilter,
        "Low-level implementation of the BrokenAverageFilter. Modifies the input image in-place.\n"
        "\n"
        "Parameters:\n"
        "\n"
        "    - data: Numpy array representing the image\n"
        "    - decode: boolean -- true = decode, false = encode"
    ),

    // Other image filter mappings
    
    STANDARD_MAPPING(
        "RandomLineImageFilter",
        ppgfilter_RandomLineImageFilter,
        "Low-level implementation of the RandomLineFilter. Modifies the input image in-place.\n"
        "\n"
        "Parameters:\n"
        "\n"
        "    - data: Numpy array representing the image\n"
        "    - candidates: list of strings of the names of filters to alternate between\n"
        "    - seed: 32-bit unsigned integer, to initialize the internal LCG\n"
        "    - corr: 32-bit unsigned integer, representing the correlation threshold -- \n"
        "      essentially the correlation probability multiplied by 2^32.\n"
        "    - decode: boolean -- true = decode, false = encode"
    ),
    
    STANDARD_MAPPING(
        "UniformGlitchImageFilter",
        ppgfilter_UniformGlitchImageFilter,
        "Low-level implementation of the UniformGlitchFilter. Modifies the input image in-place.\n"
        "\n"
        "Parameters:\n"
        "\n"
        "    - data: Numpy array representing the image\n"
        "    - seed: 32-bit unsigned integer, to initialize the internal LCG\n"
        "    - prob: 32-bit unsigned integer, representing the alteration threshold --\n"
        "      essentially the alteration rate multiplied by 2^32."
    ),
    
    {NULL, NULL, 0, NULL}
};

// Module initialization
PyMODINIT_FUNC initppgfilter(void){
    if(Py_InitModule3("ppgfilter", module_methods, MODULE_DOCSTRING) == NULL){
        return;
    }
    import_array();
}

// Generic wrappers for PNG-style filters
STANDARD_IMAGEFILTER_WRAPPER(SubImageFilter, ppgfilter_SubImageFilter)
STANDARD_IMAGEFILTER_WRAPPER(UpImageFilter, ppgfilter_UpImageFilter)
STANDARD_IMAGEFILTER_WRAPPER(PaethImageFilter, ppgfilter_PaethImageFilter)
STANDARD_IMAGEFILTER_WRAPPER(AverageImageFilter, ppgfilter_AverageImageFilter)
STANDARD_IMAGEFILTER_WRAPPER(BrokenPaethImageFilter, ppgfilter_BrokenPaethImageFilter)
STANDARD_IMAGEFILTER_WRAPPER(BrokenAverageImageFilter, ppgfilter_BrokenAverageImageFilter)

// RandomLineFilter wrapper
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

// UniformGlitchImageFilter wrapper
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
