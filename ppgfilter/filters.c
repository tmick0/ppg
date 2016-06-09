#include "filters.h"

void LineFilter(PyArrayObject *data, LineFilterType* filter, bool decode){

    import_array();

    int height   = (int) PyArray_DIM(data, 0);
    int channels = (int) PyArray_DIM(data, 2);
    
    PyArrayObject *dest;

    if(!decode){
        dest = (PyArrayObject *) PyArray_NewCopy(data, NPY_ANYORDER);
    }
    else{
        dest = data;
    }

    for(int c = 0; c < channels; c++){
        for(int i = 0; i < height; i++){
            filter(data, dest, i, c, decode);
        }
    }
    
    if(!decode){
        PyArray_CopyInto(data, dest);
        Py_DECREF(dest);
    }

}

void SubLineFilter(PyArrayObject *data, PyArrayObject *dest, int row, int chan, bool decode){

    int width    = (int) PyArray_DIM(data, 1);
 
    for(int j = 1; j < width; j++){
    
        uint8_t A = *((uint8_t *) PyArray_GETPTR3(data, row, j-1, chan));
        
        if(decode){
            *((uint8_t *) PyArray_GETPTR3(dest, row, j, chan)) += A;
        }
        else{
            *((uint8_t *) PyArray_GETPTR3(dest, row, j, chan)) -= A;
        }
        
    }
    
}

void SubImageFilter(PyArrayObject *data, bool decode){

    LineFilter(data, SubLineFilter, decode);

}
