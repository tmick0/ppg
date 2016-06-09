#include "filters.h"

int absint(int x){
    unsigned tmp = x >> (sizeof(int)*8 - 1);
    x ^= tmp;
    x += tmp & 1;
    return x;
}

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

void PaethLineFilter(PyArrayObject *data, PyArrayObject *dest, int row, int chan, bool decode){

    int width    = (int) PyArray_DIM(data, 1);
    
    if(row > 0){
     
        for(int j = 1; j < width; j++){
        
            uint8_t A = *((uint8_t *) PyArray_GETPTR3(data, row,   j-1, chan));
            uint8_t B = *((uint8_t *) PyArray_GETPTR3(data, row-1, j,   chan));
            uint8_t C = *((uint8_t *) PyArray_GETPTR3(data, row-1, j-1, chan));
            int p = A + B - C;
            
            unsigned dA = absint(p - A), dB = absint(p - B), dC = absint(p - C);
            uint8_t x;
            
            if(dA < dB && dA < dC){
                x = A;
            }
            else if(dB < dC){
                x = B;
            }
            else{
                x = C;
            }
            
            if(decode){
                *((uint8_t *) PyArray_GETPTR3(dest, row, j, chan)) += x;
            }
            else{
                *((uint8_t *) PyArray_GETPTR3(dest, row, j, chan)) -= x;
            }
            
        }
    
    }
    
}

void SubImageFilter(PyArrayObject *data, bool decode){

    LineFilter(data, SubLineFilter, decode);

}

void PaethImageFilter(PyArrayObject *data, bool decode){

    LineFilter(data, PaethLineFilter, decode);

}
