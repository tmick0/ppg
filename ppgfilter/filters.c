#include "filters.h"
#include "utils.h"

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

    for(int c = 0; c < channels && c < 3; c++){
        for(int i = 0; i < height; i++){
            filter(data, dest, i, c, decode);
        }
    }
    
    if(!decode){
        PyArray_CopyInto(data, dest);
        Py_DECREF(dest);
    }

}

void NullLineFilter(PyArrayObject *data, PyArrayObject *dest, int row, int chan, bool decode){
    // :)
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

void UpLineFilter(PyArrayObject *data, PyArrayObject *dest, int row, int chan, bool decode){

    int width    = (int) PyArray_DIM(data, 1);
 
    if(row > 0){
     
        for(int j = 1; j < width; j++){
        
            uint8_t A = *((uint8_t *) PyArray_GETPTR3(data, row - 1, j, chan));
            
            if(decode){
                *((uint8_t *) PyArray_GETPTR3(dest, row, j, chan)) += A;
            }
            else{
                *((uint8_t *) PyArray_GETPTR3(dest, row, j, chan)) -= A;
            }
            
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
            uint8_t p = A + B - C;
            
            uint8_t dA = A - p, dB = B - p, dC = C - p;
            uint8_t x;
            
            if(dA <= dB && dA <= dC){
                x = A;
            }
            else if(dB <= dC){
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

void AverageLineFilter(PyArrayObject *data, PyArrayObject *dest, int row, int chan, bool decode){

    int width    = (int) PyArray_DIM(data, 1);
    
    if(row > 0){
     
        for(int j = 1; j < width; j++){
        
            uint8_t A = *((uint8_t *) PyArray_GETPTR3(data, row,   j-1, chan));
            uint8_t B = *((uint8_t *) PyArray_GETPTR3(data, row-1, j,   chan));
            uint8_t x = (A + B) / 2;
            
            if(decode){
                *((uint8_t *) PyArray_GETPTR3(dest, row, j, chan)) += x;
            }
            else{
                *((uint8_t *) PyArray_GETPTR3(dest, row, j, chan)) -= x;
            }
            
        }
    
    }
    
}

void RandomLineImageFilter(PyArrayObject *data, uint32_t seed, bool decode){

    import_array();

    rng_t r;
    rng_set_state(&r, seed);
    
    int height   = (int) PyArray_DIM(data, 0);
    int channels = (int) PyArray_DIM(data, 2);
    
    PyArrayObject *dest;

    if(!decode){
        dest = (PyArrayObject *) PyArray_NewCopy(data, NPY_ANYORDER);
    }
    else{
        dest = data;
    }
    
    for(int c = 0; c < channels && c < 3; c++){
        for(int i = 0; i < height; i++){
            LineFilterType *filter;
            switch(rng_next(&r) % 5){
                case 0:
                    filter = NullLineFilter;
                    continue;
                case 1:
                    filter = SubLineFilter;
                    break;
                case 2:
                    filter = UpLineFilter;
                    break;
                case 3:
                    filter = PaethLineFilter;
                    break;
                case 4:
                    filter = AverageLineFilter;
                    break;
            }
            filter(data, dest, i, c, decode);
        }
    }
    
    if(!decode){
        PyArray_CopyInto(data, dest);
        Py_DECREF(dest);
    }
}

void SubImageFilter(PyArrayObject *data, bool decode){
    LineFilter(data, SubLineFilter, decode);
}

void UpImageFilter(PyArrayObject *data, bool decode){
    LineFilter(data, UpLineFilter, decode);
}

void PaethImageFilter(PyArrayObject *data, bool decode){
    LineFilter(data, PaethLineFilter, decode);
}

void AverageImageFilter(PyArrayObject *data, bool decode){
    LineFilter(data, AverageLineFilter, decode);
}

void UniformGlitchImageFilter(PyArrayObject *data, uint32_t seed, uint32_t thresh){

    import_array();

    int height   = (int) PyArray_DIM(data, 0);
    int width    = (int) PyArray_DIM(data, 1);
    int channels = (int) PyArray_DIM(data, 2);
    
    rng_t r;
    rng_set_state(&r, seed);
    
    for(int c = 0; c < channels && c < 3; c++){
        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++){
                if(rng_next(&r) < thresh){
                    *((uint8_t *) PyArray_GETPTR3(data, i, j, c)) = rng_next(&r);
                }
            }
        }
    }
    
}
