#ifndef filters_h
#define filters_h
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdbool.h>

typedef void ImageFilterType(PyArrayObject *data, bool decode);
typedef void LineFilterType(PyArrayObject *data, PyArrayObject *dest, int row, int chan, bool decode);

void LineFilter(PyArrayObject *data, LineFilterType* filter, bool decode);

void SubLineFilter(PyArrayObject *data, PyArrayObject *dest, int row, int chan, bool decode);
void SubImageFilter(PyArrayObject *data, bool decode);

void PaethLineFilter(PyArrayObject *data, PyArrayObject *dest, int row, int chan, bool decode);
void PaethImageFilter(PyArrayObject *data, bool decode);

#endif
