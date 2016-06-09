#ifndef filters_h
#define filters_h
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdbool.h>

typedef void ImageFilterType(PyArrayObject *data, bool decode);
typedef void LineFilterType(PyArrayObject *data, PyArrayObject *dest, int row, int chan, bool decode);

typedef void ImageSeededFilterType(PyArrayObject *data, uint32_t seed, bool decode);

void LineFilter(PyArrayObject *data, LineFilterType* filter, bool decode);

void SubLineFilter(PyArrayObject *data, PyArrayObject *dest, int row, int chan, bool decode);
void SubImageFilter(PyArrayObject *data, bool decode);

void UpLineFilter(PyArrayObject *data, PyArrayObject *dest, int row, int chan, bool decode);
void UpImageFilter(PyArrayObject *data, bool decode);

void PaethLineFilter(PyArrayObject *data, PyArrayObject *dest, int row, int chan, bool decode);
void PaethImageFilter(PyArrayObject *data, bool decode);

void AverageLineFilter(PyArrayObject *data, PyArrayObject *dest, int row, int chan, bool decode);
void AverageImageFilter(PyArrayObject *data, bool decode);

void NullLineFilter(PyArrayObject *data, PyArrayObject *dest, int row, int chan, bool decode);

void RandomLineImageFilter(PyArrayObject *data, uint32_t seed, bool decode);

void UniformGlitchImageFilter(PyArrayObject *data, uint32_t seed, uint32_t thresh);

#endif
