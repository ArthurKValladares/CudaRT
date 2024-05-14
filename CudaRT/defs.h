#pragma once

#include <iostream>
#include <float.h>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

#define PI 3.1415926535897932385
#define FLOAT_MAX FLT_MAX