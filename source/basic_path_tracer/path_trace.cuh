/* rapt - RichieSam's Adventures in Path Tracing
 *
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "common/typedefs.h"

#include "engine/camera.h"

#include <cuda_runtime.h>
#include <vector_types.h>


__global__ void PathTraceKernel(unsigned char *textureData, uint width, uint height, size_t pitch, DeviceCamera camera, uint hashedFrameNumber);
