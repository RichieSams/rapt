/* rapt - RichieSam's Adventures in Path Tracing
 *
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#pragma once

#include "common/typedefs.h"

#include "engine/camera.h"

#include <cuda_runtime.h>


namespace Scene {
struct Sphere;
}


__global__ void PathTraceKernel(unsigned char *textureData, uint width, uint height, size_t pitch, DeviceCamera camera, Scene::Sphere *spheres, uint numSpheres, uint hashedFrameNumber);
