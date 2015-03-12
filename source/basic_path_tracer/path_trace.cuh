/* rapt - RichieSam's Adventures in Path Tracing
 *
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#pragma once

#include "common/typedefs.h"

#include <cuda_runtime.h>


namespace Scene {
struct Sphere;
struct LambertMaterial;
}

struct DeviceCamera;


__global__ void PathTraceKernel(unsigned char *textureData, uint width, uint height, size_t pitch, DeviceCamera *g_camera, Scene::Sphere *g_spheres, uint numSpheres, Scene::LambertMaterial *g_materials, uint numMaterials, uint hashedFrameNumber);
