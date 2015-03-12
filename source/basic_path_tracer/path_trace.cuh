/* rapt - RichieSam's Adventures in Path Tracing
 *
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#pragma once

#include "common/typedefs.h"

#include <cuda_runtime.h>


namespace Scene {
struct SceneObjects;
}

struct DeviceCamera;


__global__ void PathTraceKernel(unsigned char *textureData, uint width, uint height, size_t pitch, DeviceCamera *g_camera, Scene::SceneObjects *g_sceneObjects, uint hashedFrameNumber);
