/* rapt - RichieSam's Adventures in Path Tracing
 *
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "basic_path_tracer/path_trace.cuh"


uint32 WangHash(uint32 a) {
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

/**
 * Any code that uses CUDA semantics has to be a .cu/.cuh file. Instead of forcing
 * the entire project to be .cu/.cuh files, we can just wrap each cuda kernel launch
 * in a function. We forward declare the function in other parts of the project and
 * implement the functions here, in this .cu file.
 */
cudaError PathTraceNextFrame(void *buffer, uint width, uint height, size_t pitch, DeviceCamera *camera, Scene::SceneObjects *sceneObjects,  uint frameNumber) {
	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	// The actual kernel launch call
	PathTraceKernel<<<Dg, Db>>>((unsigned char *)buffer, width, height, pitch, camera, sceneObjects, WangHash(frameNumber));

	return cudaGetLastError();
}
