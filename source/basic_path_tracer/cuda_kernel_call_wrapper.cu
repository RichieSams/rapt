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

void PathTraceNextFrame(void *buffer, uint width, uint height, size_t pitch, DeviceCamera *camera, Scene::Sphere *spheres, uint numSpheres, Scene::LambertMaterial *g_materials, uint numMaterials, uint frameNumber) {
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	PathTraceKernel<<<Dg, Db>>>((unsigned char *)buffer, width, height, pitch, camera, spheres, numSpheres, g_materials, numMaterials, WangHash(frameNumber));

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		//exit(error);
	}
}
