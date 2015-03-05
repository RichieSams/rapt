/* raic - RichieSam's Adventures in Cuda
 *
 * raic is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "basic_path_tracer/path_trace.cuh"

#include <device_launch_parameters.h>
#include <graphics/helper_math.h>
#include <curand_kernel.h>


__device__ float3 CalculateRayDirectionFromPixel(uint x, uint y, uint width, uint height, DeviceCamera &camera) {
	float3 viewVector = make_float3((((x + 0.5f /*TODO: Add jitter */) / width) * 2.0f - 1.0f) * camera.tanFovDiv2_X,
	                                (((y + 0.5f /*TODO: Add jitter */) / height) * 2.0f - 1.0f) * camera.tanFovDiv2_Y,
	                                -1.0f);

	// Matrix multiply
	return normalize(make_float3(dot(viewVector, camera.x),
	                             dot(viewVector, camera.y),
	                             dot(viewVector, camera.z)));
}

__global__ void RayTrace(unsigned char *textureData, uint width, uint height, size_t pitch, DeviceCamera camera, uint hashedFrameNumber) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	// Calculate the first ray for this pixel
	float3 ray = CalculateRayDirectionFromPixel(x, y, width, height, camera);

	// Get a pointer to the pixel at (x,y)
	float *pixel = (float *)(textureData + y * pitch) + 4 /*RGBA*/ * x;

	if (x < width && y < height) {
		// Write out pixel data
		pixel[0] = (ray.x + 1.0f) * 0.5f;
		pixel[1] = (ray.y + 1.0f) * 0.5f;
		pixel[2] = (ray.z + 1.0f) * 0.5f;
		pixel[3] = 1.0f;
	}
}