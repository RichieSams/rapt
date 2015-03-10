/* rapt - RichieSam's Adventures in Path Tracing
 *
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "basic_path_tracer/path_trace.cuh"

#include <device_launch_parameters.h>
#include <graphics/helper_math.h>
#include <curand_kernel.h>
#include <vector_types.h>


__device__ float3 CalculateRayDirectionFromPixel(uint x, uint y, uint width, uint height, DeviceCamera &camera) {
	float3 viewVector = make_float3((((x + 0.5f /*TODO: Add jitter */) / width) * 2.0f - 1.0f) * camera.TanFovDiv2_X,
	                                -(((y + 0.5f /*TODO: Add jitter */) / height) * 2.0f - 1.0f) * camera.TanFovDiv2_Y,
	                                1.0f);

	// Matrix multiply
	return normalize(make_float3(dot(viewVector, camera.X),
	                             dot(viewVector, camera.Y),
	                             dot(viewVector, camera.Z)));
}

__global__ void PathTraceKernel(unsigned char *textureData, uint width, uint height, size_t pitch, DeviceCamera camera, uint hashedFrameNumber) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	// Global threadId
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	// Create random number generator
	curandState randState;
	curand_init(hashedFrameNumber + threadId, 0, 0, &randState);

	// Calculate the first ray for this pixel
	//float3 ray = CalculateRayDirectionFromPixel(x, y, width, height, camera);

	// Generate a uniform random number
	float randNum = curand_uniform(&randState);

	// Get a pointer to the pixel at (x,y)
	float *pixel = (float *)(textureData + y * pitch) + 4 /*RGBA*/ * x;

	if (x < width && y < height) {
		// Write out pixel data
		if (randNum < 0.33f) {
			pixel[0] += 1.0f;
			pixel[1] += 0.0f;
			pixel[2] += 0.0f;
			pixel[3] = 1.0f;
		} else if (randNum < 0.66f) {
			pixel[0] += 0.0f;
			pixel[1] += 1.0f;
			pixel[2] += 0.0f;
			pixel[3] = 1.0f;
		} else {
			pixel[0] += 0.0f;
			pixel[1] += 0.0f;
			pixel[2] += 1.0f;
			pixel[3] = 1.0f;
		}
	}
}
