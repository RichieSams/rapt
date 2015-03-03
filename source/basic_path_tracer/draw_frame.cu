/* raic - RichieSam's Adventures in Cuda
 *
 * raic is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "common/typedefs.h"

#include <stdio.h>


__global__ void cuda_kernel_texture_2d(unsigned char *surface, int width, int height, size_t pitch, float t) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= width || y >= height) {
		return;
	}

	// get a pointer to the pixel at (x,y)
	float *pixel = (float *)(surface + y * pitch) + 4 * x;

	// populate it
	float value_x = 0.5f + 0.5f * cos(t + 10.0f * ((2.0f * x) / width  - 1.0f));
	float value_y = 0.5f + 0.5f * cos(t + 10.0f * ((2.0f * y) / height - 1.0f));

	pixel[0] = 0.5 * pixel[0] + 0.5 * pow(value_x, 3.0f); // red
	pixel[1] = 0.5 * pixel[1] + 0.5 * pow(value_y, 3.0f); // green
	pixel[2] = 0.5f + 0.5f * cos(t); // blue
	pixel[3] = 1.0f; // alpha
}


void RenderFrame(void *buffer, uint width, uint height, size_t pitch, float t) {
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	cuda_kernel_texture_2d<<<Dg, Db>>>((unsigned char *)buffer, width, height, pitch, t);

	error = cudaGetLastError();

	if (error != cudaSuccess) {
		printf("cuda_kernel_texture_2d() failed to launch error = %d\n", error);
	}
}