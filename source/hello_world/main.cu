/* raic - RichieSam's Adventures in Cuda
 *
 * raic is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_util.h"

typedef unsigned char byte;
typedef unsigned char uint8;
typedef signed char int8;

typedef unsigned short uint16;
typedef signed short int16;

typedef unsigned int uint32;
typedef signed int int32;
typedef unsigned int uint;

typedef __int64             int64;
typedef unsigned __int64    uint64;

typedef wchar_t wchar;

enum class DisposeAfterUse {
	NO, 
	YES 
};


#include <iostream>


__global__ void helloWorldKernel(char *inputChars, uint *indices, char *output) {
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	output[index] = inputChars[indices[index]];
}


int main() {
	char h_inputChars[10] = "Helo Wrd!";
	uint h_indexes[13] = {0, 1, 2, 2, 3, 4, 5, 3, 6, 2, 7, 8, 9};

	char h_output[13];
	
	char *d_inputChars;
	uint *d_indexes;
	char *d_output;
	CE(cudaMalloc((void**)&d_inputChars, 10 * sizeof(char)));
	CE(cudaMalloc((void**)&d_indexes, 13 * sizeof(uint)));
	CE(cudaMalloc((void**)&d_output, 13 * sizeof(char)));

	CE(cudaMemcpy(d_inputChars, h_inputChars, 9 * sizeof(char), cudaMemcpyHostToDevice));
	CE(cudaMemcpy(d_indexes, h_indexes,  13 * sizeof(uint), cudaMemcpyHostToDevice));

	helloWorldKernel<<<1, 13>>>(d_inputChars, d_indexes, d_output);

	CE(cudaDeviceSynchronize());

	CE(cudaMemcpy(h_output, d_output, 13 * sizeof(char), cudaMemcpyDeviceToHost));

	std::cout << h_output << std::endl;

	return 0;
}