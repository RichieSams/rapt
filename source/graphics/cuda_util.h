/* rapt - RichieSam's Adventures in Path Tracing
 *
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#pragma once

#include "cuda_runtime.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>


void CudaErrorWindow(const WCHAR *fileName, DWORD lineNumber, cudaError_t err, const WCHAR *message);

#if defined(DEBUG) | defined(_DEBUG)
	#ifndef CE
		#define CE(x) {                                                     \
				cudaError_t err = (x);                                      \
				if (err != cudaSuccess) {                                   \
					CudaErrorWindow(__FILEW__, (DWORD)__LINE__, err, L#x);  \
				}                                                           \
			}
		#endif

	#else
		#ifndef CE
		#define CE(x) (x)
	#endif
#endif