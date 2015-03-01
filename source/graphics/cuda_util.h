/* raic - RichieSam's Adventures in Cuda
 *
 * raic is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#pragma once

#include "cuda_runtime.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <sstream>



void CudaErrorWindow(const WCHAR *fileName, DWORD lineNumber, cudaError_t err, const WCHAR *message) {
	std::wstringstream messageStream;
	messageStream << L"File: " << std::wstring(fileName) << L"\n"
	              << L"Line: " << lineNumber << L"\n"
	              << L"Error msg: " << cudaGetErrorString(err) << L"\n"
	              << L"Calling: " << std::wstring(message) << L"\n\n"
	              << L"Do you want to debug the application?";

	int nResult = MessageBoxW(GetForegroundWindow(), messageStream.str().c_str(), L"Unexpected error encountered", MB_YESNO | MB_ICONERROR);
	if (nResult == IDYES) {
		DebugBreak();
	}
}

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