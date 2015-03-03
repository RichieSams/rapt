/* raic - RichieSam's Adventures in Cuda
 *
 * raic is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "graphics/cuda_util.h"

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
