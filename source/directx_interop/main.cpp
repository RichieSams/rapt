/* raic - RichieSam's Adventures in Cuda
 *
 * raic is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "directx_interop/directx_interop.h"

#include "graphics/d3d_util.h"


int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nCmdShow) {
	DBG_UNREFERENCED_PARAMETER(hPrevInstance);
	DBG_UNREFERENCED_PARAMETER(lpCmdLine);
	DBG_UNREFERENCED_PARAMETER(nCmdShow);

	#ifdef _DEBUG
	ID3D11Debug *debugInterface = nullptr;
	#endif

	// Force scope so we can guarantee the destructor is called before we try to use ReportLiveDeviceObjects
	{
		DirectXInterop::DirectXInterop app(hInstance);

		app.Initialize(L"DirextX Interop", 1280, 720, false);

		#ifdef _DEBUG
		app.CreateDebugInterface(&debugInterface);
		#endif

		app.Run();

		app.Shutdown();
	}

	#ifdef _DEBUG
	debugInterface->ReportLiveDeviceObjects(D3D11_RLDO_DETAIL);

	ReleaseCOM(debugInterface);
	#endif

	return 0;
}