/* raic - RichieSam's Adventures in Cuda
 *
 * raic is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#pragma once

#include "engine/raic_engine.h"


namespace Graphics {
class D3DTexture2D;
class CudaTexture2D;
}

namespace DirectXInterop {

class DirectXInterop : public Engine::RAICEngine {
public:
	DirectXInterop(HINSTANCE hinstance);

private:
	ID3D11RenderTargetView *m_backbufferRTV;
	D3D11_VIEWPORT m_screenViewport;

	Graphics::D3DTexture2D *m_hdrTextureD3D;
	Graphics::CudaTexture2D *m_hdrTextureCuda;

	ID3D11VertexShader *m_fullscreenTriangleVS;
	ID3D11PixelShader *m_copyCudaOutputToBackbufferPS;

public:
	bool Initialize(LPCTSTR mainWndCaption, uint32 screenWidth, uint32 screenHeight, bool fullscreen);
	void Shutdown();

private:
	void OnResize();
	void DrawFrame();
};

} // End of namespace DirectXInterop
