/* raic - RichieSam's Adventures in Cuda
 *
 * raic is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#pragma once

#include "engine/raic_engine.h"


namespace DirectXInterop {

class DirectXInterop : public Engine::RAICEngine {
public:
	DirectXInterop(HINSTANCE hinstance);

private:
	ID3D11RenderTargetView *m_backbufferRTV;
	D3D11_VIEWPORT m_screenViewport;

public:
	bool Initialize(LPCTSTR mainWndCaption, uint32 screenWidth, uint32 screenHeight, bool fullscreen);
	void Shutdown();

private:
	void OnResize();
	void DrawFrame();
};

} // End of namespace DirectXInterop
