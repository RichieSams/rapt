/* raic - RichieSam's Adventures in Cuda
 *
 * raic is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "directx_interop/directx_interop.h"

#include "graphics/d3d_util.h"

#include <DirectXColors.h>


namespace DirectXInterop {

DirectXInterop::DirectXInterop(HINSTANCE hinstance) 
	: Engine::RAICEngine(hinstance),
	  m_backbufferRTV(nullptr) {
}

bool DirectXInterop::Initialize(LPCTSTR mainWndCaption, uint32 screenWidth, uint32 screenHeight, bool fullscreen) {
	// Initialize the Engine
	if (!Engine::RAICEngine::Initialize(mainWndCaption, screenWidth, screenHeight, fullscreen)) {
		return false;
	}
}

void DirectXInterop::Shutdown() {
	// Release in the opposite order we initialized in

	Engine::RAICEngine::Shutdown();
}

void DirectXInterop::OnResize() {
	if (!m_d3dInitialized) {
		return;
	}

	// Release the old views and the old depth/stencil buffer.
	ReleaseCOM(m_backbufferRTV);

	// Resize the swap chain
	HR(m_swapChain->ResizeBuffers(2, m_clientWidth, m_clientHeight, DXGI_FORMAT_R8G8B8A8_UNORM, 0));

	// Recreate the render target view.
	ID3D11Texture2D *backBuffer;
	HR(m_swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&backBuffer)));
	HR(m_device->CreateRenderTargetView(backBuffer, 0, &m_backbufferRTV));
	ReleaseCOM(backBuffer);

	// Set the viewport transform.
	m_screenViewport.TopLeftX = 0;
	m_screenViewport.TopLeftY = 0;
	m_screenViewport.Width = static_cast<float>(m_clientWidth);
	m_screenViewport.Height = static_cast<float>(m_clientHeight);
	m_screenViewport.MinDepth = 0.0f;
	m_screenViewport.MaxDepth = 1.0f;

	m_immediateContext->RSSetViewports(1, &m_screenViewport);
}

void DirectXInterop::DrawFrame() {
	m_immediateContext->ClearRenderTargetView(m_backbufferRTV, DirectX::Colors::LightBlue);

	m_swapChain->Present(1u, 0u);
}

} // End of namespace DirectXInterop
