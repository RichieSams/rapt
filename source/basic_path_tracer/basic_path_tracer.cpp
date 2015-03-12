/* rapt - RichieSam's Adventures in Path Tracing
 *
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "basic_path_tracer/basic_path_tracer.h"

#include "graphics/d3d_util.h"
#include "graphics/cuda_texture2d.h"


namespace BasicPathTracer {

BasicPathTracer::BasicPathTracer(HINSTANCE hinstance) 
	: Engine::RAPTEngine(hinstance),
	  m_backbufferRTV(nullptr),
	  m_hostCamera(-DirectX::XM_PI, DirectX::XM_PIDIV2, 30.0f),
	  m_mouseLastPos_X(0),
	  m_mouseLastPos_Y(0),
	  m_cameraMoved(true),
	  m_frameNumber(0u) {
}

void BasicPathTracer::OnResize() {
	if (!m_d3dInitialized) {
		return;
	}

	// Update the camera projection
	m_hostCamera.SetProjection(1.04719755f, (float)m_clientWidth / m_clientHeight);

	// Release the old views and the old depth/stencil buffer.
	ReleaseCOM(m_backbufferRTV);

	// Resize the swap chain
	HR(m_swapChain->ResizeBuffers(2, m_clientWidth, m_clientHeight, DXGI_FORMAT_R8G8B8A8_UNORM, 0));

	// Recreate the render target view.
	ID3D11Texture2D *backBuffer;
	HR(m_swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&backBuffer)));
	HR(m_device->CreateRenderTargetView(backBuffer, 0, &m_backbufferRTV));
	ReleaseCOM(backBuffer);

	// Set the backbuffer as the rendertarget
	m_immediateContext->OMSetRenderTargets(1u, &m_backbufferRTV, nullptr);

	// Set the viewport transform.
	m_screenViewport.TopLeftX = 0;
	m_screenViewport.TopLeftY = 0;
	m_screenViewport.Width = static_cast<float>(m_clientWidth);
	m_screenViewport.Height = static_cast<float>(m_clientHeight);
	m_screenViewport.MinDepth = 0.0f;
	m_screenViewport.MaxDepth = 1.0f;

	m_immediateContext->RSSetViewports(1, &m_screenViewport);
}

void BasicPathTracer::MouseDown(WPARAM buttonState, int x, int y) {
	m_mouseLastPos_X = x;
	m_mouseLastPos_Y = y;

	SetCapture(m_hwnd);
}

void BasicPathTracer::MouseUp(WPARAM buttonState, int x, int y) {
	ReleaseCapture();
}

void BasicPathTracer::MouseMove(WPARAM buttonState, int x, int y) {
	if ((buttonState & MK_LBUTTON) != 0) {
		if (GetKeyState(VK_MENU) & 0x8000) {
			// Calculate the new phi and theta based on mouse position relative to where the user clicked
			float dPhi = ((float)(m_mouseLastPos_Y - y) / 300);
			float dTheta = ((float)(m_mouseLastPos_X - x) / 300);

			m_hostCamera.Rotate(-dTheta, dPhi);

			ResetAccumulationBuffer();
			m_cameraMoved = true;
		}
	} else if ((buttonState & MK_MBUTTON) != 0) {
		if (GetKeyState(VK_MENU) & 0x8000) {
			float dx = ((float)(m_mouseLastPos_X - x));
			float dy = ((float)(m_mouseLastPos_Y - y));

			m_hostCamera.Pan(dx * 0.1f, -dy * 0.1f);

			ResetAccumulationBuffer();
			m_cameraMoved = true;
		}
	}

	m_mouseLastPos_X = x;
	m_mouseLastPos_Y = y;
}

void BasicPathTracer::MouseWheel(int zDelta) {
	m_hostCamera.Zoom((float)zDelta * 0.01f);

	ResetAccumulationBuffer();
	m_cameraMoved = true;
}

void BasicPathTracer::ResetAccumulationBuffer() {
	m_hdrTextureCuda->MemSet(0);
	m_frameNumber = 0u;
}

} // End of namespace BasicPathTracer
