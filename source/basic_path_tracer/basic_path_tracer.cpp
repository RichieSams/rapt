/* raic - RichieSam's Adventures in Cuda
 *
 * raic is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "basic_path_tracer/basic_path_tracer.h"

#include "graphics/d3d_util.h"
#include "graphics/d3d_texture2d.h"
#include "graphics/cuda_texture2d.h"

#include <cuda_runtime.h>


namespace BasicPathTracer {

BasicPathTracer::BasicPathTracer(HINSTANCE hinstance) 
	: Engine::RAICEngine(hinstance),
	  m_backbufferRTV(nullptr),
	  m_hostCamera(0.0f, -DirectX::XM_PIDIV2, 10.0f),
	  m_frameNumber(0u) {
}

bool BasicPathTracer::Initialize(LPCTSTR mainWndCaption, uint32 screenWidth, uint32 screenHeight, bool fullscreen) {
	// Initialize the Engine
	if (!Engine::RAICEngine::Initialize(mainWndCaption, screenWidth, screenHeight, fullscreen)) {
		return false;
	}

	m_hdrTextureD3D = new Graphics::D3DTexture2D(m_device, screenWidth, screenHeight, DXGI_FORMAT_R32G32B32A32_FLOAT, D3D11_BIND_SHADER_RESOURCE, 1);
	m_hdrTextureCuda = new Graphics::CudaTexture2D(screenWidth, screenHeight, 16u);
	m_hdrTextureCuda->RegisterResource(m_hdrTextureD3D->GetTexture(), cudaGraphicsRegisterFlagsNone);
	m_hdrTextureCuda->MemSet(0);

	Graphics::LoadVertexShader(L"fullscreen_triangle_vs.cso", m_device, &m_fullscreenTriangleVS);
	Graphics::LoadPixelShader(L"copy_cuda_output_to_backbuffer_ps.cso", m_device, &m_copyCudaOutputToBackbufferPS);

	// Bind the shaders and SRV to the pipeline
	// We never bind anything else, so we can just do it once during initialization
	m_immediateContext->VSSetShader(m_fullscreenTriangleVS, nullptr, 0u);
	m_immediateContext->PSSetShader(m_copyCudaOutputToBackbufferPS, nullptr, 0u);

	ID3D11ShaderResourceView *hdrSRV = m_hdrTextureD3D->GetShaderResource();
	m_immediateContext->PSSetShaderResources(0, 1, &hdrSRV);

	return true;
}

void BasicPathTracer::Shutdown() {
	// Release in the opposite order we initialized in

	Engine::RAICEngine::Shutdown();
}

void BasicPathTracer::OnResize() {
	if (!m_d3dInitialized) {
		return;
	}

	// Update the camera projection
	m_hostCamera.SetProjection(DirectX::XM_PIDIV2, (float)m_clientWidth / m_clientHeight);

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



} // End of namespace DirectXInterop
