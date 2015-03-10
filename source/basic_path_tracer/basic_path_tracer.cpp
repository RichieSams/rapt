/* rapt - RichieSam's Adventures in Path Tracing
 *
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "basic_path_tracer/basic_path_tracer.h"

#include "graphics/d3d_util.h"
#include "graphics/d3d_texture2d.h"
#include "graphics/cuda_texture2d.h"

#include "scene/scene_objects.h"

#include <cuda_runtime.h>


namespace BasicPathTracer {

BasicPathTracer::BasicPathTracer(HINSTANCE hinstance) 
	: Engine::RAPTEngine(hinstance),
	  m_backbufferRTV(nullptr),
	  m_hostCamera(0.0f, -DirectX::XM_PIDIV2, 10.0f),
	  m_frameNumber(0u) {
}

bool BasicPathTracer::Initialize(LPCTSTR mainWndCaption, uint32 screenWidth, uint32 screenHeight, bool fullscreen) {
	// Initialize the Engine
	if (!Engine::RAPTEngine::Initialize(mainWndCaption, screenWidth, screenHeight, fullscreen)) {
		return false;
	}

	m_hdrTextureD3D = new Graphics::D3DTexture2D(m_device, screenWidth, screenHeight, DXGI_FORMAT_R32G32B32A32_FLOAT, D3D11_BIND_SHADER_RESOURCE, 1);
	m_hdrTextureCuda = new Graphics::CudaTexture2D(screenWidth, screenHeight, 16u);
	m_hdrTextureCuda->RegisterResource(m_hdrTextureD3D->GetTexture(), cudaGraphicsRegisterFlagsNone);
	m_hdrTextureCuda->MemSet(0);

	Graphics::LoadVertexShader(L"fullscreen_triangle_vs.cso", m_device, &m_fullscreenTriangleVS);
	Graphics::LoadPixelShader(L"copy_cuda_output_to_backbuffer_ps.cso", m_device, &m_copyCudaOutputToBackbufferPS);

	// Create the constant buffer for the pixel shader
	D3D11_BUFFER_DESC bufferDesc;
	bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	bufferDesc.MiscFlags = 0;
	bufferDesc.StructureByteStride = 0;

	bufferDesc.ByteWidth = static_cast<uint>(Graphics::CBSize(sizeof(float)));
	m_device->CreateBuffer(&bufferDesc, nullptr, &m_copyCudaOutputPSConstantBuffer);

	// Bind the shaders and SRV to the pipeline
	// We never bind anything else, so we can just do it once during initialization
	m_immediateContext->VSSetShader(m_fullscreenTriangleVS, nullptr, 0u);
	m_immediateContext->PSSetShader(m_copyCudaOutputToBackbufferPS, nullptr, 0u);

	ID3D11ShaderResourceView *hdrSRV = m_hdrTextureD3D->GetShaderResource();
	m_immediateContext->PSSetShaderResources(0, 1, &hdrSRV);

	m_immediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	// Create the scene
	// TODO: Use a scene description file rather than hard code the scene
	// (Can probably steal the format used by The Halfling Engine)
	Scene::Sphere spheres[5];
	spheres[0] = {make_float3(0.0f, 0.0f, 0.0f), 4.0f};
	spheres[1] = {make_float3(-4.0f, 0.0f, 0.0f), 4.0f};
	spheres[2] = {make_float3(-8.0f, 0.0f, 0.0f), 4.0f};
	spheres[3] = {make_float3(4.0f, 0.0f, 0.0f), 4.0f};
	spheres[4] = {make_float3(8.0f, 0.0f, 0.0f), 4.0f};

	CE(cudaMalloc(&d_spheres, 5 * sizeof(Scene::Sphere)));
	CE(cudaMemcpy(d_spheres, &spheres, 5 * sizeof(Scene::Sphere), cudaMemcpyHostToDevice));
	m_numSpheres = 5;

	return true;
}

void BasicPathTracer::Shutdown() {
	// Release in the opposite order we initialized in

	Engine::RAPTEngine::Shutdown();
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



} // End of namespace BasicPathTracer
