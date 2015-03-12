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
#include "scene/materials.h"
#include "scene/device_camera.h"


namespace BasicPathTracer {

bool BasicPathTracer::Initialize(LPCTSTR mainWndCaption, uint32 screenWidth, uint32 screenHeight, bool fullscreen) {
	// Initialize the Engine
	if (!Engine::RAPTEngine::Initialize(mainWndCaption, screenWidth, screenHeight, fullscreen)) {
		return false;
	}

	// Create the accumulation buffer in DirectX and in CUDA
	//
	// We have to use a RGBA format since CUDA-DirectX interop doesn't support R32G32B32_FLOAT
	m_hdrTextureD3D = new Graphics::D3DTexture2D(m_device, screenWidth, screenHeight, DXGI_FORMAT_R32G32B32A32_FLOAT, D3D11_BIND_SHADER_RESOURCE, 1);
	m_hdrTextureCuda = new Graphics::CudaTexture2D(screenWidth, screenHeight, 4 /* RGBA */ * sizeof(float));
	m_hdrTextureCuda->MemSet(0);

	// Register the DirextX version of the buffer with the CUDA version so we can copy between them later
	m_hdrTextureCuda->RegisterResource(m_hdrTextureD3D->GetTexture(), cudaGraphicsRegisterFlagsNone);


	// Create the shaders and buffers needed for DirectX
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
	// We can do this just once for the application since we never bind anything else
	m_immediateContext->VSSetShader(m_fullscreenTriangleVS, nullptr, 0u);
	m_immediateContext->PSSetShader(m_copyCudaOutputToBackbufferPS, nullptr, 0u);

	ID3D11ShaderResourceView *hdrSRV = m_hdrTextureD3D->GetShaderResource();
	m_immediateContext->PSSetShaderResources(0, 1, &hdrSRV);

	m_immediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);


	// Create the scene
	CreateScene();


	// Allocate memory for the camera on the GPU
	CE(cudaMalloc(&d_deviceCamera, sizeof(DeviceCamera)));


	return true;
}

void BasicPathTracer::Shutdown() {
	// Release in the opposite order we initialized in
	cudaFree(d_deviceCamera);
	cudaFree(d_spheres);

	ReleaseCOM(m_copyCudaOutputPSConstantBuffer);
	ReleaseCOM(m_copyCudaOutputToBackbufferPS);
	ReleaseCOM(m_fullscreenTriangleVS);
	
	delete m_hdrTextureCuda;
	delete m_hdrTextureD3D;

	Engine::RAPTEngine::Shutdown();
}

void BasicPathTracer::CreateScene() {
	// TODO: Use a scene description file rather than hard code the scene
	// (Can probably steal the format used by The Halfling Engine)

	// Create the scene and copy the data over to the GPU
	Scene::LambertMaterial materials[4];
	materials[0] = {make_float3(0.8f, 0.8f, 0.8f)};
	materials[1] = {make_float3(1.0f, 0.0f, 0.0f)};
	materials[2] = {make_float3(0.0f, 1.0f, 0.0f)};
	materials[3] = {make_float3(0.0f, 0.0f, 1.0f)};

	CE(cudaMalloc(&d_materials, 9 * sizeof(Scene::LambertMaterial)));
	CE(cudaMemcpy(d_materials, &materials, 9 * sizeof(Scene::LambertMaterial), cudaMemcpyHostToDevice));


	Scene::Plane ground = {make_float3(0.0f, -5.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f), 0u};

	CE(cudaMalloc(&d_planes, 9 * sizeof(Scene::Plane)));
	CE(cudaMemcpy(d_planes, &ground, 9 * sizeof(Scene::Plane), cudaMemcpyHostToDevice));


	Scene::Sphere spheres[9];
	spheres[0] = {make_float3(0.0f, 0.0f, 0.0f), 4.0f, 0u};
	spheres[1] = {make_float3(-3.0f, -3.0f, -3.0f), 4.0f, 2u};
	spheres[2] = {make_float3(-3.0f, -3.0f, 3.0f), 4.0f, 3u};
	spheres[3] = {make_float3(-3.0f, 3.0f, -3.0f), 4.0f, 1u};
	spheres[4] = {make_float3(-3.0f, 3.0f, 3.0f), 4.0f, 2u};
	spheres[5] = {make_float3(3.0f, -3.0f, -3.0f), 4.0f, 3u};
	spheres[6] = {make_float3(3.0f, -3.0f, 3.0f), 4.0f, 1u};
	spheres[7] = {make_float3(3.0f, 3.0f, -3.0f), 4.0f, 2u};
	spheres[8] = {make_float3(3.0f, 3.0f, 3.0f), 4.0f, 3u};

	CE(cudaMalloc(&d_spheres, 9 * sizeof(Scene::Sphere)));
	CE(cudaMemcpy(d_spheres, &spheres, 9 * sizeof(Scene::Sphere), cudaMemcpyHostToDevice));


	// Store the representation of the scene in a single object
	// CUDA only allows 256 bytes of data to be passed as arguments in the kernel launch
	// We can save some room by bundling all the scene variables together
	Scene::SceneObjects sceneObjects;
	sceneObjects.Materials = d_materials;
	sceneObjects.Planes = d_planes;
	sceneObjects.NumPlanes = 1u;
	sceneObjects.Spheres = d_spheres;
	sceneObjects.NumSpheres = 9u;

	CE(cudaMalloc(&d_sceneObjects, sizeof(Scene::SceneObjects)));
	CE(cudaMemcpy(d_sceneObjects, &sceneObjects, sizeof(Scene::SceneObjects), cudaMemcpyHostToDevice));
}

} // End of namespace BasicPathTracer
