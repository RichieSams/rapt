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
	m_exposure = 1.0f;

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

	#define NUM_MATERIALS 6
	Scene::LambertMaterial materials[NUM_MATERIALS];
	materials[0] = {Scene::MATERIAL_TYPE_DIFFUSE, make_float3(0.9f, 0.9f, 0.9f), make_float3(0.0f, 0.0f, 0.0f)};
	materials[1] = {Scene::MATERIAL_TYPE_DIFFUSE, make_float3(0.408f, 0.741f, 0.467f), make_float3(0.0f, 0.0f, 0.0f)};
	materials[2] = {Scene::MATERIAL_TYPE_DIFFUSE, make_float3(0.392f, 0.584f, 0.929f), make_float3(0.0f, 0.0f, 0.0f)};
	materials[3] = {Scene::MATERIAL_TYPE_DIFFUSE, make_float3(1.0f, 0.498f, 0.314f), make_float3(0.0f, 0.0f, 0.0f)};
	materials[4] = {Scene::MATERIAL_TYPE_DIFFUSE, make_float3(0.4f, 0.4f, 0.4f), make_float3(3.0f, 3.0f, 3.0f)};
	materials[5] = {Scene::MATERIAL_TYPE_SPECULAR, make_float3(1.0f, 1.0f, 1.0f), make_float3(0.0f, 0.0f, 0.0f)};

	CE(cudaMalloc(&d_materials, NUM_MATERIALS * sizeof(Scene::LambertMaterial)));
	CE(cudaMemcpy(d_materials, &materials, NUM_MATERIALS * sizeof(Scene::LambertMaterial), cudaMemcpyHostToDevice));


	#define NUM_PLANES 1
	Scene::Plane planes[NUM_PLANES];
	planes[0] = {make_float3(0.0f, -6.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f), 0u}; // Front
	
	CE(cudaMalloc(&d_planes, NUM_PLANES * sizeof(Scene::Plane)));
	CE(cudaMemcpy(d_planes, &planes, NUM_PLANES * sizeof(Scene::Plane), cudaMemcpyHostToDevice));

	
	#define NUM_RECTANGLES 0
	//Scene::Rectangle rectangles[NUM_RECTANGLES];
	//rectangles[0] = {make_float3(0.0f, 0.0f, -20.0f), make_float3(0.0f, 0.0f, 1.0f), make_float3(20.0f, 0.0f, 0.0f), make_float3(0.0f, 20.0f, 0.0f), 0u}; // Front
	//rectangles[1] = {make_float3(-20.0f, 0.0f, 0.0f), make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 20.0f), make_float3(0.0f, 20.0f, 0.0f), 1u}; // Left
	//rectangles[2] = {make_float3(0.0f, -20.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f), make_float3(20.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 20.0f), 0u}; // Bottom
	//rectangles[3] = {make_float3(0.0f, 0.0f, 20.0f), make_float3(0.0f, 0.0f, -1.0f), make_float3(20.0f, 0.0f, 0.0f), make_float3(0.0f, 20.0f, 0.0f), 0u}; // Back
	//rectangles[4] = {make_float3(20.0f, 0.0f, 0.0f), make_float3(-1.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 20.0f), make_float3(0.0f, 20.0f, 0.0f), 3u}; // Right
	//rectangles[5] = {make_float3(0.0f, 20.0f, 0.0f), make_float3(0.0f, -1.0f, 0.0f), make_float3(20.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 20.0f), 0u}; // Top

	//CE(cudaMalloc(&d_rectangles, NUM_RECTANGLES * sizeof(Scene::Rectangle)));
	//CE(cudaMemcpy(d_rectangles, &rectangles, NUM_RECTANGLES * sizeof(Scene::Rectangle), cudaMemcpyHostToDevice));


	#define NUM_CIRCLES 0
	//Scene::Circle circles[NUM_CIRCLES];
	//circles[0] = {make_float3(0.0f, 19.99f, 0.0f), make_float3(0.0f, -1.0f, 0.0f), 8.0f, 4u}; // Light

	//CE(cudaMalloc(&d_circles, NUM_CIRCLES * sizeof(Scene::Circle)));
	//CE(cudaMemcpy(d_circles, &circles, NUM_CIRCLES * sizeof(Scene::Circle), cudaMemcpyHostToDevice));


	#define NUM_SPHERES 9
	Scene::Sphere spheres[NUM_SPHERES];
	spheres[0] = {make_float3(0.0f, 0.0f, 0.0f), 4.0f, 4u};
	spheres[1] = {make_float3(-4.0f, -4.0f, -4.0f), 4.0f, 1u};
	spheres[2] = {make_float3(-4.0f, -4.0f, 4.0f), 4.0f, 2u};
	spheres[3] = {make_float3(-4.0f, 4.0f, -4.0f), 4.0f, 3u};
	spheres[4] = {make_float3(-4.0f, 4.0f, 4.0f), 4.0f, 5u};
	spheres[5] = {make_float3(4.0f, -4.0f, -4.0f), 4.0f, 5u};
	spheres[6] = {make_float3(4.0f, -4.0f, 4.0f), 4.0f, 3u};
	spheres[7] = {make_float3(4.0f, 4.0f, -4.0f), 4.0f, 2u};
	spheres[8] = {make_float3(4.0f, 4.0f, 4.0f), 4.0f, 1u};

	CE(cudaMalloc(&d_spheres, NUM_SPHERES * sizeof(Scene::Sphere)));
	CE(cudaMemcpy(d_spheres, &spheres, NUM_SPHERES * sizeof(Scene::Sphere), cudaMemcpyHostToDevice));


	// Store the representation of the scene in a single object
	// CUDA only allows 256 bytes of data to be passed as arguments in the kernel launch
	// We can save some room by bundling all the scene variables together
	Scene::SceneObjects sceneObjects;
	sceneObjects.Materials = d_materials;
	sceneObjects.Planes = d_planes;
	sceneObjects.NumPlanes = NUM_PLANES;
	sceneObjects.Rectangles = d_rectangles;
	sceneObjects.NumRectangles = NUM_RECTANGLES;
	sceneObjects.Circles = d_circles;
	sceneObjects.NumCircles = NUM_CIRCLES;
	sceneObjects.Spheres = d_spheres;
	sceneObjects.NumSpheres = NUM_SPHERES;

	CE(cudaMalloc(&d_sceneObjects, sizeof(Scene::SceneObjects)));
	CE(cudaMemcpy(d_sceneObjects, &sceneObjects, sizeof(Scene::SceneObjects), cudaMemcpyHostToDevice));
}

} // End of namespace BasicPathTracer
