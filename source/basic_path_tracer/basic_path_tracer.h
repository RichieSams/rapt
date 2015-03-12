/* rapt - RichieSam's Adventures in Path Tracing
 *
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#pragma once

#include "engine/rapt_engine.h"

#include "scene/host_camera.h"


namespace Scene {
struct LambertMaterial;
struct Plane;
struct Sphere;
struct SceneObjects;
}

namespace Graphics {
class D3DTexture2D;
class CudaTexture2D;
}

namespace BasicPathTracer {

class BasicPathTracer : public Engine::RAPTEngine {
public:
	BasicPathTracer(HINSTANCE hinstance);

private:
	ID3D11RenderTargetView *m_backbufferRTV;
	D3D11_VIEWPORT m_screenViewport;

	Graphics::D3DTexture2D *m_hdrTextureD3D;
	Graphics::CudaTexture2D *m_hdrTextureCuda;

	ID3D11VertexShader *m_fullscreenTriangleVS;
	ID3D11PixelShader *m_copyCudaOutputToBackbufferPS;
	ID3D11Buffer *m_copyCudaOutputPSConstantBuffer;

	Scene::HostCamera m_hostCamera;
	int m_mouseLastPos_X;
	int m_mouseLastPos_Y;
	bool m_cameraMoved;

	uint m_frameNumber;

	// CUDA device variables
	DeviceCamera *d_deviceCamera;

	Scene::LambertMaterial *d_materials;
	Scene::Plane *d_planes;
	Scene::Sphere *d_spheres;
	Scene::SceneObjects *d_sceneObjects;

public:
	bool Initialize(LPCTSTR mainWndCaption, uint32 screenWidth, uint32 screenHeight, bool fullscreen);
	void Shutdown();

private:
	void CreateScene();

	void OnResize();
	void DrawFrame();

	void MouseDown(WPARAM buttonState, int x, int y);
	void MouseUp(WPARAM buttonState, int x, int y);
	void MouseMove(WPARAM buttonState, int x, int y);
	void MouseWheel(int zDelta);

	void ResetAccumulationBuffer();
};

} // End of namespace BasicPathTracer
