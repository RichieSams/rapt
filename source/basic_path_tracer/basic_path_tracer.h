/* rapt - RichieSam's Adventures in Path Tracing
 *
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#pragma once

#include "engine/rapt_engine.h"
#include "engine/camera.h"


namespace Scene {
struct Sphere;
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

	Engine::HostCamera m_hostCamera;

	uint m_frameNumber;

	// CUDA device variables
	Scene::Sphere *d_spheres;
	uint m_numSpheres;

public:
	bool Initialize(LPCTSTR mainWndCaption, uint32 screenWidth, uint32 screenHeight, bool fullscreen);
	void Shutdown();

private:
	void OnResize();
	void DrawFrame();
};

} // End of namespace BasicPathTracer
