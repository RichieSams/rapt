/* raic - RichieSam's Adventures in Cuda
 *
 * raic is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "basic_path_tracer/basic_path_tracer.h"

#include "graphics/cuda_util.h"
#include "graphics/cuda_texture2d.h"
#include "graphics/d3d_texture2d.h"

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

#include <DirectXColors.h>


void RenderFrame(void *buffer, uint width, uint height, size_t pitch, float t);


namespace BasicPathTracer {

void BasicPathTracer::DrawFrame() {
	static float t = 0.0f;

	// Map the resources
	cudaGraphicsResource *resource = m_hdrTextureCuda->GetCurrentGraphicsResource();
	CE(cudaGraphicsMapResources(1, &resource));

	// Run the kernel 
	RenderFrame(m_hdrTextureCuda->GetTextureData(), m_clientWidth, m_clientHeight, m_hdrTextureCuda->GetTexturePitch(), t);

	// Copy the frame over to the d3d texture
	m_hdrTextureCuda->CopyTextureDataToRegisteredResource();

	// Unmap the resources
	CE(cudaGraphicsUnmapResources(1, &resource));


	// Draw the frame to the screen
	m_immediateContext->VSSetShader(m_fullscreenTriangleVS, nullptr, 0u);
	m_immediateContext->PSSetShader(m_copyCudaOutputToBackbufferPS, nullptr, 0u);

	ID3D11ShaderResourceView *hdrSRV = m_hdrTextureD3D->GetShaderResource();
	m_immediateContext->PSSetShaderResources(0, 1, &hdrSRV);

	m_immediateContext->Draw(3u, 0u);

	m_swapChain->Present(1u, 0u);

	t += 0.1f;
}

} // End of namespace DirectXInterop