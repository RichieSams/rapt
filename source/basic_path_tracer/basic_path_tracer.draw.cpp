/* raic - RichieSam's Adventures in Cuda
 *
 * raic is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "basic_path_tracer/basic_path_tracer.h"

#include "engine/camera.h"

#include "graphics/cuda_util.h"
#include "graphics/cuda_texture2d.h"
#include "graphics/d3d_texture2d.h"
#include "graphics/d3d_util.h"

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

#include <DirectXColors.h>


void RenderFrame(void *buffer, uint width, uint height, size_t pitch, DeviceCamera &camera, uint frameNumber);


namespace BasicPathTracer {

void BasicPathTracer::DrawFrame() {
	// Map the resources
	cudaGraphicsResource *resource = m_hdrTextureCuda->GetCurrentGraphicsResource();
	CE(cudaGraphicsMapResources(1, &resource));

	// Run the kernel
	RenderFrame(m_hdrTextureCuda->GetTextureData(), m_clientWidth, m_clientHeight, m_hdrTextureCuda->GetTexturePitch(), m_hostCamera.GetDeviceCamera(), m_frameNumber++);

	// Copy the frame over to the d3d texture
	m_hdrTextureCuda->CopyTextureDataToRegisteredResource();

	// Unmap the resources
	CE(cudaGraphicsUnmapResources(1, &resource));


	// Write the constant buffer data
	float invFrameCount = 1.0f / m_frameNumber;

	D3D11_MAPPED_SUBRESOURCE mappedResource;

	// Lock the constant buffer so it can be written to.
	HR(m_immediateContext->Map(m_copyCudaOutputPSConstantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));
	memcpy(mappedResource.pData, &invFrameCount, sizeof(float));
	m_immediateContext->Unmap(m_copyCudaOutputPSConstantBuffer, 0);

	m_immediateContext->PSSetConstantBuffers(0u, 1u, &m_copyCudaOutputPSConstantBuffer);

	// Draw the frame to the screen
	m_immediateContext->Draw(3u, 0u);

	m_swapChain->Present(1u, 0u);
}

} // End of namespace DirectXInterop