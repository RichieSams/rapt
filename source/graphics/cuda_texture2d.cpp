/* raic - RichieSam's Adventures in Cuda
 *
 * raic is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "graphics/cuda_texture2d.h"

#include <cassert>

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>


namespace Graphics {

CudaTexture2D::CudaTexture2D(uint width, uint height, uint bytesPerPixel)
		: m_width(width), 
		  m_height(height),
		  m_bytesPerPixel(bytesPerPixel),
		  m_graphicsResource(nullptr) {
	CE(cudaMallocPitch(&m_textureData, &m_texturePitch, width * bytesPerPixel, height));
}

CudaTexture2D::~CudaTexture2D() {
	if (m_graphicsResource) {
		CE(cudaGraphicsUnregisterResource(m_graphicsResource));
	}

	CE(cudaFree(m_textureData));
}

void CudaTexture2D::RegisterResource(ID3D11Resource *d3dResource, cudaGraphicsRegisterFlags flags) {
	// Release the previously registered resource
	if (m_graphicsResource) {
		CE(cudaGraphicsUnregisterResource(m_graphicsResource));
	}

	CE(cudaGraphicsD3D11RegisterResource(&m_graphicsResource, d3dResource, flags));
}

void CudaTexture2D::CopyTextureDataToRegisteredResource() {
	assert(m_graphicsResource);

	cudaArray *cuArray;
	CE(cudaGraphicsSubResourceGetMappedArray(&cuArray, m_graphicsResource, 0, 0));

	CE(cudaMemcpy2DToArray(
		cuArray, // dst array
		0, 0,    // offset
		m_textureData, m_texturePitch,       // src
		m_width * m_bytesPerPixel, m_height, // extent
		cudaMemcpyDeviceToDevice));
}

} // End of namespace Graphics