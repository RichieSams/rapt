/* raic - RichieSam's Adventures in Cuda
 *
 * raic is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#pragma once

#include "common/typedefs.h"

#include <cuda_runtime_api.h>



struct ID3D11Resource;
struct cudaGraphicsResource;

namespace Graphics {

class CudaTexture2D {
public:
	CudaTexture2D(uint width, uint height, uint bytesPerPixel);
	~CudaTexture2D();

private:
	uint m_width;
	uint m_height;
	uint m_bytesPerPixel;

	void *m_textureData;
	size_t m_texturePitch;
	cudaGraphicsResource *m_graphicsResource;

public:
	void RegisterResource(ID3D11Resource *d3dResource, cudaGraphicsRegisterFlags flags);
	void CopyTextureDataToRegisteredResource();

	inline void *GetTextureData() { return m_textureData; }
	inline size_t GetTexturePitch() { return m_texturePitch; }
	inline cudaGraphicsResource *GetCurrentGraphicsResource() { return m_graphicsResource; }
};

} // End of namespace Graphics