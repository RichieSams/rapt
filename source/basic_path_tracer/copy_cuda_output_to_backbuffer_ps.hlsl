/* raic - RichieSam's Adventures in Cuda
 *
 * raic is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "basic_path_tracer/types.hlsli"


Texture2D<float3> gHDRInput : register(t0);

float4 CopyCudaOutputToBackbufferPS(CalculatedTrianglePixelIn input) : SV_TARGET {
	return float4(gHDRInput[input.positionClip.xy], 1.0f);
}
