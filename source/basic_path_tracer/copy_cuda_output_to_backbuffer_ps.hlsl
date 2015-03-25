/* rapt - RichieSam's Adventures in Path Tracing
 *
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "basic_path_tracer/types.hlsli"

cbuffer constants {
	float gInverseNumPasses;
	float gExposure;
};


Texture2D<float3> gHDRInput : register(t0);


float4 CopyCudaOutputToBackbufferPS(CalculatedTrianglePixelIn input) : SV_TARGET {
	float3 pixelSample = gHDRInput[input.positionClip.xy];
	pixelSample *= gInverseNumPasses; // Normalize, since we're doing progressive rendering
	pixelSample *= gExposure;  // Hardcoded exposure adjustment

	return float4(pixelSample, 1.0f);
}
