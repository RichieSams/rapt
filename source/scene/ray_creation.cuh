/* rapt - RichieSam's Adventures in Path Tracing
 *
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#pragma once

#include "scene/scene_objects.h"

#include <graphics/helper_math.h>
#include <vector_types.h>

#include <curand_kernel.h>


#define TWO_PI 6.283185307f

__device__ float3 CalculateRayDirectionFromPixel(uint x, uint y, uint width, uint height, DeviceCamera &camera, curandState *randState) {
	float3 viewVector = make_float3((((x + curand_uniform(randState)) / width) * 2.0f - 1.0f) * camera.TanFovXDiv2,
	                                -(((y + curand_uniform(randState)) / height) * 2.0f - 1.0f) * camera.TanFovYDiv2,
	                                1.0f);

	// Matrix multiply
	return normalize(make_float3(dot(viewVector, camera.ViewToWorldMatrixR0),
	                             dot(viewVector, camera.ViewToWorldMatrixR1),
	                             dot(viewVector, camera.ViewToWorldMatrixR2)));
}

__device__ float3 CreateRandomDirectionInNormalHemisphere(float3 normal, curandState *randState) {
	// Create a random coordinate in spherical space
	float randAngle = TWO_PI * curand_uniform(randState);
	float randDistance = curand_uniform(randState);
	float distanceFromCenter = sqrt(randDistance);
	
	// Find an axis that is not parallel to normal.x
	float3 majorAxis = abs(normal.x > 0.1f) ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);

	// Use majorAxis to create a coordinate system relative to world space
	float3 u = normalize(cross(majorAxis, normal));
	float3 v = cross(normal, u);
	float3 w = normal;

	// Transform from spherical coordinates to the cartesian coordinates space
	// we just defined above, then use the definition to transform to world space
	return normalize(u * cos(randAngle) * distanceFromCenter +
	                 v * sin(randAngle) * distanceFromCenter +
	                 w * sqrt(1.0f - randDistance));
}