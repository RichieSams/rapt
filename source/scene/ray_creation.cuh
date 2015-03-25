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
#define INV_SQRT_THREE 0.5773502691896257645091487805019574556476f 

__device__ float3 CalculateRayDirectionFromPixel(uint x, uint y, uint width, uint height, DeviceCamera &camera, curandState *randState) {
	float3 viewVector = make_float3((((x + curand_uniform(randState)) / width) * 2.0f - 1.0f) * camera.TanFovXDiv2,
	                                -(((y + curand_uniform(randState)) / height) * 2.0f - 1.0f) * camera.TanFovYDiv2,
	                                1.0f);

	// Matrix multiply
	return normalize(make_float3(dot(viewVector, camera.ViewToWorldMatrixR0),
	                             dot(viewVector, camera.ViewToWorldMatrixR1),
	                             dot(viewVector, camera.ViewToWorldMatrixR2)));
}



/**
 * Creates a uniformly random direction in the hemisphere defined by the normal  
 *
 * Based off the algorithm and code in 'Physically Based Rendering, 2nd Ed.' - Pharr & Humphreys
 * Chapter 13 - Example 13.6.1 - Pg. 664
 *
 * @param normal        The normal that defines the hemisphere
 * @param randState     The random state to use for internal random number generation        
 * @return              A uniformly random direction in the hemisphere
 */
__device__ float3 CreateUniformDirectionInHemisphere(float3 normal, curandState *randState) {
	// Create a random coordinate in spherical space
	float z = curand_uniform(randState);
	float r = sqrt(1.0f - z * z);
	float phi = TWO_PI * curand_uniform(randState);
	
	// Find an axis that is not parallel to normal
	float3 majorAxis;
	if (abs(normal.x) < INV_SQRT_THREE) { 
		majorAxis = make_float3(1, 0, 0);
	} else if (abs(normal.y) < INV_SQRT_THREE) { 
		majorAxis = make_float3(0, 1, 0);
	} else {
		majorAxis = make_float3(0, 0, 1);
	}

	// Use majorAxis to create a coordinate system relative to world space
	float3 u = normalize(cross(majorAxis, normal));
	float3 v = cross(normal, u);
	float3 w = normal;

	// The position in local space
	float x = r * cos(phi);
	float y = r * sin(phi);

	// Transform from spherical coordinates to the cartesian coordinates space
	// we just defined above, then use the definition to transform to world space
	return normalize(u * x +
	                 v * y +
	                 w * z);
}


/**
 * Creates a random direction in the hemisphere defined by the normal, weighted by a cosine lobe  
 *
 * Based off the algorithm and code in 'Physically Based Rendering, 2nd Ed.' - Pharr & Humphreys
 * Chapter 13 - Example 13.6.2 & 13.63 - Pgs. 664 to 669
 *
 * @param normal        The normal that defines the hemisphere
 * @param randState     The random state to use for internal random number generation        
 * @return              A cosine weighted random direction in the hemisphere
 */
__device__ float3 CreateCosineWeightedDirectionInHemisphere(float3 normal, curandState *randState) {
	// Create a random coordinate in spherical space
	float theta = TWO_PI * curand_uniform(randState);
	float r = curand_uniform(randState);
	
	// The position in local space
	float x = r * cos(theta);
	float y = r * sin(theta);

	// Find an axis that is not parallel to normal
	float3 majorAxis;
	if (abs(normal.x) < INV_SQRT_THREE) { 
		majorAxis = make_float3(1, 0, 0);
	} else if (abs(normal.y) < INV_SQRT_THREE) { 
		majorAxis = make_float3(0, 1, 0);
	} else {
		majorAxis = make_float3(0, 0, 1);
	}

	// Use majorAxis to create a coordinate system relative to world space
	float3 u = normalize(cross(majorAxis, normal));
	float3 v = cross(normal, u);
	float3 w = normal;

	// Transform from local coordinates to world coordinates
	// Z is projected from the unit disk up to the hemisphere
	//     This creates the cosine weighting. 
	//     This is known as "Mallay's Method"
	return normalize(u * x +
	                 v * y +
	                 w * sqrt(1.0f - x * x - y * y));
}
