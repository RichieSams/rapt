/* rapt - RichieSam's Adventures in Path Tracing
 *
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "basic_path_tracer/path_trace.cuh"

#include "scene/device_camera.h"
#define BACKFACE_CULL_SPHERES
#include "scene/object_intersection.cuh"

#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <float.h>

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


__global__ void PathTraceKernel(unsigned char *textureData, uint width, uint height, size_t pitch, DeviceCamera *g_camera, Scene::Sphere *g_spheres, uint numSpheres, uint hashedFrameNumber) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Create a local copy of the camera
	DeviceCamera camera = *g_camera;

	// Global threadId
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	// Create random number generator
	curandState randState;
	curand_init(hashedFrameNumber + threadId, 0, 0, &randState);

	// Calculate the first ray for this pixel
	Scene::Ray ray = {camera.Origin, CalculateRayDirectionFromPixel(x, y, width, height, camera, &randState)};


	float3 pixelColor = make_float3(0.0f, 0.0f, 0.0f);
	float3 accumulatedMaterialColor = make_float3(1.0f, 1.0f, 1.0f);

	for (uint i = 0; i < 20; ++i) {
		// Initialize the intersection variables
		float closestIntersection = FLT_MAX;
		float3 normal;

		// Try to intersect with the ground plane
		{
			Scene::Plane ground = {make_float3(0.0f, -8.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f)};
		
			float3 newNormal;
			float intersection = TestRayPlaneIntersection(ray, ground, newNormal);
			if (intersection > 0.0f && intersection < closestIntersection) {
				closestIntersection = intersection;
				normal = newNormal;
			}
		}

		// Try to intersect with the spheres;
		for (uint i = 0; i < numSpheres; ++i) {
			float3 newNormal;
			float intersection = TestRaySphereIntersection(ray, g_spheres[i], newNormal);
			if (intersection > 0.0f && intersection < closestIntersection) {
				closestIntersection = intersection;
				normal = newNormal;
			}
		}

		if (closestIntersection < FLT_MAX) {
			// We hit an object
			accumulatedMaterialColor *= make_float3(0.8f, 0.8f, 0.8f);

			ray.Origin = ray.Origin + ray.Direction * closestIntersection;
			ray.Direction = CreateRandomDirectionInNormalHemisphere(normal, &randState);
		} else {
			// We didn't hit anything
			// Use the sky color instead and stop bouncing rays
			pixelColor = make_float3(0.846, 0.933, 0.949) * accumulatedMaterialColor;

			break;
		}
	}
	

	if (x < width && y < height) {
		// Get a pointer to the pixel at (x,y)
		float *pixel = (float *)(textureData + y * pitch) + 4 /*RGBA*/ * x;

		// Write out pixel data
		pixel[0] += pixelColor.x;
		pixel[1] += pixelColor.y;
		pixel[2] += pixelColor.z;
		// Ignore alpha, since it's hardcoded to 1.0f in the display
		// We have to use a RGBA format since CUDA-DirectX interop doesn't support R32G32B32_FLOAT
	}
}
