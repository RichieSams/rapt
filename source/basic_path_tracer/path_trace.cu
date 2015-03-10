/* rapt - RichieSam's Adventures in Path Tracing
 *
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "basic_path_tracer/path_trace.cuh"

#include "scene/scene_objects.h"

#include <device_launch_parameters.h>
#include <graphics/helper_math.h>
#include <curand_kernel.h>
#include <vector_types.h>

#include <float.h>


__device__ float3 CalculateRayDirectionFromPixel(uint x, uint y, uint width, uint height, DeviceCamera &camera, curandState *randState) {
	float3 viewVector = make_float3((((x + curand_uniform(randState)) / width) * 2.0f - 1.0f) * camera.TanFovDiv2_X,
	                                -(((y + curand_uniform(randState)) / height) * 2.0f - 1.0f) * camera.TanFovDiv2_Y,
	                                1.0f);

	// Matrix multiply
	return normalize(make_float3(dot(viewVector, camera.ViewToWorldMatrixR0),
	                             dot(viewVector, camera.ViewToWorldMatrixR1),
	                             dot(viewVector, camera.ViewToWorldMatrixR2)));
}

/**
 * Test for the intersection of a ray with a sphere
 *
 * NOTE: Source adapted from Scratchapixel.com Lesson 7 - Intersecting Simple Shapes
 *       http://www.scratchapixel.com/old/lessons/3d-basic-lessons/lesson-7-intersecting-simple-shapes/ray-sphere-intersection/
 *
 * @param ray       The ray
 * @param sphere    The sphere
 * @return          The distance from the ray origin to the nearest intersection. -1.0f if no intersection
 */
__device__ float TestRaySphereIntersection(Scene::Ray &ray, Scene::Sphere &sphere, float3 &normal_out) {
	float3 L = sphere.Center - ray.Origin;
    float projectedRay = dot(L, ray.Direction);

	// Ray points away from the sphere
	if (projectedRay < 0) {
		return -1.0f;
	}

    float distanceToRaySquared = dot(L, L) - projectedRay * projectedRay;

	// Ray misses the sphere
    if (distanceToRaySquared > sphere.RadiusSquared) {
		return -1.0f;
	}

	// See http://www.scratchapixel.com/old/assets/Uploads/Lesson007/l007-raysphereisect1.png for definition of thc
    float thc = sqrt(sphere.RadiusSquared - distanceToRaySquared);

    float firstIntersection = projectedRay - thc;
    float secondIntersection = projectedRay + thc;

	float nearestIntersection;
	float normalDirection;
	if (firstIntersection > 0 && secondIntersection > 0) {
		// Two intersections
		// Return the nearest of the two
		nearestIntersection = min(firstIntersection, secondIntersection);

		normalDirection = 1.0f;
	} else {
		// Ray starts inside the sphere
		// Return the far side of the sphere
		nearestIntersection = max(firstIntersection, secondIntersection);

		// We reverse the direction of the normal, since we are inside the sphere
		normalDirection = -1.0f;
	}

	normal_out = normalize(((ray.Origin + (ray.Direction * nearestIntersection)) - sphere.Center) * normalDirection);

	return nearestIntersection;
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

	// Generate a uniform random number
	//float randNum = curand_uniform(&randState);

	// Try to intersect with the spheres;
	float closestIntersection = FLT_MAX;
	float3 normal;
	for (uint i = 0; i < numSpheres; ++i) {
		float3 newNormal;
		float intersection = TestRaySphereIntersection(ray, g_spheres[i], newNormal);
		if (intersection > 0.0f && intersection < closestIntersection) {
			closestIntersection = intersection;
			normal = newNormal;
		}
	}

	float3 pixelColor;
	if (closestIntersection < FLT_MAX) {
		float attentuation = max(dot(normal, make_float3(0.70710678118f, 0.70710678118f, -0.70710678118f)), 0.0f);
		pixelColor = make_float3(0.846, 0.933, 0.949) * attentuation + make_float3(0.15f, 0.15f, 0.15f);
	} else {
		pixelColor = make_float3(0.0f, 0.0f, 0.0f);
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
