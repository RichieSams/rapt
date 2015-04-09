/* rapt - RichieSam's Adventures in Path Tracing
 *
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "basic_path_tracer/path_trace.cuh"

#include "scene/device_camera.h"
#include "scene/ray_creation.cuh"
#include "scene/materials.h"

#define BACKFACE_CULL_SPHERES
#include "scene/object_intersection.cuh"

#include <device_launch_parameters.h>
#include <float.h>

#define IMPORTANCE_SAMPLE


__global__ void PathTraceKernel(unsigned char *textureData, uint width, uint height, size_t pitch, DeviceCamera *g_camera, Scene::SceneObjects *g_sceneObjects, uint hashedFrameNumber) {
	// Create a local copy of the arguments
	DeviceCamera camera = *g_camera;
	Scene::SceneObjects sceneObjects = *g_sceneObjects;

	// Global threadId
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	// Create random number generator
	curandState randState;
	curand_init(hashedFrameNumber + threadId, 0, 0, &randState);


	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Calculate the first ray for this pixel
	Scene::Ray ray = {camera.Origin, CalculateRayDirectionFromPixel(x, y, width, height, camera, &randState)};


	float3 pixelColor = make_float3(0.0f, 0.0f, 0.0f);
	float3 accumulatedMaterialColor = make_float3(1.0f, 1.0f, 1.0f);

	// Bounce the ray around the scene
	for (uint bounces = 0; bounces < 10; ++bounces) {
		// Initialize the intersection variables
		float closestIntersection = FLT_MAX;
		float3 normal;
		Scene::LambertMaterial material;

		TestSceneIntersection(ray, sceneObjects, &closestIntersection, &normal, &material);

		// Find out if we hit anything
		if (closestIntersection < FLT_MAX) {
			// We hit an object

			// Add the emmisive light
			pixelColor += accumulatedMaterialColor * material.EmmisiveColor;


			// Shoot a new ray

			// Set the origin at the intersection point
			ray.Origin = ray.Origin + ray.Direction * closestIntersection;
			// Offset the origin to prevent self intersection
			ray.Origin += normal * 0.001f;

			// Choose the direction based on the material
			if (material.MaterialType == Scene::MATERIAL_TYPE_DIFFUSE) {
				#ifdef IMPORTANCE_SAMPLE
					ray.Direction = CreateCosineWeightedDirectionInHemisphere(normal, &randState);

					// Accumulate the diffuse/specular color
					accumulatedMaterialColor *= material.MainColor; // * dot(ray.Direction, normal) / PI     // Cancels with pdf

					// Divide by the pdf
					//accumulatedMaterialColor /= dot(ray.Direction, normal) / PI
				#else
					ray.Direction = CreateUniformDirectionInHemisphere(normal, &randState);

					// Accumulate the diffuse/specular color
					accumulatedMaterialColor *= material.MainColor /* * (1 / PI)  <- this cancels with the PI in the pdf */ * dot(ray.Direction, normal);

					// Divide by the pdf
					accumulatedMaterialColor *= 2.0f; // pdf == 1 / (2 * PI)
				#endif
			} else if (material.MaterialType == Scene::MATERIAL_TYPE_SPECULAR) {
				ray.Direction = reflect(ray.Direction, normal);

				// Accumulate the diffuse/specular color
				accumulatedMaterialColor *= material.MainColor;
			}

			
			

			// Russian Roulette
			if (bounces > 3) {
				float p = max(accumulatedMaterialColor.x, max(accumulatedMaterialColor.y, accumulatedMaterialColor.z));
				if (curand_uniform(&randState) > p) {
					return;
				}
				accumulatedMaterialColor *= 1 / p;
			}
		} else {
			// We didn't hit anything, return the sky color
			pixelColor += accumulatedMaterialColor * make_float3(0.846f, 0.933f, 0.949f);

			break;
		}
	}


	if (x < width && y < height) {
		// Get a pointer to the pixel at (x,y)
		float *pixel = (float *)(textureData + y * pitch) + 4 /*RGBA*/ * x;

		// Write pixel data
		pixel[0] += pixelColor.x;
		pixel[1] += pixelColor.y;
		pixel[2] += pixelColor.z;
		// Ignore alpha, since it's hardcoded to 1.0f in the display
		// We have to use a RGBA format since CUDA-DirectX interop doesn't support R32G32B32_FLOAT
	}
}
