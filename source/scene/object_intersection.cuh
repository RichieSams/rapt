/* rapt - RichieSam's Adventures in Path Tracing
 *
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#pragma once

#include "scene/scene_objects.h"

#include <graphics/helper_math.h>
#include <vector_types.h>


/**
 * Test for the intersection of a ray with a sphere
 *
 * NOTE: Source adapted from Scratchapixel.com Lesson 7 - Intersecting Simple Shapes
 *       http://www.scratchapixel.com/old/lessons/3d-basic-lessons/lesson-7-intersecting-simple-shapes/ray-sphere-intersection/
 *
 * @param ray           The ray to test
 * @param sphere        The sphere to test
 * @param normal_out    Filled with normal of the surface at the intersection point. Not changed if no intersection.
 * @return              The distance from the ray origin to the nearest intersection. -1.0f if no intersection
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

	// See http://www.scratchapixel.com/old/assets/Uploads/Lesson007/l007-raysphereisect1.png for definition of 'thc'
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

		#ifdef BACKFACE_CULL_SPHERES
			return -1.0f;
		#else
			// Return the far side of the sphere
			nearestIntersection = max(firstIntersection, secondIntersection);

			// We reverse the direction of the normal, since we are inside the sphere
			normalDirection = -1.0f;
		#endif
	}

	normal_out = normalize(((ray.Origin + (ray.Direction * nearestIntersection)) - sphere.Center) * normalDirection);

	return nearestIntersection;
}


/**
 * Test for the intersection of a ray with a plane   
 *
 * @param ray           The ray to test
 * @param plane         The plane to test
 * @param normal_out    Filled with normal of the surface at the intersection point. Not changed if no intersection.
 * @return              The distance from the ray origin to the nearest intersection. -1.0f if no intersection
 */
__device__ float TestRayPlaneIntersection(Scene::Ray &ray, Scene::Plane &plane, float3 &normal_out) {
    float denominator = dot(plane.Normal, ray.Direction);

	// If dot product between the vectors is greater than -epison,
	// the ray is perpendicular to the plane or points away from the plane normal
	if (denominator > -1.0e-6f) {
		return -1.0f;
	}

	normal_out = plane.Normal;
	return dot(plane.Normal, plane.Point - ray.Origin) / denominator;
}
