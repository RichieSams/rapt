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
	// NOTE: See http://richiesams.blogspot.com/2015/03/shooting-objects-from-across-way.html for explaination of algorithm

	float3 L = sphere.Center - ray.Origin;
    float t_ca = dot(L, ray.Direction);

	// Ray points away from the sphere
	if (t_ca < 0) {
		return -1.0f;
	}

    float d_squared = dot(L, L) - t_ca * t_ca;

	// Ray misses the sphere
    if (d_squared > sphere.RadiusSquared) {
		return -1.0f;
	}

    float t_hc = sqrt(sphere.RadiusSquared - d_squared);

    float t_0 = t_ca - t_hc;
    float t_1 = t_ca + t_hc;

	float nearestIntersection;
	float normalDirection;
	if (t_0 > 0 && t_1 > 0) {
		// Two intersections
		// Return the nearest of the two
		nearestIntersection = min(t_0, t_1);

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

	// If dot product between the vectors is greater than 0.0f,
	// the ray is perpendicular to the plane or points away from the plane normal
	if (denominator > 0.0f) {
		return -1.0f;
	}

	normal_out = plane.Normal;
	return dot(plane.Normal, plane.Point - ray.Origin) / denominator;
}


/**
 * Test for the intersection of a ray with a rectangle   
 *
 * @param ray           The ray to test
 * @param rectangle     The rectangle to test
 * @param normal_out    Filled with normal of the surface at the intersection point. Not changed if no intersection.
 * @return              The distance from the ray origin to the nearest intersection. -1.0f if no intersection
 */
__device__ float TestRayRectangleIntersection(Scene::Ray &ray, Scene::Rectangle &rectangle, float3 &normal_out) {
	// Test if the ray intersects the plane that the rectangle is on
	float denominator = dot(rectangle.Normal, ray.Direction);

	// If dot product between the vectors is greater than 0.0f,
	// the ray is perpendicular to the plane or points away from the plane normal
	if (denominator > 0.0f) {
		return -1.0f;
	}

	// Calculate the intersection point with the plane
	float intersectionDistance = dot(rectangle.Normal, rectangle.Point - ray.Origin) / denominator;
	float3 intersectionPoint = ray.Origin + ray.Direction * intersectionDistance;


	// Test if the intersection point is inside the rectangle

	float3 vectorToIntersection = intersectionPoint - rectangle.Point;
	
	float leg1_lengthSquared = dot(rectangle.Leg1, rectangle.Leg1);
	float leg2_lengthSquared = dot(rectangle.Leg2, rectangle.Leg2);

	// Project the intersection point onto each leg
	// NOTE: leg*ProjectionRatio will be negative if the outside the rectangle
	//       We could check for it here, but the saved computations aren't worth
	//       the extra branch
	float leg1ProjectionRatio = dot(vectorToIntersection, rectangle.Leg1) / leg1_lengthSquared;
	float leg2ProjectionRatio = dot(vectorToIntersection, rectangle.Leg2) / leg2_lengthSquared;

	// Use the ratios to create a projected vector for each leg
	float3 projectionOntoLeg1 = leg1ProjectionRatio * rectangle.Leg1;
	float3 projectionOntoLeg2 = leg2ProjectionRatio * rectangle.Leg2;


	// Dot the vectors with themselves to get their length squared
	float projection1LengthSquared = dot(projectionOntoLeg1, projectionOntoLeg1);
	float projection2LengthSquared = dot(projectionOntoLeg2, projectionOntoLeg2);

	// Do the final comparison
	// A ray is inside the rectangle if the projections are positive and shorter than the legs
	if (leg1ProjectionRatio >= 0.0f && projection1LengthSquared <= leg1_lengthSquared &&
		leg2ProjectionRatio >= 0.0f && projection2LengthSquared <= leg2_lengthSquared) {
		normal_out = rectangle.Normal;
		return intersectionDistance;
	} else {
		return -1.0f;
	}
}


/**
 * Test for the intersection of a ray with a circle   
 *
 * @param ray           The ray to test
 * @param circle        The circle to test
 * @param normal_out    Filled with normal of the surface at the intersection point. Not changed if no intersection.
 * @return              The distance from the ray origin to the nearest intersection. -1.0f if no intersection
 */
__device__ float TestRayCircleIntersection(Scene::Ray &ray, Scene::Circle &circle, float3 &normal_out) {
	// Test if the ray intersects the plane that the circle is on
	float denominator = dot(circle.Normal, ray.Direction);

	// If dot product between the vectors is greater than 0.0f,
	// the ray is perpendicular to the plane or points away from the plane normal
	if (denominator > 0.0f) {
		return -1.0f;
	}

	// Calculate the intersection point with the plane
	float intersectionDistance = dot(circle.Normal, circle.Point - ray.Origin) / denominator;
	float3 intersectionPoint = ray.Origin + ray.Direction * intersectionDistance;


	// Test if the intersection point is inside the circle

	float3 vectorToIntersection = intersectionPoint - circle.Point;
	float distanceSquared = dot(vectorToIntersection, vectorToIntersection);

	// A ray is inside the circle if the distance from the center of the circle to the intersection point
	// is less than the radius of the circle
	if (distanceSquared < circle.RadiusSquared) {
		normal_out = circle.Normal;
		return intersectionDistance;
	} else {
		return -1.0f;
	}
}


__device__ void TestSceneIntersection(Scene::Ray &ray, Scene::SceneObjects &sceneObjects, float *closestIntersection, float3 *normal, Scene::LambertMaterial *material) {
	// Try to intersect with the planes
	for (uint j = 0; j < sceneObjects.NumPlanes; ++j) {
		// Make a local copy
		Scene::Plane plane = sceneObjects.Planes[j];

		float3 newNormal;
		float intersection = TestRayPlaneIntersection(ray, plane, newNormal);
		if (intersection > 0.0f && intersection < *closestIntersection) {
			*closestIntersection = intersection;
			*normal = newNormal;
			*material = sceneObjects.Materials[plane.MaterialId];
		}
	}

	// Try to intersect with the rectangles;
	for (uint j = 0; j < sceneObjects.NumRectangles; ++j) {
		// Make a local copy
		Scene::Rectangle rectangle = sceneObjects.Rectangles[j];

		float3 newNormal;
		float intersection = TestRayRectangleIntersection(ray, rectangle, newNormal);
		if (intersection > 0.0f && intersection < *closestIntersection) {
			*closestIntersection = intersection;
			*normal = newNormal;
			*material = sceneObjects.Materials[rectangle.MaterialId];
		}
	}

	// Try to intersect with the circles;
	for (uint j = 0; j < sceneObjects.NumCircles; ++j) {
		// Make a local copy
		Scene::Circle circle = sceneObjects.Circles[j];

		float3 newNormal;
		float intersection = TestRayCircleIntersection(ray, circle, newNormal);
		if (intersection > 0.0f && intersection < *closestIntersection) {
			*closestIntersection = intersection;
			*normal = newNormal;
			*material = sceneObjects.Materials[circle.MaterialId];
		}
	}

	// Try to intersect with the spheres;
	for (uint j = 0; j < sceneObjects.NumSpheres; ++j) {
		// Make a local copy
		Scene::Sphere sphere = sceneObjects.Spheres[j];

		float3 newNormal;
		float intersection = TestRaySphereIntersection(ray, sphere, newNormal);
		if (intersection > 0.0f && intersection < *closestIntersection) {
			*closestIntersection = intersection;
			*normal = newNormal;
			*material = sceneObjects.Materials[sphere.MaterialId];
		}
	}
}
