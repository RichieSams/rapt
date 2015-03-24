/* rapt - RichieSam's Adventures in Path Tracing
 *
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#pragma once

#include <vector_types.h>


namespace Scene {

struct Ray {
	float3 Origin;
	float3 Direction;
};

struct Sphere {
	float3 Center;
	float RadiusSquared;
	uint MaterialId;
};

struct Plane {
	float3 Point;
	float3 Normal;
	uint MaterialId;
};

struct Rectangle {
	float3 Point;
	float3 Normal; // We could derive this by doing cross(Leg1, Leg2), but this would require 
	               // the end-user to be very careful in choosing the Point and Legs in order
	               // to get the correct direction of the normal
	float3 Leg1;
	float3 Leg2;
	uint MaterialId;
};

struct Circle {
	float3 Point;
	float3 Normal;
	float RadiusSquared;
	uint MaterialId;
};


struct LambertMaterial;

struct SceneObjects {
	LambertMaterial *Materials;

	Plane *Planes;
	uint NumPlanes;

	Rectangle *Rectangles;
	uint NumRectangles;

	Circle *Circles;
	uint NumCircles;

	Sphere *Spheres;
	uint NumSpheres;
};

} // End of namespace Scene
