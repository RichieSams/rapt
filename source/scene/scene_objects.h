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

struct LambertMaterial;

struct SceneObjects {
	LambertMaterial *Materials;

	Plane *Planes;
	uint NumPlanes;

	Sphere *Spheres;
	uint NumSpheres;
};

} // End of namespace Scene
