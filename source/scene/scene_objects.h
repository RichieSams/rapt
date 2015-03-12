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
};

struct Plane {
	float3 Point;
	float3 Normal;
};

} // End of namespace Scene
