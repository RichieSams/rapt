/* rapt - RichieSam's Adventures in Path Tracing
 *
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#pragma once

#include <vector_types.h>


namespace Scene {

enum MaterialType {
	MATERIAL_TYPE_DIFFUSE,
	MATERIAL_TYPE_SPECULAR,
	MATERIAL_TYPE_EMMISIVE
};

struct LambertMaterial {
	MaterialType Type;
	float3 Color;
};

} // End of namespace Scene
