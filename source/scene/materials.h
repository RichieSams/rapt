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
	MATERIAL_TYPE_SPECULAR
};

struct LambertMaterial {
	MaterialType MaterialType;
	float3 MainColor;
	float3 EmmisiveColor;
};

} // End of namespace Scene
