/* rapt - RichieSam's Adventures in Path Tracing
 *
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#pragma once

#include <vector_types.h>


struct DeviceCamera {
	float3 ViewToWorldMatrixR0;
	float3 ViewToWorldMatrixR1;
	float3 ViewToWorldMatrixR2;

	float3 Origin;

	float TanFovXDiv2;
	float TanFovYDiv2;
};
