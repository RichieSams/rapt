/* The Halfling Project - A Graphics Engine and Projects
 *
 * The Halfling Project is the legal property of Adrian Astley
 * Copyright Adrian Astley 2013 - 2015
 */

/**
 * Modified for use in raic - RichieSam's Adventures in Cuda
 * raic is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "engine/camera.h"

#include <algorithm>
#include <DirectXMath.h>

#define PI 3.141592654f
#define TWO_PI 6.283185307f


namespace Engine {

void HostCamera::Rotate(float dTheta, float dPhi) {
	if (m_up > 0.0f) {
		m_theta += dTheta;
	} else {
		m_theta -= dTheta;
	}

	m_phi += dPhi;

	// Keep phi within -2PI to +2PI for easy 'up' comparison
	if (m_phi > TWO_PI) {
		m_phi -= TWO_PI;
	} else if (m_phi < -TWO_PI) {
		m_phi += TWO_PI;
	}

	// If phi is between 0 to PI or -PI to -2PI, make 'up' be positive Y, other wise make it negative Y
	if ((m_phi > 0 && m_phi < PI) || (m_phi < -PI && m_phi > -TWO_PI)) {
		m_up = 1.0f;
	} else {
		m_up = -1.0f;
	}
}

void HostCamera::Zoom(float distance) {
	m_radius -= distance;

	// Don't let the radius go negative
	// If it does, re-project our target down the look vector
	if (m_radius <= 0.0f) {
		m_radius = 30.0f;
		float3 look = normalize(m_target - GetCameraPosition());
		m_target += look * 30.0f;
	}
}

void HostCamera::Pan(float dx, float dy) {
	float3 look = normalize(m_target - GetCameraPosition());
	float3 worldUp = make_float3(0.0f, m_up, 0.0f);

	float3 right = normalize(cross(worldUp, look));
	float3 up = cross(look, right);

	m_target += (right * dx) + (up * dy);
}

DeviceCamera HostCamera::GetDeviceCamera() const {
	float3 worldUp = make_float3(0.0f, m_up, 0.0f);

	DeviceCamera camera;
	camera.origin = GetCameraPosition();
	
	camera.z = normalize(m_target - camera.origin);
	camera.x = normalize(cross(worldUp, camera.z));
	camera.y = cross(camera.z, camera.x);

	camera.tanFovDiv2_X = m_tanFovDiv2_X;
	camera.tanFovDiv2_Y = m_tanFovDiv2_Y;

	return camera;
}

} // End of namespace Scene
