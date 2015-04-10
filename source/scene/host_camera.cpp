/* The Halfling Project - A Graphics Engine and Projects
 *
 * The Halfling Project is the legal property of Adrian Astley
 * Copyright Adrian Astley 2013 - 2015
 */

/**
 * Modified for use in rapt - RichieSam's Adventures in Path Tracing
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "scene/host_camera.h"

#include "graphics/cuda_util.h"

#include "scene/device_camera.h"

#include <algorithm>

#define PI 3.141592654f
#define TWO_PI 6.283185307f


namespace Scene {

void HostCamera::SetCamera(float theta, float phi, float radius, float3 target) {
	m_theta = theta;
	m_phi = phi;
	m_radius = radius;
	m_target = target;

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

void HostCamera::UpdateDeviceCameraWithInternalState(DeviceCamera *deviceCamera) const {
	float3 worldUp = make_float3(0.0f, m_up, 0.0f);

	float3 origin = GetCameraPosition();
	
	float3 zAxis = normalize(m_target - origin);
	float3 xAxis = normalize(cross(worldUp,zAxis));
	float3 yAxis = cross(zAxis, xAxis);

	DeviceCamera camera;
	camera.ViewToWorldMatrixR0 = make_float3(xAxis.x, yAxis.x, zAxis.x);
	camera.ViewToWorldMatrixR1 = make_float3(xAxis.y, yAxis.y, zAxis.y);
	camera.ViewToWorldMatrixR2 = make_float3(xAxis.z, yAxis.z, zAxis.z);

	camera.Origin = origin;

	camera.TanFovXDiv2 = m_tanFovXDiv2;
	camera.TanFovYDiv2 = m_tanFovYDiv2;

	CE(cudaMemcpy(deviceCamera, &camera, sizeof(DeviceCamera), cudaMemcpyHostToDevice));
}

} // End of namespace Scene
