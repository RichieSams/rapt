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

#pragma once

#include <graphics/helper_math.h>


struct DeviceCamera {
	float3 ViewToWorldMatrixR0;
	float3 ViewToWorldMatrixR1;
	float3 ViewToWorldMatrixR2;
	
	float3 Origin;

	float TanFovDiv2_X;
	float TanFovDiv2_Y;
};

namespace Engine {

/** 
 * A class for emulating a camera looking into the scene. 
 * It stores the view matrix and projection matrix for use by the renderer
 * and provides methods for moving the camera around the scene
 */
class HostCamera {
public:
	HostCamera() 
		: m_theta(0.0f), 
		  m_phi(0.0f), 
		  m_radius(0.0f), 
		  m_up(1.0f),
		  m_tanFovDiv2_X(tan(0.785398f /* PI / 4 */)),
		  m_tanFovDiv2_Y(tan(0.785398f * 19.0f / 10.0f)),
		  m_target(make_float3(0.0f, 0.0f, 0.0f)) {
	}
	HostCamera(float theta, float phi, float radius) 
		: m_theta(theta), 
		  m_phi(phi), 
		  m_radius(radius), 
		  m_up(1.0f),
		  m_tanFovDiv2_X(tan(0.785398f /* PI / 4 */)),
		  m_tanFovDiv2_Y(tan(0.785398f * 19.0f / 10.0f)),
		  m_target(make_float3(0.0f, 0.0f, 0.0f)) {
	}

private:
	float m_theta;
	float m_phi;
	float m_radius;
	float m_up;

	float m_tanFovDiv2_X;
	float m_tanFovDiv2_Y;

	float3 m_target;

public:
	/**
	 * Rotate the camera about a point in front of it (m_target). Theta is a rotation 
	 * that tilts the camera forward and backward. Phi tilts the camera side to side. 
	 *
	 * @param dTheta    The number of radians to rotate in the theta direction
	 * @param dPhi      The number of radians to rotate in the phi direction
	 */
	void Rotate(float dTheta, float dPhi);
	/**
	 * Move the camera down the look vector, closer to m_target. If we overtake m_target,
	 * it is reprojected 30 units down the look vector
	 *
	 * TODO: Find a way to *not* hard-code the reprojection distance. Perhaps base it on the 
	 *       scene size? Or maybe have it defined in an settings.ini file
	 *
	 * @param distance    The distance to zoom. Negative distance will move the camera away from the target, positive will move towards
	 */
	void Zoom(float distance);
	/**
	 * Moves the camera within its local X-Y plane
	 *
	 * @param dx    The amount to move the camera right or left
	 * @param dy    The amount to move the camera up or down
	 */
	void Pan(float dx, float dy);

	inline void SetProjection(float fov, float aspectRatio) {
		m_tanFovDiv2_X = tan(fov * 0.5f);
		m_tanFovDiv2_Y = tan(fov * 0.5f) / aspectRatio;
	}

	/**
	 * Returns the position of the camera in Cartesian coordinates
	 *
	 * @return    The position of the camera
	 */
	inline float3 GetCameraPosition() const { 
		float x = m_radius * sinf(m_phi) * sinf(m_theta);
		float y = m_radius * cosf(m_phi);
		float z = m_radius * sinf(m_phi) * cosf(m_theta);

		return m_target + make_float3(x, y, z);
	}

	void SetDeviceCamera(DeviceCamera *deviceCamera) const;
};

} // End of namespace Scene
