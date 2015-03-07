/* The Halfling Project - A Graphics Engine and Projects
 *
 * The Halfling Project is the legal property of Adrian Astley
 * Copyright Adrian Astley 2013 - 2014
 */

/**
 * Modified for use in rapt - RichieSam's Adventures in Path Tracing
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#pragma once

#include "common/sys_headers.h"
#include "common/typedefs.h"


#include <d3d11.h>


namespace Engine {

class RAPTEngine {
public:
	RAPTEngine(HINSTANCE hinstance);
	~RAPTEngine();

private:
	LPCTSTR m_mainWndCaption;
	HINSTANCE m_hinstance;

	bool m_appPaused;
	bool m_isMinOrMaximized;
	bool m_resizing;

protected:
	HWND m_hwnd;
	bool m_fullscreen;
	uint32 m_clientWidth;
	uint32 m_clientHeight;

	ID3D11Device *m_device;
	ID3D11DeviceContext *m_immediateContext;
	IDXGISwapChain *m_swapChain;

	bool m_d3dInitialized;

	public:
	/**
	 * Initializes the Engine.
	 * This will create and register the window, and create and initialize D3D
	 *
	 * @param mainWndCaption    The caption for the window header bar
	 * @param screenWidth       The width of the client area of the window
	 * @param screenHeight      The height of the client area of the window
	 * @param fullscreen        Should the window be fullscreen or not?
	 * @return                  Returns true if initialization succeeded
	 */
	virtual bool Initialize(LPCTSTR mainWndCaption, uint32 screenWidth, uint32 screenHeight, bool fullscreen);
	/**
	 * Shuts down the Engine
	 * This will shut down D3D, release any COM devices, and then shutdown the window.
	 */
	virtual void Shutdown();
	/**
	 * The main window loop
	 * It will loop until it receives a WM_QUIT message
	 *
	 * The loop is as follows:
	 * 1. Process all windows messages
	 * 2. DrawFrame()
	 */
	void Run();

	/**
	 * The main window Message Handler.
	 * Override if you need to handle special messages. However, any messages
	 * that you don't handle should be passed to this base method.
	 *
	 * @param hwnd      The handle of the main window
	 * @param msg       The message code
	 * @param wParam    The wParam of the message
	 * @param lParam    The lParam of the message
	 * @return          A LRESULT code signaling whether a message was handled or not
	 */
	virtual LRESULT MsgProc(HWND hwnd, uint msg, WPARAM wParam, LPARAM lParam);

#ifdef _DEBUG
	inline void CreateDebugInterface(ID3D11Debug **debugInterface) {
		m_device->QueryInterface(__uuidof(ID3D11Debug), reinterpret_cast<void**>(debugInterface));
	}
#endif

protected:
	/**
	 * Called every time the window is resized
	 */
	virtual void OnResize() = 0;
	/**
	 * Called once every loop
	 *
	 * @param deltaTime    The deltaTime passed since the last loop
	 */
	virtual void DrawFrame() {}

	/**
	 * Called every time one of the mouse buttons is pressed down
	 *
	 * @param buttonState    The wParam describing the button state
	 * @param x              The x position of the cursor (In window coordinates)
	 * @param y              The y position of the cursor (In window coordinates)
	 */
	virtual void MouseDown(WPARAM buttonState, int x, int y) {
		UNREFERENCED_PARAMETER(buttonState);
		UNREFERENCED_PARAMETER(x);
		UNREFERENCED_PARAMETER(y);
	}
	/**
	 * Called every time one of the mouse buttons is let go
	 *
	 * @param buttonState    The wParam describing the button state
	 * @param x              The x position of the cursor (In window coordinates)
	 * @param y              The y position of the cursor (In window coordinates)
	 */
	virtual void MouseUp(WPARAM buttonState, int x, int y) {
		UNREFERENCED_PARAMETER(buttonState);
		UNREFERENCED_PARAMETER(x);
		UNREFERENCED_PARAMETER(y);
	}
	/**
	 * Called every time the mouse moves
	 *
	 * @param buttonState    The wParam describing the button state
	 * @param x              The x position of the cursor (In window coordinates)
	 * @param y              The y position of the cursor (In window coordinates)
	 */
	virtual void MouseMove(WPARAM buttonState, int x, int y) {
		UNREFERENCED_PARAMETER(buttonState);
		UNREFERENCED_PARAMETER(x);
		UNREFERENCED_PARAMETER(y);
	}
	/**
	 * Called every time the middle mouse wheel is scrolled
	 *
	 * @param zDelta    The number of units scrolled. The number of units per full circle of the wheel depends on the mouse vendor
	 */
	virtual void MouseWheel(int zDelta) {
		UNREFERENCED_PARAMETER(zDelta);
	}

	virtual void CharacterInput(wchar character) {
		UNREFERENCED_PARAMETER(character);
	}

private:
	/** Creates the window and registers it */
	void InitializeWindow();
	/** Un-registers the window and destroys it*/
	void ShutdownWindow();
};

} // End of namespace Engine

// This is used to forward Windows messages from a global window
// procedure to our member function window procedure because we cannot
// assign a member function to WNDCLASS::lpfnWndProc.
static Engine::RAPTEngine *g_engine = NULL;
