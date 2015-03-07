/* rapt - RichieSam's Adventures in Path Tracing
 *
 * rapt is the legal property of Adrian Astley
 * Copyright Adrian Astley 2015
 */

#include "common/typedefs.h"


// Only include the base windows libraries
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>

// Un-define min and max from the windows headers
#ifdef min
	#undef min
#endif

#ifdef max
	#undef max
#endif
