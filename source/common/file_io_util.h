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

#include <fstream>


namespace Common {

char *ReadWholeFile(const wchar *name, DWORD *bytesRead);

/**
 * Reads a line from an istream using /r, /r/n, and /n as line delimiters
 *
 * @param inputStream    The stream to read from
 * @param output         Will be filled with the line read
 * @return               The input stream
 */
std::istream &SafeGetLine(std::istream &inputStream, std::string &output);

std::ostream &BinaryWriteUInt64(std::ostream &stream, const uint64 &value);
std::ostream &BinaryWriteInt64(std::ostream &stream, const int64 &value);

std::ostream &BinaryWriteUInt32(std::ostream &stream, const uint32 &value);
std::ostream &BinaryWriteInt32(std::ostream &stream, const int32 &value);

std::ostream &BinaryWriteUInt16(std::ostream &stream, const uint16 &value);
std::ostream &BinaryWriteInt16(std::ostream &stream, const int16 &value);

std::ostream &BinaryWriteByte(std::ostream &stream, const byte &value);

} // End of namespace Common
