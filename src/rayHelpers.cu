#pragma once
#include "cuda_runtime.h"
#include "vector_functions.h"
#include "common_functions.h"
#include "math_functions.h"
#include "sharedStructs.h"
#include <stdlib.h>
#include <stdio.h>

// clamp x to range [a, b]
inline __device__ float clamp(float x, float a, float b)
{
	return max(a, min(b, x));
}

// convert floating point rgb color to 8-bit integer
inline __device__ int rgbToInt(float3 rgb)
{
	rgb.x = clamp(rgb.x, 0.0f, 65535.f);
	rgb.y = clamp(rgb.y, 0.0f, 65535.f);
	rgb.z = clamp(rgb.z, 0.0f, 65535.f);
	return (int(rgb.z) << 16) | (int(rgb.y) << 8) | int(rgb.x);
}

// the reverse
inline __device__ float3 intToRgb(int val)
{
	float r =  val % 256;
	float g = (val % (256*256)) / 256;
	float b = val / (256 * 256); 
	return make_float3(r, g, b);
}


inline __device__ float3 rotateAngleAxis(const float3 vector, const float angleDeg, const float3& axis) 
{
	double S, C;
	sincos(angleDeg, &S, &C);

	const float XX = axis.x * axis.x;
	const float YY = axis.y * axis.y;
	const float ZZ = axis.z * axis.z;

	const float XY = axis.x * axis.y;
	const float YZ = axis.y * axis.z;
	const float ZX = axis.z * axis.x;

	const float XS = axis.x * S;
	const float YS = axis.y * S;
	const float ZS = axis.z * S;

	const float OMC = 1.f - C;

	return make_float3(
		(OMC * XX + C) * vector.x + (OMC * XY - ZS) * vector.y + (OMC * ZX + YS) * vector.z,
		(OMC * XY + ZS) * vector.x + (OMC * YY + C) * vector.y + (OMC * YZ - XS) * vector.z,
		(OMC * ZX - YS) * vector.x + (OMC * YZ + XS) * vector.y + (OMC * ZZ + C) *vector.z 
	);
}


inline __device__ float rand(float2 co) {

	float val= sinf(dot(make_float2(co.x, co.y), make_float2(12.9898, 78.233)) * 43758.5453);
	return val - floor(val);
}



#define sampleImageAt(pos) make_float3(texture.image[pos], texture.image[pos + 1], texture.image[pos + 2])
// 8 bit wrap around bilinear sampling
inline __device__ float3 sampleTexture(float2 pos, inputImage texture) {
	//pos = make_float2(max(0, pos.x), max(0, pos.y));
	//float2 Position = make_float2((unsigned int)floor(pos.x) % texture.width, (unsigned int)floor(pos.y) % texture.height);

	int samplePixelPosX = (int) floor(pos.x * texture.width) % texture.width;
	int samplePixelPosY = (int) floor(pos.y * texture.height) % texture.height;
	float2 Position = make_float2(samplePixelPosX, samplePixelPosY);
	int LLPos = ((texture.width * Position.y) + Position.x) * 3;
	int LRPos = ((texture.width * Position.y) + (int) (Position.x + 1) % texture.width) * 3;
	int ULPos = ((texture.width * ((int) (Position.y + 1) % texture.height)) + Position.x) * 3;
	int URPos = ((texture.width * ((int) (Position.y + 1) % texture.height)) + (int) (Position.x+1) % texture.width) * 3;


	// bilinear sampling
	float3 LL = sampleImageAt(LLPos);
	float3 LR = sampleImageAt(LRPos);
	float3 UL = sampleImageAt(ULPos);
	float3 UR = sampleImageAt(URPos);

	float remX = pos.x * texture.width - floor(pos.x * texture.width);
	float remY = pos.y * texture.height - floor(pos.y * texture.height);

	float3 Lower = remX * LR + (1. - remX) * LL;
	float3 Upper = remX * UR + (1. - remX) * UL;
	float3 Combined = remY * Upper + (1. - remY) * Lower;

	return Combined;

}
