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
inline __device__ int rgbToInt(float r, float g, float b)
{
	r = clamp(r, 0.0f, 255.0f);
	g = clamp(g, 0.0f, 255.0f);
	b = clamp(b, 0.0f, 255.0f);
	return (int(b) << 16) | (int(g) << 8) | int(r);
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

