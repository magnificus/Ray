#pragma once

#include "cuda_runtime.h"
#include "vector_functions.h"
#include "common_functions.h"
#include "math_functions.h"
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

inline __device__ float3 operator+(const float3& a, const float3& b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator*(const float3& a, const float3& b) {
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}


inline __device__ float3 operator*(const float& a, const float3& b) {
	return make_float3(a * b.x, a * b.y, a * b.z);
}

inline __device__ float3 operator-(const float3& a, const float3& b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}


inline __device__  float dot(float3 v1, float3 v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

inline __device__  float3 cross(float3 v1, float3 v2)
{
	return make_float3(v1.y * v2.z - v1.z * v2.y, v1.x * v2.z - v1.z * v2.x, v1.x * v2.y - v1.y * v2.x);
}

inline __device__ float length(float3 v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}


inline __device__ float3 inverse(float3 v)
{
	return make_float3(-v.x, -v.y, -v.z);
}

inline __device__ float3 normalize(float3 v)
{
	float invLen = /*1.0f / length(v);// */1 / sqrtf(dot(v, v));
	return invLen * v;
}

struct sphereInfo {
	float3 pos;
	float rad;
	float rad2;
};

inline __device__ sphereInfo make_sphereInfo(float3 pos, float rad) {
	sphereInfo s;
	s.pos = pos;
	s.rad = rad;
	s.rad2 = rad * rad;
	return s;
}


struct planeInfo {
	float3 point;
	float3 normal;
};

inline __device__ planeInfo make_planeInfo(float3 point, float3 normal) {
	planeInfo p;
	p.point = point;
	p.normal = normal;
	return p;
}

enum shape { sphere, plane };

struct objectInfo {
	shape s;
	void* shapeData;
	float reflectivity;
	float refractivity;
	float refractiveIndex;
	float3 color;
};

inline __device__ objectInfo make_objectInfo(shape s, void* shapeData, float reflectivity, float3 color, float refractivity, float refractiveIndex) {
	objectInfo o;
	o.s = s;
	o.shapeData = shapeData;
	o.reflectivity = reflectivity;
	o.color = color;
	o.refractivity = refractivity;
	o.refractiveIndex = refractiveIndex;
	return o;
}


inline __device__ float3 rotateAngleAxis(const float3 vector, const float angleDeg, const float3& axis) 
{
	double S, C;
	sincos(angleDeg, &S, &C);
	//FMath::SinCos(&S, &C, FMath::DegreesToRadians(angleDeg));

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



