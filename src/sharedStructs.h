#pragma once

#include "cuda_runtime.h"


struct sphereInfo {
	float3 pos;
	float rad;
	float rad2;
};

struct shapeInfo {
	float3 pos;
	float3 normal;
	float rad;
};

inline __device__ shapeInfo make_shapeInfo(float3 pos, float3 normal, float rad) {
	shapeInfo info;
	info.pos = pos;
	info.normal = normal;
	info.rad = rad;
	return info;
}

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
	shapeInfo shapeData;
	float reflectivity;
	float refractivity;
	float refractiveIndex;
	float3 color;
};

inline __device__ objectInfo make_objectInfo(shape s, shapeInfo shapeData, float reflectivity, float3 color, float refractivity, float refractiveIndex) {
	objectInfo o;
	o.s = s;
	o.shapeData = shapeData;
	o.reflectivity = reflectivity;
	o.color = color;
	o.refractivity = refractivity;
	o.refractiveIndex = refractiveIndex;
	return o;
}

struct inputPointers {
	unsigned int* g_odata;
	objectInfo* objects;
	int numObjects;
};


struct inputStruct {
	float currPosX;
	float currPosY;
	float currPosZ;

	float forwardX;
	float forwardY;
	float forwardZ;

	float upX;
	float upY;
	float upZ;
};
