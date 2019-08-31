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

struct rayHitInfo {
	float reflectivity;
	float refractivity;
	float refractiveIndex;
	float insideColorDensity;
	float3 color;
};

struct objectInfo {
	shape s;
	shapeInfo shapeData;
	rayHitInfo rayInfo;

};

inline __device__ objectInfo make_objectInfo(shape s, shapeInfo shapeData, float reflectivity, float3 color, float refractivity, float refractiveIndex, float insideColorDensity) {
	objectInfo o;
	o.s = s;
	o.shapeData = shapeData;
	o.rayInfo.reflectivity = reflectivity;
	o.rayInfo.color = color;
	o.rayInfo.refractivity = refractivity;
	o.rayInfo.refractiveIndex = refractiveIndex;
	o.rayInfo.insideColorDensity = insideColorDensity;
	return o;
}

// total size will be pow(GRID_SIZE,3) bc of xyz
#define GRID_SIZE 15
#define GRID_SIZE2 GRID_SIZE*GRID_SIZE
#define GRID_DEPTH 1

#define GRID_POS(x,y,z) GRID_SIZE2*x + GRID_SIZE*y + z

struct triangleMesh {
	float3* vertices; 
	float3* normals; 
	unsigned int* indices; 
	int numIndices = 0;
	int numVertices = 0;

	rayHitInfo rayInfo;

	// acceleration structure
	float3 bbMin;
	float3 bbMax;
	unsigned int** grid; // lists with unsigned int marking which triangles intersect
	unsigned int* gridSizes;
	float3 gridBoxDimensions;
};

struct sceneInfo {

	// objects are pure mathematical objects, while meshes are triangle meshes
	const float currTime;
	objectInfo* objects;
	int numObjects;

	const triangleMesh* meshes;
	int numMeshes;
};


struct inputPointers {
	unsigned int* g_odata; // texture position

	sceneInfo scene;
	//objectInfo* objects;
	//int numObjects;

	//triangleMesh* meshes;
	//int numMeshes;

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


inline __device__ float3 operator+(const float3& a, const float3& b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator*(const float3& a, const float3& b) {
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __device__ float3 operator/(const float3& a, const float3& b) {
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
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
	return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

inline __device__ float length(float3 v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline __device__ float length1(float3 v)
{
	return v.x + v.y + v.z;
}

inline __device__ float3 inverse(float3 v)
{
	return make_float3(-v.x, -v.y, -v.z);
}

inline __device__ float3 normalize(float3 v)
{
	float invLen = 1 / sqrtf(dot(v, v));
	return invLen * v;
}

inline __device__ bool intersectBox(const float3& orig, const float3& dir, const float3& min, const float3 max, float &tmin, float &tmax)
{
	tmin = (min.x - orig.x) / dir.x;
	tmax = (max.x - orig.x) / dir.x;

	if (tmin > tmax) {
		float temp = tmin; tmin = tmax; tmax = temp;
	}

	float tymin = (min.y - orig.y) / dir.y;
	float tymax = (max.y - orig.y) / dir.y;

	if (tymin > tymax) {
		float temp = tymin; tymin = tymax; tymax = temp;
	}

	if ((tmin > tymax) || (tymin > tmax))
		return false;

	if (tymin > tmin)
		tmin = tymin;

	if (tymax < tmax)
		tmax = tymax;

	float tzmin = (min.z - orig.z) / dir.z;
	float tzmax = (max.z - orig.z) / dir.z;

	if (tzmin > tzmax) {
		float temp = tzmin; tzmin = tzmax; tzmax = temp;

	}

	if ((tmin > tzmax) || (tzmin > tmax))
		return false;

	if (tzmin > tmin)
		tmin = tzmin;

	if (tzmax < tmax)
		tmax = tzmax;

	return true;


}
