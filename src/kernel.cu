#include "cuda_runtime.h"
#include "vector_functions.h"
#include "common_functions.h"
#include "math_functions.h"
#include <stdlib.h>
#include <stdio.h>
#include "rayHelpers.cu"

cudaError_t cuda();

__global__ void kernel(){
  
}

// clamp x to range [a, b]
__device__ float clamp(float x, float a, float b)
{
	return max(a, min(b, x));
}

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b)
{
	r = clamp(r, 0.0f, 255.0f);
	g = clamp(g, 0.0f, 255.0f);
	b = clamp(b, 0.0f, 255.0f);
	return (int(b) << 16) | (int(g) << 8) | int(r);
}

__device__ float3 operator+(const float3& a, const float3& b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator*(const float3& a, const float3& b) {
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}


__device__ float3 operator*(const float& a, const float3& b) {
	return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ float3 operator-(const float3& a, const float3& b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__  float dot(float3 v1, float3 v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
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
	float invLen = 1.0f / length(v);// sqrtf(dot(v, v));
	return invLen * v;
}


struct sphereInfo {
	float3 pos;
	float rad;
	float rad2;
};

__device__ sphereInfo make_sphereInfo(float3 pos, float rad) {
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

__device__ planeInfo make_planeInfo(float3 point, float3 normal) {
	planeInfo p;
	p.point = point;
	p.normal = normal;
	return p;
}

enum shape{sphere, plane};

struct objectInfo {
	shape s;
	void* shapeData;
	float reflectivity;
	float refractivity;
	float refractiveIndex;
	float3 color;
};

__device__ objectInfo make_objectInfo(shape s, void* shapeData, float reflectivity, float3 color, float refractivity, float refractiveIndex) {
	objectInfo o;
	o.s = s;
	o.shapeData = shapeData;
	o.reflectivity = reflectivity;
	o.color = color;
	o.refractivity = refractivity;
	o.refractiveIndex = refractiveIndex;
	return o;
}


__device__ bool intersectsSphere(const float3 &origin, const float3& dir,  const sphereInfo& info, float &t) {

		float t0, t1; // solutions for t if the ray intersects 

		float3 L = info.pos - origin;
		float tca = dot(dir, L);
		// if (tca < 0) return false;
		float d2 = dot(L,L) - tca * tca;
		if (d2 > info.rad2) return false;
		float thc = sqrt(info.rad2 - d2);
		t0 = tca - thc;
		t1 = tca + thc;

		if (t0 > t1) {
			float temp = t0;
			t0 = t1;
			t1 = temp;
		}

		if (t0 < 0) {
			t0 = t1; // if t0 is negative, let's use t1 instead 
			if (t0 < 0) return false; // both t0 and t1 are negative 
		}
		t = t0;
		return true;
}

// plane normal, plane point, ray start, ray dir, point along line
__device__ bool intersectPlane(const planeInfo& p, const float3& l0, const float3& l, float& t)
{
	// assuming vectors are all normalized
	float denom = dot(p.normal, l);
	if (denom > 1e-6) {
		float3 p0l0 = p.point - l0;
		t = dot(p0l0, p.normal) / denom;
		return (t >= 0);
	}
	return false;
}


__device__ void fresnel(const float3& I, const float3& N, const float& ior, float& kr)
{
	float cosi = clamp(-1, 1, dot(I, N));
	float etai = 1, etat = ior;
	if (cosi > 0) { float temp = etai; etai = etat; etat = temp;}
	// Compute sini using Snell's law
	float sint = etai / etat * sqrtf(max(0.f, 1 - cosi * cosi));
	// Total internal reflection
	if (sint >= 1) {
		kr = 1;
	}
	else {
		float cost = sqrtf(max(0.f, 1 - sint * sint));
		cosi = abs(cosi);
		float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
		float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
		kr = (Rs * Rs + Rp * Rp) / 2;
	}
	// As a consequence of the conservation of energy, transmittance is given by:
	// kt = 1 - kr;
}


__device__ float3 refract(const float3& I, const float3& N, const float& ior)
{
	float cosi = clamp(-1, 1, dot(I, N));
	float etai = 1, etat = ior;
	float3 n = N;
	if (cosi < 0) { cosi = -cosi; }
	else { float temp = etai; etai = etat; etat = temp; n = inverse(N); }
	float eta = etai / etat;
	float k = 1 - eta * eta * (1 - cosi * cosi);
	return k < 0 ? make_float3(0,0,0) : eta * I + (eta * cosi - sqrtf(k)) * n;
}

__device__ float3 trace(float3 currRayPos, float3 currRayDir, int remainingDepth, const float &currTime, const objectInfo objects[], int numObjects) {
	if (remainingDepth <= 0) {
		return make_float3(0,0,0);
	}

	float closestDist = 1000000;
	float3 nextDir;
	float3 nextPos;
	float3 normal;
	int closestObjectIndex = -1;

	//for (int j = 0; j < 100; j++) {
		for (int i = 0; i < numObjects; i++) {
			objectInfo curr = objects[i];
			float currDist;

			switch (curr.s) {
			case plane: {
				planeInfo* p1 = (planeInfo*)curr.shapeData;
				if (intersectPlane(*p1, currRayPos, currRayDir, currDist) && currDist < closestDist) {
					closestDist = currDist;
					closestObjectIndex = i;

					float3 diffAngle = inverse(p1->normal) - inverse(currRayDir);
					nextDir = normalize(inverse(p1->normal) + diffAngle);
					normal = p1->normal;
					nextPos = currRayPos + currDist * currRayDir + 0.001 * nextDir;
				}

				break;
			}
			case sphere: {
				sphereInfo* s1 = (sphereInfo*)curr.shapeData;
				if (intersectsSphere(currRayPos, currRayDir, *s1, currDist) && currDist < closestDist) {
					closestDist = currDist;
					closestObjectIndex = i;

					nextPos = currRayPos + currDist * currRayDir;
					normal = normalize(nextPos - s1->pos);
					float3 diffAngle = normal - inverse(currRayDir);
					nextDir = normalize(normal + diffAngle);
					nextPos = nextPos + 0.001 * normal;

				}
				break;
			}
			}
		}
	//}

	if (closestObjectIndex == -1) {
		return make_float3(0,0,0);
	}
	else {
		objectInfo currObject = objects[closestObjectIndex];
		float3 reflected = make_float3(0, 0, 0);
		float3 refracted = make_float3(0, 0, 0);
		if (currObject.refractivity > 0.) {
			float3 refractedDir = refract(currRayDir, normal, currObject.refractiveIndex);
			refracted = trace(nextPos+0.002*refractedDir, refractedDir, remainingDepth - 1, currTime, objects, numObjects);

		}
		if (currObject.reflectivity > 0.) {
			reflected = currObject.reflectivity * trace(nextPos, nextDir, remainingDepth - 1, currTime, objects, numObjects);
		}
		float3 color = (1 - currObject.reflectivity - currObject.refractivity) * currObject.color;
		return 10 * (1 / powf(length(nextPos), 1)) * (color + reflected + refracted);
	}

}




__global__ void
cudaRender(unsigned int *g_odata, int imgw, int imgh, float currTime)
{

	extern __shared__ uchar4 sdata[];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;

	float sizeNearPlane = 1;
	float sizeFarPlane = 2.0;
	float distBetweenPlanes = 1.0;


	float3 center = make_float3(imgw / 2.0, imgh / 2.0, 0.);
	float3 distFromCenter = make_float3(x - center.x, y - center.y, 0) *make_float3(1.0f / imgw, 1.0f / imgh, 1.); // coordinate dist from center
	float3 firstPlanePos = sizeNearPlane*distFromCenter;

	float3 secondPlanePos = sizeFarPlane * distFromCenter;
	secondPlanePos.z = -distBetweenPlanes;

	firstPlanePos = make_float3(0, 0, 0);
	float3 dirVector = normalize(secondPlanePos - firstPlanePos);
	//int out = 0;


	sphereInfo s1 = make_sphereInfo(make_float3(sin(currTime) * 2.0, -3, cos(currTime) * 2 - 15), 1);
	sphereInfo s2 = make_sphereInfo(make_float3(-8, -4, -10), 4);
	sphereInfo s3 = make_sphereInfo(make_float3(5, 3, -20), 5);
	planeInfo p1 = make_planeInfo(make_float3(0, -4.0, 0), make_float3(0, -1, 0));
	planeInfo p2 = make_planeInfo(make_float3(0, 10.0, 0), make_float3(0, 1, 0));

	objectInfo objects[4];
	objects[0] = make_objectInfo(sphere, &s1, 0.0, make_float3(1, 0, 0),1.0,1.5);
	objects[1] = make_objectInfo(sphere, &s2, 0.1, make_float3(0, 1, 0),0.5,1.5);
	objects[2] = make_objectInfo(plane, &p1, 0.2, make_float3(0, 1, 1),0,0);
	objects[3] = make_objectInfo(sphere, &s3, 0.7, make_float3(1, 1, 1), 0,0);
	objects[4] = make_objectInfo(plane, &p2, 0.5, make_float3(1, 1, 1), 0,0);


	float3 out = 255*trace(firstPlanePos, dirVector, 6, currTime, objects, 5);



	g_odata[y * imgw + x] = rgbToInt(out.x, out.y, out.z);
}

extern "C" void
launch_cudaRender(dim3 grid, dim3 block, int sbytes, unsigned int *g_odata, int imgw, int imgh, float currTime)
{
	cudaRender << < grid, block, sbytes >> >(g_odata, imgw, imgh, currTime);
}
