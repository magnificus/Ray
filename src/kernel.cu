#include "cuda_runtime.h"
#include "vector_functions.h"
#include "common_functions.h"
#include "math_functions.h"
#include <stdlib.h>
#include <stdio.h>

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

__device__ bool intersectsSphere(const float3 &origin, const float3& dir,  const sphereInfo& info, float &t) {
		// geometric solution

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
	float sizeFarPlane = 1.0;
	float distBetweenPlanes = 2.0;


	float3 center = make_float3(imgw / 2.0, imgh / 2.0, 0.);
	float3 distFromCenter = make_float3(x - center.x, y - center.y, 0) *make_float3(1.0f / imgw, 1.0f / imgh, 1.);
	float3 firstPlanePos = sizeNearPlane*distFromCenter;

	float3 secondPlanePos = sizeFarPlane * distFromCenter;
	secondPlanePos.z = -distBetweenPlanes;

	float3 dirVector = normalize(secondPlanePos);
	int out = 0;

	sphereInfo s1;

	//currTime = 1000;

	//int a = 0;
	//for (int i = 0; i < 1000; i++) {
	//	a++;
	//}

	s1.pos = make_float3(sin(currTime)*2.0, -3,cos(currTime)*2 -30);
	s1.rad = 1;
	s1.rad2 = 1;

	float t;

	if (intersectsSphere(make_float3(0,0,0), dirVector, s1, t)) {
		out = 255;
	}
	else {
		out = 0;
	}

	//out = length(distFromCenter)*255;

	//int distToMid = (powf(x - (imgw / 2), 2) + powf(y - (imgh / 2), 2));

	//if (distToMid < powf((imgw / 4), 2)) {
	//	out = 255;
	//}
	//uchar4 c4 = make_uchar4((x & 0x20) ? 100 : 0, 0, (y & 0x20) ? 100 : 0, 0);
	g_odata[y * imgw + x] = /*out << 16 | out << 8 | out;*/rgbToInt(out, out, out);
}

extern "C" void
launch_cudaRender(dim3 grid, dim3 block, int sbytes, unsigned int *g_odata, int imgw, int imgh, float currTime)
{
	cudaRender << < grid, block, sbytes >> >(g_odata, imgw, imgh, currTime);
}
