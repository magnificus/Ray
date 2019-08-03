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

inline __device__ float3 normalize(float3 v)
{
	float invLen = sqrtf(dot(v, v));
	return invLen * v;
}




__global__ void
cudaRender(unsigned int *g_odata, int imgw, int imgh)
{

	float3 sphereLoc = make_float3(0, 2, -10);
	float sphereSize = 2;

	extern __shared__ uchar4 sdata[];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;

	float sizeNearPlane = 1;
	float sizeFarPlane = 5;
	float distBetweenPlanes = 5;


	float3 center = make_float3(imgw / 2.0, imgh / 2.0, 0.);
	float3 distFromCenter = make_float3(x - center.x, y - center.y, 0);

	float3 firstPlanePos = sizeNearPlane*distFromCenter;

	float3 secondPlanePos = sizeFarPlane * distFromCenter;
	secondPlanePos.z = -distBetweenPlanes;

	float3 dirVector = normalize(secondPlanePos - firstPlanePos);


	int out = 0;

	int distToMid = (powf(x - (imgw / 2), 2) + powf(y - (imgh / 2), 2));

	if (distToMid < powf((imgw / 4), 2)) {
		out = 255;
	}
	//uchar4 c4 = make_uchar4((x & 0x20) ? 100 : 0, 0, (y & 0x20) ? 100 : 0, 0);
	g_odata[y * imgw + x] = out << 16 | out << 8 | out;// rgbToInt(out, out, out);
}

extern "C" void
launch_cudaRender(dim3 grid, dim3 block, int sbytes, unsigned int *g_odata, int imgw, int imgh)
{
	cudaRender << < grid, block, sbytes >> >(g_odata, imgw, imgh);
}
