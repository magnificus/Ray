#include "cuda_runtime.h"
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

__device__ int clamp(int x, int a, int b)
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


__global__ void
cudaRender(unsigned int *g_odata, int imgw, int imgh)
{
	extern __shared__ uchar4 sdata[];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;

	float3 vec;
	vec.x = x;
	vec.y = y;
	vec.z = -1.0;

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
