#include "rayHelpers.cu"


__global__ void
cudaBloom(inputPointers pointers, int imgw, int imgh, float currTime, inputStruct input)
{
	extern __shared__ uchar4 sdata[];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x * bw + tx;
	int y = blockIdx.y * bh + ty;

	float bloomThreshold = 255;
	float bloomStrength = 0.02;


	float weights[] =
	{
	  0.01, 0.02, 0.04, 0.02, 0.01,
	  0.02, 0.04, 0.08, 0.04, 0.02,
	  0.04, 0.08, 0.16, 0.08, 0.04,
	  0.02, 0.04, 0.08, 0.04, 0.02,
	  0.01, 0.02, 0.04, 0.02, 0.01
	};


	int firstPos = (y * imgw + x) * 4;
	float3 PrevC = make_float3(pointers.image1[firstPos], pointers.image1[firstPos+1], pointers.image1[firstPos+2]);
	float extraLight = 0;
	for (int xD = -2; xD <= 2; xD++) {
		for (int yD = -2; yD <= 2; yD++) {

			float factor = weights[(yD + 2) * 5 + xD + 2];
			int currPos = firstPos + 4 * (yD * imgw + xD);
			float3 CurrC = make_float3(pointers.image1[currPos], pointers.image1[currPos + 1], pointers.image1[currPos + 2]);
			float luma = (CurrC.x + CurrC.y + CurrC.z) / 3;

			PrevC = PrevC + max(0., luma - bloomThreshold)*bloomStrength* factor * CurrC;

			//float luma = (value.x + value.y + value.z) / 3;


		}
	}


	//float3 prev = intToRgb((int)pointers.image1[y * imgw + x]);
	//prev.x ?
	pointers.image2[firstPos] = PrevC.x;
	pointers.image2[firstPos + 1] = PrevC.y;
	pointers.image2[firstPos + 2] = PrevC.z;
}




extern "C" void
launch_cudaBloom(dim3 grid, dim3 block, int sbytes, inputPointers pointers, int imgw, int imgh, float currTime, inputStruct input)
{

	cudaBloom << < grid, block, sbytes >> > (pointers, imgw, imgh, currTime, input);
}
