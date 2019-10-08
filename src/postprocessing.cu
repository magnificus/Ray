#include "rayHelpers.cu"


__global__ void
cudaBloomSample(PostProcessPointers pointers, int imgw, int imgh)
{
	extern __shared__ uchar4 sdata[];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x * bw + tx;
	int y = blockIdx.y * bh + ty;

	float bloomThreshold = 255;
	float bloomStrength = 0.01;

	int firstPos = (y * imgw + x) * 4;
	float3 CurrC = make_float3(pointers.inputImage[firstPos], pointers.inputImage[firstPos + 1], pointers.inputImage[firstPos + 2]);
	float luma = (CurrC.x + CurrC.y + CurrC.z) / 3;
	CurrC = max(0., luma - bloomThreshold)*bloomStrength * CurrC;


	pointers.processWrite[firstPos] =  CurrC.x;
	pointers.processWrite[firstPos + 1] = CurrC.y;
	pointers.processWrite[firstPos + 2] = CurrC.z;
}

__global__ void
cudaBlur(PostProcessPointers pointers, int imgw, int imgh)
{
	extern __shared__ uchar4 sdata[];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x * bw + tx;
	int y = blockIdx.y * bh + ty;

	float weights[] =
	{
	  0.01, 0.02, 0.04, 0.02, 0.01,
	  0.02, 0.04, 0.08, 0.04, 0.02,
	  0.04, 0.08, 0.16, 0.08, 0.04,
	  0.02, 0.04, 0.08, 0.04, 0.02,
	  0.01, 0.02, 0.04, 0.02, 0.01
	};


	int firstPos = (y * imgw + x) * 4;
	float3 next = make_float3(0,0,0);// make_float3(pointers.processRead[firstPos], pointers.processRead[firstPos + 1], pointers.processRead[firstPos + 2]);
	for (int xD = -2; xD <= 2; xD++) {
		for (int yD = -2; yD <= 2; yD++) {

			float factor = weights[(yD + 2) * 5 + xD + 2];
			int currPos = firstPos + 4 * (yD * imgw + xD);
			float3 CurrC = make_float3(pointers.processRead[currPos], pointers.processRead[currPos + 1], pointers.processRead[currPos + 2]);
			next = next + CurrC;

		}
	}

	pointers.processWrite[firstPos] = next.x;
	pointers.processWrite[firstPos + 1] = next.y;
	pointers.processWrite[firstPos + 2] = next.z;
}

__global__ void
cudaBloomOutput(PostProcessPointers pointers, int imgw, int imgh)
{
	extern __shared__ uchar4 sdata[];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x * bw + tx;
	int y = blockIdx.y * bh + ty;


	int firstPos = (y * imgw + x) * 4;

	pointers.finalOut[firstPos] = pointers.processRead[firstPos] + pointers.inputImage[firstPos];
	pointers.finalOut[firstPos + 1] = pointers.processRead[firstPos+1] + pointers.inputImage[firstPos+1];
	pointers.finalOut[firstPos + 2] = pointers.processRead[firstPos+2] + pointers.inputImage[firstPos+2];
}






extern "C" void
launch_cudaBloomSample(dim3 grid, dim3 block, int sbytes, int imgw, int imgh, PostProcessPointers pointers)
{

	cudaBloomSample << < grid, block, sbytes >> > (pointers, imgw, imgh);
}

extern "C" void
launch_cudaBloomOutput(dim3 grid, dim3 block, int sbytes, int imgw, int imgh, PostProcessPointers pointers)
{

	cudaBloomOutput << < grid, block, sbytes >> > (pointers, imgw, imgh);
}


extern "C" void
launch_cudaBlur(dim3 grid, dim3 block, int sbytes, int imgw, int imgh, PostProcessPointers pointers)
{

	cudaBlur << < grid, block, sbytes >> > (pointers, imgw, imgh);
}
