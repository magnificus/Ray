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
	float bloomStrength = 0.0001;

	int firstPos = (y * imgw + x) * 4;
	float3 CurrC = make_float3(pointers.inputImage[firstPos], pointers.inputImage[firstPos + 1], pointers.inputImage[firstPos + 2]);
	float luma = (CurrC.x + CurrC.y + CurrC.z) / 3;
	CurrC = max(0., powf(luma - bloomThreshold, 1))*bloomStrength * CurrC;


	pointers.processWrite[firstPos] =  CurrC.x;
	pointers.processWrite[firstPos + 1] = CurrC.y;
	pointers.processWrite[firstPos + 2] = CurrC.z;
}

__global__ void
cudaBlur(PostProcessPointers pointers, int imgw, int imgh, int currRatio)
{
	extern __shared__ uchar4 sdata[];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int bt = blockDim.z;
	int x = blockIdx.x * bw + tx;
	int y = blockIdx.y * bh + ty;
	int z = blockIdx.z + bt * tz;

	int firstPos = (z * (imgw * imgw) + y * imgw + x) * 4;

	float weights[] =
	{
	  0.01, 0.02, 0.04, 0.02, 0.01,
	  0.02, 0.04, 0.08, 0.04, 0.02,
	  0.04, 0.08, 0.16, 0.08, 0.04,
	  0.02, 0.04, 0.08, 0.04, 0.02,
	  0.01, 0.02, 0.04, 0.02, 0.01
	};

	float3 next = make_float3(0,0,0);
	for (int xD = -2; xD <= 2; xD++) {
		for (int yD = -2; yD <= 2; yD++) {

			float factor = weights[(yD + 2) *5 + xD + 2];
			int xToUse = max(0, min(x + xD,imgw-1));
			int YToUse = max(0, min(y + yD,imgh-1));
			int currPos = 4 * (YToUse * imgw * currRatio + xToUse + (z*imgw*imgw));
			float3 CurrC = make_float3(pointers.processRead[currPos], pointers.processRead[currPos + 1], pointers.processRead[currPos + 2]);
			next = next + factor*CurrC;

		}
	}

	pointers.processWrite[firstPos] =  next.x;
	pointers.processWrite[firstPos + 1] = next.y;
	pointers.processWrite[firstPos + 2] = next.z;
}

#define BLOOM_KERNEL_SIZE 5

__global__ void
cudaBlur2(PostProcessPointers pointers, int imgw, int imgh, bool isHorizontal, int currRatio)
{
	extern __shared__ uchar4 sdata[];


	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int bt = blockDim.z;
	int x = blockIdx.x * bw + tx;
	int y = blockIdx.y * bh + ty;
	int z = blockIdx.z + bt * tz;

	int firstPos = (z * (imgw * imgh) + y * imgw + x) * 4;
	float weight[BLOOM_KERNEL_SIZE] = {0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216}; // 5
	//float weight[BLOOM_KERNEL_SIZE] = { 0.104379, 	0.075216, 0.02812, 	0.005441, 0.000543 , 0.000028 , 0.000001 }; // 7


	float3 result = make_float3(pointers.processRead[firstPos], pointers.processRead[firstPos + 1], pointers.processRead[firstPos + 2]) * weight[0]; 
	if (isHorizontal) {
		for (int i = 1; i < BLOOM_KERNEL_SIZE; ++i)
		{
			int xU = min(i, imgw - 1 - x)*4;
			int xL = min(i, x)*4;
			result = result + make_float3(pointers.processRead[firstPos + xU], pointers.processRead[firstPos + xU + 1], pointers.processRead[firstPos + xU + 2])*weight[i];
			result = result + make_float3(pointers.processRead[firstPos - xL], pointers.processRead[firstPos - xL + 1], pointers.processRead[firstPos - xL + 2])*weight[i];
		}
	}
	else {
		for (int i = 1; i < BLOOM_KERNEL_SIZE; ++i)
		{
			int yU = min(i, imgh - 1 - y) * 4;
			int yL = min(i, y) * 4;
			result = result + make_float3(pointers.processRead[firstPos + yU * imgw *currRatio], pointers.processRead[firstPos + yU * imgw * currRatio  + 1], pointers.processRead[firstPos + yU * imgw * currRatio  + 2]) * weight[i];
			result = result + make_float3(pointers.processRead[firstPos - yL * imgw * currRatio], pointers.processRead[firstPos - yL * imgw * currRatio  + 1], pointers.processRead[firstPos - yL * imgw * currRatio + 2]) * weight[i];
		}
	}

	pointers.processWrite[firstPos] = result.x;
	pointers.processWrite[firstPos + 1] = result.y;
	pointers.processWrite[firstPos + 2] = result.z;
}




__global__ void
cudaDownSampleToHalfRes(PostProcessPointers pointers, int imgw, int imgh, int currRatio)
{
	extern __shared__ uchar4 sdata[];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x * bw + tx;
	int y = blockIdx.y * bh + ty;

	int myPos = (y * imgw * currRatio + x)*4;
	int prevPos = (y * 2 * (imgw * currRatio) + x*2) * 4;

	float3 total = make_float3(0,0,0);
	for (int xD = 0; xD < 2; xD++) {
		for (int yD = 0; yD < 2; yD++) {
			int nextPos = prevPos +(yD * (imgw * currRatio) + xD) * 4;
			total.x += pointers.processRead[nextPos];
			total.y += pointers.processRead[nextPos+1];
			total.z += pointers.processRead[nextPos+2];
		}
	}
	total = total * 0.25;

	pointers.processWrite[myPos] = total.x;
	pointers.processWrite[myPos + 1] = total.y;
	pointers.processWrite[myPos + 2] = total.z;
}

__global__ void
cudaUpSampleToDoubleRes(PostProcessPointers pointers, int imgw, int imgh, int currRatio)
{
	extern __shared__ uchar4 sdata[];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x * bw + tx;
	int y = blockIdx.y * bh + ty;

	int myPos = ((y +1)* imgw * currRatio + x + 1) * 4;

	int nextX = max(0, min((x / 2) + 1, imgw - 1));
	int nextY = max(0, min((y / 2) + 1, imgh - 1));

	int prevPosUL = ((y / 2) * imgw * currRatio + (x /2)) * 4;
	int prevPosUR = prevPosUL + 1*4;//((y / 2) * imgw * currRatio + nextX) * 4;
	int prevPosLL = prevPosUL + (imgw * currRatio)*4;//(nextY * imgw * currRatio + (x / 2)) * 4;
	int prevPosLR = prevPosUL + (imgw * currRatio+ 1) * 4;//(nextY * imgw * currRatio + nextX) * 4;

	float3 UL = make_float3(pointers.processRead[prevPosUL], pointers.processRead[prevPosUL +1], pointers.processRead[prevPosUL +2]);
	float3 UR = make_float3(pointers.processRead[prevPosUR], pointers.processRead[prevPosUR +1], pointers.processRead[prevPosUR +2]);
	float3 LL = make_float3(pointers.processRead[prevPosLL], pointers.processRead[prevPosLL +1], pointers.processRead[prevPosLL +2]);
	float3 LR = make_float3(pointers.processRead[prevPosLR], pointers.processRead[prevPosLR +1], pointers.processRead[prevPosLR +2]);

	float XLRatio = x % 2 == 1 ? 0.25 : 0.75;
	float YLRatio = y % 2 == 1 ? 0.25 : 0.75;
	float3 UpperTotal = UL * XLRatio + UR * (1.-XLRatio);
	float3 LowerTotal = LL * XLRatio + LR * (1.-XLRatio);
	float3 Total = UpperTotal*YLRatio + LowerTotal * (1. - YLRatio);

	pointers.processWrite[myPos] = Total.x;
	pointers.processWrite[myPos + 1] = Total.y;
	pointers.processWrite[myPos + 2] = Total.z;
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

	pointers.finalOut[firstPos] = powf(pointers.processRead[firstPos],1) + pointers.inputImage[firstPos];
	pointers.finalOut[firstPos + 1] = powf(pointers.processRead[firstPos+1],1) + pointers.inputImage[firstPos+1];
	pointers.finalOut[firstPos + 2] = powf(pointers.processRead[firstPos+2],1) + pointers.inputImage[firstPos+2];
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
launch_cudaBlur(dim3 grid, dim3 block, int sbytes, int imgw, int imgh, int currRatio, PostProcessPointers pointers)
{

	cudaBlur << < grid, block, sbytes >> > (pointers, imgw, imgh, currRatio);
}

extern "C" void
launch_cudaBlur2(dim3 grid, dim3 block, int sbytes, int imgw, int imgh, bool isHorizontal, int currRatio, PostProcessPointers pointers)
{

	cudaBlur2 << < grid, block, sbytes >> > (pointers, imgw, imgh, isHorizontal, currRatio);
}

extern "C" void
launch_cudaDownSampleToHalfRes(dim3 grid, dim3 block, int sbytes, int imgw, int imgh, int currRatio, PostProcessPointers pointers)
{

	cudaDownSampleToHalfRes << < grid, block, sbytes >> > (pointers, imgw, imgh, currRatio);
}

extern "C" void
launch_cudaUpSampleToDoubleRes(dim3 grid, dim3 block, int sbytes, int imgw, int imgh, int currRatio, PostProcessPointers pointers)
{

	cudaUpSampleToDoubleRes<< < grid, block, sbytes >> > (pointers, imgw, imgh, currRatio);
}
