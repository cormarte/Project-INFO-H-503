#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "image.h"

// Defininitions
#define BLOCKDIMX 16
#define BLOCKDIMY 16

typedef unsigned int HistogramType;

__global__ void gpuHistogram2D(unsigned char* devImageF, unsigned char* devImageR, int width, int height, HistogramType* devHistogramFR) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char f = devImageF[i + width * j];
	unsigned char r = devImageR[i + width * j];

	atomicAdd(&devHistogramFR[f + 256 * r], 1);
}

void gpuRegistration(Image hostImageF, Image hostImageR) {

	int width = hostImageF.width;
	int height = hostImageF.height;

	// Handles declaration for both floating and reference images
	unsigned char* devImageF;
	unsigned char* devImageR;
	HistogramType* devHistogramFR;

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);

	// Device memory allocation
	cudaStatus = cudaMalloc((void**)&devImageF, width * height * sizeof(unsigned char));
	cudaStatus = cudaMalloc((void**)&devImageR, width * height * sizeof(unsigned char));
	cudaStatus = cudaMalloc((void**)&devHistogramFR, 256 * 256 * sizeof(HistogramType));

	// Device copy
	cudaStatus = cudaMemcpy(devImageF, hostImageF.pixels, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(devImageR, hostImageR.pixels, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

	// Bloc and grid dimensions
	dim3 blocDimensions(BLOCKDIMX, BLOCKDIMY);
	dim3 griDimensions((height + BLOCKDIMX - 1) / BLOCKDIMX, (width + BLOCKDIMY - 1) / BLOCKDIMY);

	gpuHistogram2D << < griDimensions, blocDimensions >> >(devImageF, devImageR, width, height, devHistogramFR);

	cudaStatus = cudaDeviceSynchronize();
}