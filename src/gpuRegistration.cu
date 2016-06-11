#define _USE_MATH_DEFINES

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <math.h>
#include <vector>

#include "common.h"
#include "image.h"
#include "powell.h"
#include "transform.h"

using namespace std;

// Defininitions
#define BLOCKDIMX 16
#define BLOCKDIMY 16
#define BLOCKDIM1D 1024 // For histogram reduction, to adapt to max number of threads per block
#define PI 3.14159265358979f

typedef unsigned int HistogramType;

// Variables
size_t imagePitch;
unsigned int hostWidth;
unsigned int hostHeight;
unsigned char* devFloatingImage;
unsigned char* devReferenceImage;
unsigned char* devTransformedImage;
unsigned char* hostTransformedImage;
HistogramType* devTransformedHistogram;
HistogramType* devReferenceHistogram;
HistogramType* devHistogram2D;
float* devPartialMutualInformation;
float* devReducedPartialMutualInformation;
float* hostReducedPartialMutualInformation;
const unsigned int nbReductionBlocks = (256 * 256 + BLOCKDIM1D - 1) / (2 * BLOCKDIM1D); // First addition performed during shared memory loading, the number of blocks is thus reduced by two


__global__ void gpuApplyTransform(const unsigned char* originalImage, unsigned char* transformedImage, const unsigned int width, const unsigned int height, const size_t pitch, const float tx, const float ty, const float rz) {

	// Pixel coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {

		float cosrz, sinrz;
		__sincosf(-rz * PI / 180.0f, &sinrz, &cosrz);

		float centerX = width / 2 - tx;
		float centerY = height / 2 - ty;

		int originalX = (int)((x - centerX)*cosrz - (y - centerY)*sinrz - tx + centerX);
		int originalY = (int)((x - centerX)*sinrz + (y - centerY)*cosrz - ty + centerY);

		if (originalX >= 0 && originalX < width && originalY >= 0 && originalY < height) {

			*(unsigned char*)(((char*)transformedImage + y * pitch) + x) = *(unsigned char*)(((char*)originalImage + originalY * pitch) + originalX);
		}

		else {

			*(unsigned char*)(((char*)transformedImage + y * pitch) + x) = 255;
		}
	}
}



__global__ void gpuGlobalHistogram1D(const unsigned char* image, const unsigned int width, const unsigned int height, const size_t pitch, HistogramType* histogram) {

	// Pixel coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {

		atomicAdd(&histogram[*(unsigned char*)(((char*)image + y * pitch) + x)], 1);
	}
}

__global__ void gpuSharedHistogram1D(const unsigned char* image, const unsigned int width, const unsigned int height, const size_t pitch, HistogramType* histogram) {

	__shared__  HistogramType localHistogram[256];

	// Pixel coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// 1D thread index within a block
	int bt = threadIdx.x + threadIdx.y * blockDim.x;

	// Number of threads within a bloc
	int bnt = blockDim.x * blockDim.y;

	// Local histogram initialization
	for (int i = bt; i < 256; i += bnt) {

		localHistogram[i] = 0;
	}

	__syncthreads();

	if (x < width && y < height) {

		atomicAdd(&localHistogram[*(unsigned char*)(((char*)image + y * pitch) + x)], 1);
	}

	__syncthreads();

	for (int i = bt; i < 256; i += bnt) {

		atomicAdd(&histogram[bt], localHistogram[i]);
	}
}


__global__ void gpuGlobalHistogram2D(const unsigned char* image1, const unsigned char* image2, const unsigned int width, const unsigned int height, const size_t imagePitch, HistogramType* histogram2D) {

	/* Computes 2D histogram in global memory */


	// Pixel coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Image boundaries check and global histogram incrementation using atomicAdd
	if (x < width && y < height) {

		unsigned char f = *(unsigned char*)(((char*)image1 + y * imagePitch) + x);
		unsigned char r = *(unsigned char*)(((char*)image2 + y * imagePitch) + x);

		atomicAdd(&histogram2D[f + 256 * r], 1);
	}
}


__global__ void gpuPartialMutualInformation(const HistogramType* histogram1, const HistogramType* histogram2, const HistogramType* histogram2D, const unsigned int width, const unsigned int height, float* partialMutualInformation) {

	// Bin coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Partial mutual information
	if (x < 256 && y < 256) {

		unsigned int histogramSum = width * height;
		float p1 = (1.0f * histogram1[x]) / histogramSum;
		float p2 = (1.0f * histogram2[y]) / histogramSum;
		float p12 = (1.0f * histogram2D[x + 256 * y]) / histogramSum;

		if (p12 != 0) {

			partialMutualInformation[x + 256 * y] = p12*__log2f(p12 / (p1 * p2));
		}

		else {

			partialMutualInformation[x + 256 * y] = 0.0f;
		}		 
	}
}

__global__ void gpuNaiveReduce(const float* inputData, float* outputData) {

	__shared__ float localData[BLOCKDIM1D];

	unsigned int tid = threadIdx.x;
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;

	localData[tid] = inputData[i];

	__syncthreads();

	for (unsigned int s=1; s<blockDim.x; s*=2) {

		if (tid % (2 * s) == 0) {

			localData[tid] += localData[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {

		outputData[blockIdx.x] = localData[0];
	}
}

template <unsigned int blockSize>
__global__ void gpuReduce(const float* inputData, float* outputData)
{
	// Shared memory allocation
	__shared__ float localData[blockSize];

	unsigned int tid = threadIdx.x;

	// First addition during shared memory loading, number of blocks is thus reduced by two
	unsigned int i = tid + blockIdx.x * (blockDim.x * 2);
	localData[tid] = inputData[i] + inputData[i + blockDim.x];

	__syncthreads();

	// Complete unrolling
	// Statement choice made at compile time according to template argument
	if (blockSize >= 1024) { if (tid < 512) { localData[tid] += localData[tid + 512]; } __syncthreads(); }
	if (blockSize >= 512)  { if (tid < 256) { localData[tid] += localData[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256)  { if (tid < 128) { localData[tid] += localData[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128)  { if (tid < 64)  { localData[tid] += localData[tid +  64]; } __syncthreads(); }

	// Last warp unrolling
	// No synchronisation required within a single warp
	if (tid < 32) {

		if (blockSize >= 64) localData[tid] += localData[tid + 32];
		if (blockSize >= 32) localData[tid] += localData[tid + 16];
		if (blockSize >= 16) localData[tid] += localData[tid +  8];
		if (blockSize >= 8)  localData[tid] += localData[tid +  4];
		if (blockSize >= 4)  localData[tid] += localData[tid +  2];
		if (blockSize >= 2)  localData[tid] += localData[tid +  1];
	}

	// Copy the restul in global memory
	if (tid == 0) outputData[blockIdx.x] = localData[0];
}

float gpuPowellFunction(float* transformVector) {

	/* Calculates the mutual information of the reference image
	and the transformed image for a transform vector 'transformVector' */

	Transform transform = { transformVector[0], transformVector[1], 0, 0, 0, transformVector[2] };
	//cout << transformVector[0] << ", " << transformVector[1] << ", " << transformVector[2] << endl;

	// Blocks and grid dimensions
	dim3 blockDimensions = dim3(BLOCKDIMX, BLOCKDIMY);
	dim3 gridDimensions = dim3((hostWidth + BLOCKDIMX - 1) / BLOCKDIMX, (hostHeight + BLOCKDIMY - 1) / BLOCKDIMY);

	// Transform
	gpuApplyTransform << < gridDimensions, blockDimensions >> >(devFloatingImage, devTransformedImage, hostWidth, hostHeight, imagePitch, transform.tx, transform.ty, transform.rz);

	// Histogram 1D
	CHECK(cudaMemset(devTransformedHistogram, 0, 256 * sizeof(HistogramType)));
	gpuGlobalHistogram1D << < gridDimensions, blockDimensions >> >(devTransformedImage, hostWidth, hostHeight, imagePitch, devTransformedHistogram);

	// Histogram 2D
	CHECK(cudaMemset(devHistogram2D, 0, 256 * 256 * sizeof(HistogramType)));
	gpuGlobalHistogram2D << < gridDimensions, blockDimensions >> >(devTransformedImage, devReferenceImage, hostWidth, hostHeight, imagePitch, devHistogram2D);

	// Grid redimensioning
	gridDimensions = dim3((256 + BLOCKDIMX - 1) / BLOCKDIMX, (256 + BLOCKDIMY - 1) / BLOCKDIMY);

	// Partial mutual information
	gpuPartialMutualInformation << < gridDimensions, blockDimensions >> > (devTransformedHistogram, devReferenceHistogram, devHistogram2D, hostWidth, hostHeight, devPartialMutualInformation);

	// Blocks and grid redimensioning
	blockDimensions = dim3(BLOCKDIM1D);
	gridDimensions = dim3(nbReductionBlocks);

	// Partial mutual information reduction
	gpuReduce <BLOCKDIM1D> << < gridDimensions, blockDimensions >> > (devPartialMutualInformation, devReducedPartialMutualInformation);

	// Wait for GPU
	CHECK(cudaDeviceSynchronize());

	// Reduced partial mutual information copy
	CHECK(cudaMemcpy(hostReducedPartialMutualInformation, devReducedPartialMutualInformation, nbReductionBlocks * sizeof(float), cudaMemcpyDeviceToHost));

	// Final reduction on CPU
	float hostMutualInformation = 0;

	for (int i = 0; i < nbReductionBlocks; i++) {

		hostMutualInformation += hostReducedPartialMutualInformation[i];
	}

	//cout << hostMutualInformation << endl;

	return (-1.0f)*hostMutualInformation;
}

Image gpuRegister(const Image& hostFloatingImage, const Image& hostReferenceImage) {

	// Initialisations
	hostWidth = hostFloatingImage.width;
	hostHeight = hostFloatingImage.height;
	hostTransformedImage = new unsigned char[hostWidth * hostHeight];
	hostReducedPartialMutualInformation = new float[nbReductionBlocks];

	// Device selection
	CHECK(cudaSetDevice(0));
	
	// Device memory allocation
	CHECK(cudaMallocPitch(&devFloatingImage, &imagePitch, hostWidth*sizeof(unsigned char), hostHeight*sizeof(unsigned char)));
	CHECK(cudaMallocPitch(&devReferenceImage, &imagePitch, hostWidth*sizeof(unsigned char), hostHeight*sizeof(unsigned char)));
	CHECK(cudaMallocPitch(&devTransformedImage, &imagePitch, hostWidth*sizeof(unsigned char), hostHeight*sizeof(unsigned char)));
	CHECK(cudaMalloc((void**)&devTransformedHistogram, 256 * sizeof(HistogramType)));
	CHECK(cudaMalloc((void**)&devReferenceHistogram, 256 * sizeof(HistogramType)));
	CHECK(cudaMalloc((void**)&devHistogram2D, 256 * 256 * sizeof(HistogramType)));
	CHECK(cudaMalloc((void**)&devPartialMutualInformation, 256 * 256 * sizeof(float)));
	CHECK(cudaMalloc((void**)&devReducedPartialMutualInformation, nbReductionBlocks * sizeof(float)));

	// Host to device copy
	CHECK(cudaMemcpy2D(devFloatingImage, imagePitch, hostFloatingImage.pixels, hostWidth*sizeof(unsigned char), hostWidth*sizeof(unsigned char), hostHeight*sizeof(unsigned char), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy2D(devReferenceImage, imagePitch, hostReferenceImage.pixels, hostWidth*sizeof(unsigned char), hostWidth*sizeof(unsigned char), hostHeight*sizeof(unsigned char), cudaMemcpyHostToDevice));

	// Blocks and grid dimensions
	dim3 blockDimensions(BLOCKDIMX, BLOCKDIMY);
	dim3 gridDimensions((hostWidth + BLOCKDIMX - 1) / BLOCKDIMX, (hostHeight + BLOCKDIMY - 1) / BLOCKDIMY);

	// Reference image histogram 1D
	// Should only be computed once
	CHECK(cudaMemset(devReferenceHistogram, 0, 256 * sizeof(HistogramType)));
	gpuGlobalHistogram1D << < gridDimensions, blockDimensions >> >(devReferenceImage, hostWidth, hostHeight, imagePitch, devReferenceHistogram);
	
	// Transformed image histogram 1D
	// Should only be computed once
	/*CHECK(cudaMemset(devTransformedHistogram, 0, 256 * sizeof(HistogramType)));
	gpuGlobalHistogram1D << < gridDimensions, blockDimensions >> >(devFloatingImage, hostWidth, hostHeight, imagePitch, devTransformedHistogram); */

	// Powell
	float* maxMutualInformation = new float(FLT_MAX);
	float transformVector[3] = { 0.0f, 0.0f, 0.0f };

	powell(maxMutualInformation, transformVector, 3, 2.0e-4f, gpuPowellFunction);

	// Final transform
	Transform transform = { transformVector[0], transformVector[1], 0, 0, 0, transformVector[2] };
	cout << "GPU optimal transform: Tx: " << transform.tx << ", Ty: " << transform.ty << ", Rz: " << transform.rz << endl;
	cout << "GPU max mutual information: " << (-1.0f) * *maxMutualInformation << endl;

	// Result
	gpuApplyTransform << < gridDimensions, blockDimensions >> >(devFloatingImage, devTransformedImage, hostWidth, hostHeight, imagePitch, transform.tx, transform.ty, transform.rz);

	// Wait for GPU
	CHECK(cudaDeviceSynchronize());

	// Device to host copy
	CHECK(cudaMemcpy2D(hostTransformedImage, hostWidth*sizeof(unsigned char), devTransformedImage, imagePitch, hostWidth * sizeof(unsigned char), hostHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	// Delete
	delete hostReducedPartialMutualInformation;
	CHECK(cudaFree(devFloatingImage));
	CHECK(cudaFree(devReferenceImage));
	CHECK(cudaFree(devTransformedImage));
	CHECK(cudaFree(devReferenceHistogram));
	CHECK(cudaFree(devHistogram2D));
	CHECK(cudaFree(devPartialMutualInformation));
	CHECK(cudaFree(devReducedPartialMutualInformation));

	// Transformed image
	Image transformedImage = { (unsigned int)hostWidth, (unsigned int)hostHeight, hostTransformedImage };

	return transformedImage;
}