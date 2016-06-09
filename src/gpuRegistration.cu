#define _USE_MATH_DEFINES

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <math.h>
#include <vector>

#include "common.h"
#include "image.h"
#include "transform.h"

using namespace std;

// Defininitions
#define BLOCKDIMX 16
#define BLOCKDIMY 16
#define BLOCKDIM1D 1024 // For histogram reduction, to adapt to max number of threads per block
#define PI 3.14159265358979f

typedef unsigned int HistogramType;


__global__ void gpuApplyTransform(const unsigned char* devOriginalImage, unsigned char* devTransformedImage, const unsigned int width, const unsigned int height, const size_t pitch, const int tx, const int ty, const float rz) {

	// Pixel coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {

		float cosrz, sinrz;
		__sincosf(-rz * PI / 180.0f, &sinrz, &cosrz);

		int centerX = width / 2 - tx;
		int centerY = height / 2 - ty;

		int originalX = (int)((x - centerX)*cosrz - (y - centerY)*sinrz - tx + centerX);
		int originalY = (int)((x - centerX)*sinrz + (y - centerY)*cosrz - ty + centerY);

		if (originalX >= 0 && originalX < width && originalY >= 0 && originalY < height) {

			*(unsigned char*)(((char*)devTransformedImage + y * pitch) + x) = *(unsigned char*)(((char*)devOriginalImage + originalY * pitch) + originalX);
		}

		else {

			*(unsigned char*)(((char*)devTransformedImage + y * pitch) + x) = 255;
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


__global__ void gpuGlobalHistogram2D(const unsigned char* devFloatingImage, const unsigned char* devReferenceImage, const unsigned int width, const unsigned int height, const size_t imagePitch, HistogramType* histogram2D) {

	/* Computes 2D histogram in global memory */


	// Pixel coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Image boundaries check and global histogram incrementation using adomicAdd
	if (x < width && y < height) {

		unsigned char f = *(unsigned char*)(((char*)devFloatingImage + y * imagePitch) + x);
		unsigned char r = *(unsigned char*)(((char*)devReferenceImage + y * imagePitch) + x);

		atomicAdd(&histogram2D[f + 256 * r], 1);
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


template <unsigned int blockSize>
__global__ void gpuReduce(const float* inputData, float* outputData)
{
	// Dynamic shared memory allocation
	__shared__ float localData[blockSize];

	unsigned int tid = threadIdx.x;

	// First addition during shared memory loading, number of blocks is thus reduced by two
	unsigned int i = tid + blockIdx.x * (blockDim.x * 2);
	localData[tid] = inputData[i] + inputData[i + blockDim.x];

	__syncthreads();

	// Complete unrolling
	// Statement choice made at compile time accroding to template argument
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


Image gpuRegister(const Image& hostFloatingImage, const Image& hostReferenceImage) {

	const unsigned int width = hostFloatingImage.width;
	const unsigned int height = hostFloatingImage.height;
	const unsigned int nbReductionBlocks = (256 * 256 + BLOCKDIM1D - 1) / (2 * BLOCKDIM1D); // First addition performed during shared memory loading, the number of blocks is thus reduced by two

	// Declarations
	size_t imagePitch;
	unsigned char* devFloatingImage;
	unsigned char* devReferenceImage;
	unsigned char* devTransformedImage;
	unsigned char* hostTransformedImage = new unsigned char[width * height];
	HistogramType* devTransformedHistogram;
	HistogramType* devReferenceHistogram;
	HistogramType* devHistogram2D;
	float* devPartialMutualInformation;
	float* devReducedPartialMutualInformation;
	float* hostReducedPartialMutualInformation = new float[nbReductionBlocks];

	// Device selection
	CHECK(cudaSetDevice(0));
	
	// Device memory allocation
	CHECK(cudaMallocPitch(&devFloatingImage, &imagePitch, width*sizeof(unsigned char), height*sizeof(unsigned char)));
	CHECK(cudaMallocPitch(&devReferenceImage, &imagePitch, width*sizeof(unsigned char), height*sizeof(unsigned char)));
	CHECK(cudaMallocPitch(&devTransformedImage, &imagePitch, width*sizeof(unsigned char), height*sizeof(unsigned char)));
	CHECK(cudaMalloc((void**)&devTransformedHistogram, 256 * sizeof(HistogramType)));
	CHECK(cudaMalloc((void**)&devReferenceHistogram, 256 * sizeof(HistogramType)));
	CHECK(cudaMalloc((void**)&devHistogram2D, 256 * 256 * sizeof(HistogramType)));
	CHECK(cudaMalloc((void**)&devPartialMutualInformation, 256 * 256 * sizeof(float)));
	CHECK(cudaMalloc((void**)&devReducedPartialMutualInformation, nbReductionBlocks * sizeof(float)));

	// Host to device copy
	CHECK(cudaMemcpy2D(devFloatingImage, imagePitch, hostFloatingImage.pixels, width*sizeof(unsigned char), width*sizeof(unsigned char), height*sizeof(unsigned char), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy2D(devReferenceImage, imagePitch, hostReferenceImage.pixels, width*sizeof(unsigned char), width*sizeof(unsigned char), height*sizeof(unsigned char), cudaMemcpyHostToDevice));

	// Blocks and grid dimensions
	dim3 blockDimensions(BLOCKDIMX, BLOCKDIMY);
	dim3 gridDimensions((width + BLOCKDIMX - 1) / BLOCKDIMX, (height + BLOCKDIMY - 1) / BLOCKDIMY);

	// Reference image histogram 1D
	// Should only be computed once
	CHECK(cudaMemset(devReferenceHistogram, 0, 256 * sizeof(HistogramType)));
	gpuGlobalHistogram1D << < gridDimensions, blockDimensions >> >(devReferenceImage, width, height, imagePitch, devReferenceHistogram);
	
	// Transformed image histogram 1D
	CHECK(cudaMemset(devTransformedHistogram, 0, 256 * sizeof(HistogramType)));
	gpuGlobalHistogram1D << < gridDimensions, blockDimensions >> >(devFloatingImage, width, height, imagePitch, devTransformedHistogram);

	// Tested transforms
	vector<int> translationsX;
	vector<int> translationsY;
	vector<float> rotationsZ;

	for (int i = 0; i != 1; i++) {

		translationsX.push_back(i + 10);
		translationsY.push_back(i - 20);
		rotationsZ.push_back(i - 30);
	}

	// Resgistration
	Transform optimalTransform = { 0, 0, 0, 0, 0, 0 };
	float hostMaxMutualInformation = 0;

	for (int a = 0; a != translationsX.size(); a++) {

		for (int b = 0; b != translationsY.size(); b++) {

			for (int c = 0; c != rotationsZ.size(); c++) {

				Transform transform = { translationsX[a], translationsY[b], 0, 0, 0, rotationsZ[c] };				

				// Blocks and grid dimensions
				blockDimensions = dim3(BLOCKDIMX, BLOCKDIMY);
				gridDimensions = dim3((width + BLOCKDIMX - 1) / BLOCKDIMX, (height + BLOCKDIMY - 1) / BLOCKDIMY);

				// Transform
				gpuApplyTransform << < gridDimensions, blockDimensions >> >(devFloatingImage, devTransformedImage, width, height, imagePitch, transform.tx, transform.ty, transform.rz);

				// Transformed image histogram 1D
				// There's no 1-to-1 correspondance between floating and transformed image, due to nearest neighbour approximation
				// Transformed image histogram have to be recomputed at after every transform (=> check if the results are significantly better !!!)
				//CHECK(cudaMemset(devTransformedHistogram, 0, 256 * sizeof(HistogramType)));
				//gpuGlobalHistogram1D << < gridDimensions, blockDimensions >> >(devTransformedImage, width, height, imagePitch, devTransformedHistogram);
				
				// Histogram 2D
				CHECK(cudaMemset(devHistogram2D, 0, 256 * 256 * sizeof(HistogramType)));
				gpuGlobalHistogram2D << < gridDimensions, blockDimensions >> >(devTransformedImage, devReferenceImage, width, height, imagePitch, devHistogram2D);

				// Grid redimensioning
				gridDimensions = dim3((256 + BLOCKDIMX - 1) / BLOCKDIMX, (256 + BLOCKDIMY - 1) / BLOCKDIMY);

				// Partial mutual information
				gpuPartialMutualInformation << < gridDimensions, blockDimensions >> > (devTransformedHistogram, devReferenceHistogram, devHistogram2D, width, height, devPartialMutualInformation);
				
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

				cout << "GPU mutual information: " << hostMutualInformation << endl;

				// Transformation evaluation
				if (hostMutualInformation > hostMaxMutualInformation) {

					hostMaxMutualInformation = hostMutualInformation;
					optimalTransform = transform;
				}
			}
		}
	}

	// Blocks and grid dimensions
    blockDimensions = dim3(BLOCKDIMX, BLOCKDIMY);
	gridDimensions = dim3((width + BLOCKDIMX - 1) / BLOCKDIMX, (height + BLOCKDIMY - 1) / BLOCKDIMY);

	// Result
	cout << "Optimal transform: Tx: " << optimalTransform.tx << ", Ty: " << optimalTransform.ty << ", Rz: " << optimalTransform.rz << endl;
	gpuApplyTransform << < gridDimensions, blockDimensions >> >(devFloatingImage, devTransformedImage, width, height, imagePitch, optimalTransform.tx, optimalTransform.ty, optimalTransform.rz);

	// Wait for GPU
	CHECK(cudaDeviceSynchronize());

	// Device to host copy
	CHECK(cudaMemcpy2D(hostTransformedImage, width*sizeof(unsigned char), devTransformedImage, imagePitch, width * sizeof(unsigned char), height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

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
	Image transformedImage = { (unsigned int)width, (unsigned int)height, hostTransformedImage };
	return transformedImage;
}