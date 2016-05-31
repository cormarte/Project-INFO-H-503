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

typedef int HistogramType;




__global__ void gpuApplyTransform(const unsigned char* devOriginalImage, unsigned char* devTransformedImage, const int width, const int height, const double tx, const double ty, const double rz) {

	// Pixel coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {

		int centerX = width / 2 - tx;
		int centerY = height / 2 - ty;

		int originalX = (int)((x - centerX)*cos(-rz * M_PI / 180.0) - (y - centerY)*sin(-rz * M_PI / 180.0) - tx + centerX);
		int originalY = (int)((x - centerX)*sin(-rz * M_PI / 180.0) + (y - centerY)*cos(-rz * M_PI / 180.0) - ty + centerY);

		if (originalX >= 0 && originalX < width && originalY >= 0 && originalY < height) {

			devTransformedImage[x + width * y] = devOriginalImage[originalX + width * originalY];
		}

		else {

			devTransformedImage[x + width * y] = 0;
		}
	}
}




__global__ void gpuGlobalHistogram2D(const unsigned char* devFloatingImage, const unsigned char* devReferenceImage, const int width, const int height, HistogramType* devHistogram2D) {

	/* Computes 2D histogram in global memory */


	// Pixel coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Image boundaries check and global histogram incrementation using adomicAdd
	if (x < width && y < height) {

		unsigned char f = devFloatingImage[x + width * y];
		unsigned char r = devReferenceImage[x + width * y];

		atomicAdd(&devHistogram2D[f + 256 * r], 1);
	}
}




__global__ void gpuSharedHistogram2D(const unsigned char* devFloatingImage, const unsigned char* devReferenceImage, const int width, const int height, HistogramType* devHistogram2D) {

	/* Uses shared memory to store local 2D histograms. However, local histograms size cannot be 256x256,
	   which would require 64kB of share memory (only 48kB are available). 2 thread blocks are used for
	   each pixel block instead. The first one is in charge of the F range [0 127] and the second one of
	   the F range [128 255], leading to 128*256 = 32kB of shared memory. */


	__shared__  unsigned char localHistogramFR[128 * 256];

	// Pixel coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// 1D thread index within a block
	int bt = threadIdx.x + threadIdx.y * blockDim.x;

	// Number of threads within a bloc
	int bnt = blockDim.x * blockDim.y;
	
	// Local histogram initialization
	for (int i = bt; i < 128 * 256; i += bnt) {

		localHistogramFR[i] = 0;
	}

	__syncthreads();

	if (x < width && y < height) {

		unsigned char f = devFloatingImage[x + width * y];
		unsigned char r = devReferenceImage[x + width * y];

		if ((!(blockIdx.x % 2) && f < 128) || ((blockIdx.x % 2) && f >= 128)) {
		

			//atomicAdd(&localHistogramFR[f - 128 * (blockIdx.x % 2) + 128 * r], 1);

			// Check for local histogram bin overflow
			if (localHistogramFR[f - 128 * (blockIdx.x % 2) + 128 * r] == 255) {
			
				//atomicExch(&localHistogramFR[f - 128 * (blockIdx.x % 2) + 128 * r], 0);
				atomicAdd(&devHistogram2D[f + 256 * r], 255);
			}
		}
	}

	__syncthreads();

	for (int i = bt; i < 128 * 256; i += bnt) {

		unsigned char f = i % 128 + 128 * (blockIdx.x % 2);
		unsigned char r = i / 128;

		atomicAdd(&devHistogram2D[f + 256 * r], localHistogramFR[i]);
	}
}




__global__ void gpuSharedHistogram1D(const unsigned char* image, const int width, const int height, HistogramType* histogram) {

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
		
		atomicAdd(&localHistogram[ image[x + width * y] ], 1);
	}

	__syncthreads();
	
	for (int i = bt; i < 256; i += bnt) {
	
		atomicAdd(&histogram[bt], localHistogram[i]);
	}
}




__global__ void gpuPartialMutualInformation(const HistogramType* histogram1, const HistogramType* histogram2, const HistogramType* histogram2D, int width, int height,  double* partialMutualInformation) {

	// Bin coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Partial mutual information
	if (x < 256 && y < 256) {

		int histogramSum = width * height;
		double p1 = (1.0 * histogram1[x]) / histogramSum;
		double p2 = (1.0 * histogram2[y]) / histogramSum;
		double p12 = (1.0 * histogram2D[x + 256 * y]) / histogramSum;

		if (p12 != 0) {

			partialMutualInformation[x + 256 * y] = p12*log2(p12 / (p1 * p2));
		}

		else {
			partialMutualInformation[x + 256 * y] = 0;
		}		 
	}
}




template <unsigned int blockSize>
__global__ void gpuReduce(const double* inputData, double* outputData)
{
	// Dynamic shared memory allocation
	__shared__ double localData[blockSize];

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




// To be replaced by a gpu implementation
template<typename HistogramType, int histogramSize>
double cpuMutualInformation(const HistogramType* histogram2D){

	double histogramSum = 0;
	HistogramType histogram1[histogramSize] = {};
	HistogramType histogram2[histogramSize] = {};

	for (int bin1 = 0; bin1 != histogramSize; bin1++) {

		for (int bin2 = 0; bin2 != histogramSize; bin2++) {

			histogramSum += histogram2D[bin1 + histogramSize * bin2];
			histogram1[bin1] += histogram2D[bin1 + histogramSize * bin2];
			histogram2[bin2] += histogram2D[bin1 + histogramSize * bin2];
		}
	}

	double mutualInformation = 0;

	for (int bin1 = 0; bin1 != histogramSize; bin1++) {

		for (int bin2 = 0; bin2 != histogramSize; bin2++) {

			double p1 = histogram1[bin1] / histogramSum;
			double p2 = histogram2[bin2] / histogramSum;
			double p12 = histogram2D[bin1 + histogramSize * bin2] / histogramSum;

			if (p12 != 0) {

				mutualInformation += p12*log2(p12 / (p1 * p2));
			}
		}
	}

	return mutualInformation;
}




Image gpuRegister(const Image& hostImageF, const Image& hostImageR) {

	const int width = hostImageF.width;
	const int height = hostImageF.height;
	const int nbReductionBlocks = (256 * 256 + BLOCKDIM1D - 1) / (2 * BLOCKDIM1D); // First addition performed during shared memory loading, the number of blocks is thus reduced by two

	// Declarations
	unsigned char* devFloatingImage;
	unsigned char* devReferenceImage;
	unsigned char* devTransformedImage;
	unsigned char* hostTransformedImage = new unsigned char[width * height];
	HistogramType* devTransformedHistogram;
	HistogramType* devReferenceHistogram;
	HistogramType* devHistogram2D;
	HistogramType* hostFloatingHistogram = new HistogramType[256](); // DEBUG 
	HistogramType* hostReferenceHistogram = new HistogramType[256](); // DEBUG
	HistogramType* hostHistogram2D = new HistogramType[256 * 256]();
	double* devPartialMutualInformation;
	double* devReducedPartialMutualInformation;
	double* hostReducedPartialMutualInformation = new double[nbReductionBlocks];

	// Device selection
	CHECK(cudaSetDevice(0));

	// Limit size
	//CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1073741824));

	// Device memory allocation
	CHECK(cudaMalloc((void**)&devFloatingImage, width * height * sizeof(unsigned char)));
	CHECK(cudaMalloc((void**)&devReferenceImage, width * height * sizeof(unsigned char)));
	CHECK(cudaMalloc((void**)&devTransformedImage, width * height * sizeof(unsigned char)));
	CHECK(cudaMalloc((void**)&devTransformedHistogram, 256 * sizeof(HistogramType)));
	CHECK(cudaMalloc((void**)&devReferenceHistogram, 256 * sizeof(HistogramType)));
	CHECK(cudaMalloc((void**)&devHistogram2D, 256 * 256 * sizeof(HistogramType)));
	CHECK(cudaMalloc((void**)&devPartialMutualInformation, 256 * 256 * sizeof(double)));
	CHECK(cudaMalloc((void**)&devReducedPartialMutualInformation, nbReductionBlocks * sizeof(double)));

	// Host to device copy
	CHECK(cudaMemcpy(devFloatingImage, hostImageF.pixels, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(devReferenceImage, hostImageR.pixels, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));

	// Blocks and grid dimensions
	dim3 blockDimensions(BLOCKDIMX, BLOCKDIMY);
	dim3 gridDimensions((width + BLOCKDIMX - 1) / BLOCKDIMX, (height + BLOCKDIMY - 1) / BLOCKDIMY);

	// Reference image histogram 1D
	// Should only be computed once
	CHECK(cudaMemset(devReferenceHistogram, 0, 256 * sizeof(HistogramType)));
	gpuSharedHistogram1D << < gridDimensions, blockDimensions >> >(devReferenceImage, width, height, devReferenceHistogram);
	
	// Transformed image histogram 1D
	// There's no 1-to-1 correspondance between floating and transformed image, due to nearest neighbour approximation
	// Transformed image histogram have to be recomputed at after every transform (=> check if the results are significantly better !!!)
	/* CHECK(cudaMemset(devTransformedHistogram, 0, 256 * sizeof(HistogramType)));
	gpuSharedHistogram1D << < gridDimensions, blockDimensions >> >(devTransformedImage, width, height, devTransformedHistogram); */

	// Tested transforms
	vector<double> translationsX;
	vector<double> translationsY;
	vector<double> rotationsZ;

	for (int i = 0; i != 1; i++) {

		translationsX.push_back(i + 10);
		translationsY.push_back(i - 20);
		rotationsZ.push_back(i - 30);
	}

	// Resgistration
	Transform optimalTransform = { 0, 0, 0, 0, 0, 0 };
	double hostMaxMutualInformation = 0;
	/*double progress = 0;
	double step = 100.0 / (translationsX.size()*translationsY.size()*rotationsZ.size());*/

	for (int a = 0; a != translationsX.size(); a++) {

		for (int b = 0; b != translationsY.size(); b++) {

			for (int c = 0; c != rotationsZ.size(); c++) {

				Transform transform = { translationsX[a], translationsY[b], 0, 0, 0, rotationsZ[c] };				

				// Blocks and grid dimensions
				blockDimensions = dim3(BLOCKDIMX, BLOCKDIMY);
				gridDimensions = dim3((width + BLOCKDIMX - 1) / BLOCKDIMX, (height + BLOCKDIMY - 1) / BLOCKDIMY);

				// Transform
				gpuApplyTransform << < gridDimensions, blockDimensions >> >(devFloatingImage, devTransformedImage, width, height, transform.tx, transform.ty, transform.rz);

				// Transformed image histogram 1D
				// There's no 1-to-1 correspondance between floating and transformed image, due to nearest neighbour approximation
				// Transformed image histogram have to be recomputed at after every transform (=> check if the results are significantly better !!!)
				CHECK(cudaMemset(devTransformedHistogram, 0, 256 * sizeof(HistogramType)));
				gpuSharedHistogram1D << < gridDimensions, blockDimensions >> >(devTransformedImage, width, height, devTransformedHistogram);
				
				// Histogram 2D
				CHECK(cudaMemset(devHistogram2D, 0, 256 * 256 * sizeof(HistogramType)));
				gpuGlobalHistogram2D << < gridDimensions, blockDimensions >> >(devTransformedImage, devReferenceImage, width, height, devHistogram2D);

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
				CHECK(cudaMemcpy(hostReducedPartialMutualInformation, devReducedPartialMutualInformation, nbReductionBlocks * sizeof(double), cudaMemcpyDeviceToHost));

				// Final reduction on CPU
				double hostMutualInformation = 0;

				for (int i = 0; i < nbReductionBlocks; i++) {
				
					hostMutualInformation += hostReducedPartialMutualInformation[i];

					//cout << hostReducedPartialMutualInformation[i] << endl;
				}

				//cout << "GPU mutual information: " << hostMutualInformation << endl;

				// Transformation evaluation
				if (hostMutualInformation > hostMaxMutualInformation) {

					hostMaxMutualInformation = hostMutualInformation;
					optimalTransform = transform;
				}

				/*progress += step;
				cout << progress << "%" << endl;*/

				// Old version using CPU MI computation
				/* // Device to host copy
				CHECK(cudaMemcpy(hostHistogram2D, devHistogram2D, 256 * 256 * sizeof(HistogramType), cudaMemcpyDeviceToHost));

				// Mutual information
				double mutualInformation = cpuMutualInformation<HistogramType, 256>(hostHistogram2D); */

				//
			}
		}
	}

	// Blocks and grid dimensions
    blockDimensions = dim3(BLOCKDIMX, BLOCKDIMY);
	gridDimensions = dim3((width + BLOCKDIMX - 1) / BLOCKDIMX, (height + BLOCKDIMY - 1) / BLOCKDIMY);

	// Result
	cout << "Optimal transform: Tx: " << optimalTransform.tx << ", Ty: " << optimalTransform.ty << ", Rz: " << optimalTransform.rz << endl;
	gpuApplyTransform << < gridDimensions, blockDimensions >> >(devFloatingImage, devTransformedImage, width, height, optimalTransform.tx, optimalTransform.ty, optimalTransform.rz);

	// Wait for GPU
	CHECK(cudaDeviceSynchronize());

	// Device to host copy
	CHECK(cudaMemcpy(hostTransformedImage, devTransformedImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	// Delete
	delete hostHistogram2D;
	delete hostReducedPartialMutualInformation;
	// /!\ DEALLOCATE GPU MEMORY /!\

	// Transformed image
	Image transformedImage = { width, height, hostTransformedImage };
	return transformedImage;
	


	
	/* // Test using globalHistogram2D

	// Bloc and grid dimensions
	dim3 blockDimensions(BLOCKDIMX, BLOCKDIMY);
	dim3 gridDimensions((width + BLOCKDIMX - 1) / BLOCKDIMX, (height + BLOCKDIMY - 1) / BLOCKDIMY);

	gpuGlobalHistogram2D << < gridDimensions, blockDimensions >> >(devFloatingImage, devReferenceImage, width, height, devHistogram2D);

	// Wait for GPU
	CHECK(cudaDeviceSynchronize());

	// Device to host copy
	CHECK(cudaMemcpy(hostHistogram2D, devHistogram2D, 256 * 256 * sizeof(HistogramType), cudaMemcpyDeviceToHost));

	/* // Print histogram
	for (int binF = 0; binF != 256; binF++) {

		for (int binR = 0; binR != 256; binR++) {

			if (hostHistogram2D[binF + 256 * binR] != 0) {

				cout << "[" << binF << ", " << binR << "] : " << hostHistogram2D[binF + 256 * binR] << endl;
			}
		}
	} */ 




	/* // Test using gpuApplyTransform

	// Bloc and grid dimensions
	blockDimensions = dim3(BLOCKDIMX, BLOCKDIMY);
	gridDimensions = dim3((width + BLOCKDIMX - 1) / BLOCKDIMX, (height + BLOCKDIMY - 1) / BLOCKDIMY);

	gpuApplyTransform << < gridDimensions, blockDimensions >> >(devFloatingImage, devTransformedImage, width, height, 19, -4, -20);

	// Wait for GPU
	CHECK(cudaDeviceSynchronize());

	// Device to host copy
	CHECK(cudaMemcpy(hostTransformedImage, devTransformedImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	// Transformed image
	Image transformedImage = {width, height, hostTransformedImage};
	return transformedImage; */




	/* // Test using locallHistogram2D

	// Bloc and grid dimensions
	blockDimensions = dim3(BLOCKDIMX, BLOCKDIMY);
	gridDimensions = dim3(2 * (height + BLOCKDIMX - 1) / BLOCKDIMX, (width + BLOCKDIMY - 1) / BLOCKDIMY);

	gpuSharedHistogram2D << < gridDimensions, blockDimensions >> >(devFloatingImage, devReferenceImage, width, height, devHistogram2D);

	// Wait for GPU
	CHECK(cudaDeviceSynchronize());

	// Device to host copy
	CHECK(cudaMemcpy(hostHistogram2D, devHistogram2D, 256 * 256 * sizeof(HistogramType), cudaMemcpyDeviceToHost));

	// Test
	for (int binF = 0; binF != 256; binF++) {

		for (int binR = 0; binR != 256; binR++) {

			if (hostHistogram2D[binF + 256 * binR] != 0) {

				cout << "[" << binF << ", " << binR << "] : " << hostHistogram2D[binF + 256 * binR] << endl;
			}
		}
	} */
}