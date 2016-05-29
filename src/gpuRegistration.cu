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

typedef unsigned int HistogramType;




__global__ void gpuGlobalHistogram2D(const unsigned char* devImageF, const unsigned char* devImageR, const int width, const int height, HistogramType* devHistogramFR) {

	/* Computes 2D histogram in global memory */


	// Pixel coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// 1D global thread index
	int t = x + blockDim.x * gridDim.x * y;

	// Total number of threads
	int nt = gridDim.x * gridDim.y * blockDim.x * blockDim.y;

	// Histogam initialization to 0 is not guaranteed by using cudaMalloc but depends on the API
	// Either initialize it using cudaMemset or in a parallel way using the following code
	for (int i = t; i < 256 * 256; i += nt) {
	
		devHistogramFR[i] = 0;
	}

	__syncthreads();

	// Image boundaries check and global histogram incrementation using adomicAdd
	if (x < width && y < height) {

		unsigned char f = devImageF[x + width * y];
		unsigned char r = devImageR[x + width * y];

		atomicAdd(&devHistogramFR[f + 128 * r], 1);
	}
}




__global__ void gpuSharedHistogram2D(const unsigned char* devImageF, const unsigned char* devImageR, const int width, const int height, HistogramType* devHistogramFR) {

	/* Uses shared memory to store local 2D histograms. However, local histograms size cannot be 256x256,
	   which would require 64kB of share memory (only 48kB are available). 2 thread blocks are used for
	   each pixel block instead. The first one is in charge of the F range [0 127] and the second one of
	   the F range [128 255], leading to 128*256 = 32kB of shared memory. */


	__shared__  unsigned char localHistogramFR[128 * 256];

	// Pixel coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// 1D global thread index
	int t = x + blockDim.x * gridDim.x * y;

	// Total number of threads
	int nt = gridDim.x * gridDim.y * blockDim.x * blockDim.y;

	// 1D thread index within a block
	int bt = threadIdx.x + threadIdx.y * blockDim.x;

	// Number of threads within a bloc
	int bnt = blockDim.x * blockDim.y;

	// Global histogam initialization to 0 is not guaranteed by using cudaMalloc but depends on the API
	// Either initialize it using cudaMemset or in a parallel way using the following code
	for (int i = t; i < 256 * 256; i += nt) {

		devHistogramFR[i] = 0;
	}

	// Local histogram initialization
	for (int i = bt; i < 128 * 256; i += bnt) {

		localHistogramFR[i] = 0;
	}

	__syncthreads();

	if (x < width && y < height) {

		unsigned char f = devImageF[x + width * y];
		unsigned char r = devImageR[x + width * y];

		if ((!(blockIdx.x % 2) && f < 128) || ((blockIdx.x % 2) && f >= 128)) {
		

			//atomicAdd(&localHistogramFR[f - 128 * (blockIdx.x % 2) + 128 * r], 1);

			// Check for local histogram bin overflow
			if (localHistogramFR[f - 128 * (blockIdx.x % 2) + 128 * r] == 255) {
			
				//atomicExch(&localHistogramFR[f - 128 * (blockIdx.x % 2) + 128 * r], 0);
				atomicAdd(&devHistogramFR[f + 256 * r], 255);
			}
		}
	}

	__syncthreads();

	for (int i = bt; i < 128 * 256; i += bnt) {

		unsigned char f = i % 128 + 128 * (blockIdx.x % 2);
		unsigned char r = i / 128;

		atomicAdd(&devHistogramFR[f + 256 * r], localHistogramFR[i]);
	}
}





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

	int width = hostImageF.width;
	int height = hostImageF.height;

	// Declarations
	unsigned char* devImageF;
	unsigned char* devImageR;
	unsigned char* devTransformedImage;
	unsigned char* hostTransformedImage = new unsigned char[width * height];
	HistogramType* devHistogramFR;
	HistogramType* hostHistogramFR = new HistogramType[256 * 256]();

	// Device selection
	CHECK(cudaSetDevice(0));

	// Device memory allocation
	CHECK(cudaMalloc((void**)&devImageF, width * height * sizeof(unsigned char)));
	CHECK(cudaMalloc((void**)&devImageR, width * height * sizeof(unsigned char)));
	CHECK(cudaMalloc((void**)&devTransformedImage, width * height * sizeof(unsigned char)));
	CHECK(cudaMalloc((void**)&devHistogramFR, 256 * 256 * sizeof(HistogramType)));

	// Host to device copy
	CHECK(cudaMemcpy(devImageF, hostImageF.pixels, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(devImageR, hostImageR.pixels, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));

	// Histogam initialization to 0 is not guaranteed by using cudaMalloc but depends on the API
	// Either initialize it using cudaMemset or in a parallel way within the kernel
	/* CHECK(cudaMemset(devHistogramFR, 0, 256 * 256 * sizeof(HistogramType))); */

	// Blocks and grid dimensions
	dim3 blocDimensions(BLOCKDIMX, BLOCKDIMY);
	dim3 griDimensions((width + BLOCKDIMX - 1) / BLOCKDIMX, (height + BLOCKDIMY - 1) / BLOCKDIMY);

	// Tested transforms
	vector<double> translationsX;
	vector<double> translationsY;
	vector<double> rotationsZ;

	for (int i = 0; i != 20; i++) {

		translationsX.push_back(i + 10);
		translationsY.push_back(i - 20);
		rotationsZ.push_back(i - 30);
	}

	// Resgistration
	Transform optimalTransform = { 0, 0, 0, 0, 0, 0 };
	double maxMutualInformation = 0;
	/*double progress = 0;
	double step = 100.0 / (translationsX.size()*translationsY.size()*rotationsZ.size());*/

	for (int a = 0; a != translationsX.size(); a++) {

		for (int b = 0; b != translationsY.size(); b++) {

			for (int c = 0; c != rotationsZ.size(); c++) {

				Transform transform = { translationsX[a], translationsY[b], 0, 0, 0, rotationsZ[c] };

				// Transform
				gpuApplyTransform << < griDimensions, blocDimensions >> >(devImageF, devTransformedImage, width, height, transform.tx, transform.ty, transform.rz);

				// Wait for GPU
				CHECK(cudaDeviceSynchronize());

				// Histogram 2D
				gpuGlobalHistogram2D << < griDimensions, blocDimensions >> >(devTransformedImage, devImageR, width, height, devHistogramFR);

				// Wait for GPU
				CHECK(cudaDeviceSynchronize());

				// Device to host copy
				CHECK(cudaMemcpy(hostHistogramFR, devHistogramFR, 256 * 256 * sizeof(HistogramType), cudaMemcpyDeviceToHost));

				// Mutual information
				double mutualInformation = cpuMutualInformation<HistogramType, 256>(hostHistogramFR);

				// Test
				if (mutualInformation > maxMutualInformation) {

					maxMutualInformation = mutualInformation;
					optimalTransform = transform;
				}

				/*progress += step;
				cout << progress << "%" << endl;*/
			}
		}
	}

	// Result
	cout << "Optimal transform: Tx: " << optimalTransform.tx << ", Ty: " << optimalTransform.ty << ", Rz: " << optimalTransform.rz << endl;
	gpuApplyTransform << < griDimensions, blocDimensions >> >(devImageF, devTransformedImage, width, height, optimalTransform.tx, optimalTransform.ty, optimalTransform.rz);

	// Wait for GPU
	CHECK(cudaDeviceSynchronize());

	// Device to host copy
	CHECK(cudaMemcpy(hostTransformedImage, devTransformedImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	// Delete
	delete hostHistogramFR;
	// /!\ DEALLOCATE GPU MEMORY /!\

	// Transformed image
	Image transformedImage = { width, height, hostTransformedImage };
	return transformedImage;
	


	
	/* // Test using globalHistogram2D

	// Bloc and grid dimensions
	dim3 blocDimensions(BLOCKDIMX, BLOCKDIMY);
	dim3 griDimensions((width + BLOCKDIMX - 1) / BLOCKDIMX, (height + BLOCKDIMY - 1) / BLOCKDIMY);

	gpuGlobalHistogram2D << < griDimensions, blocDimensions >> >(devImageF, devImageR, width, height, devHistogramFR);

	// Wait for GPU
	CHECK(cudaDeviceSynchronize());

	// Device to host copy
	CHECK(cudaMemcpy(hostHistogramFR, devHistogramFR, 256 * 256 * sizeof(HistogramType), cudaMemcpyDeviceToHost));

	/* // Print histogram
	for (int binF = 0; binF != 256; binF++) {

		for (int binR = 0; binR != 256; binR++) {

			if (hostHistogramFR[binF + 256 * binR] != 0) {

				cout << "[" << binF << ", " << binR << "] : " << hostHistogramFR[binF + 256 * binR] << endl;
			}
		}
	} */ 




	/* // Test using gpuApplyTransform

	// Bloc and grid dimensions
	blocDimensions = dim3(BLOCKDIMX, BLOCKDIMY);
	griDimensions = dim3((width + BLOCKDIMX - 1) / BLOCKDIMX, (height + BLOCKDIMY - 1) / BLOCKDIMY);

	gpuApplyTransform << < griDimensions, blocDimensions >> >(devImageF, devTransformedImage, width, height, 19, -4, -20);

	// Wait for GPU
	CHECK(cudaDeviceSynchronize());

	// Device to host copy
	CHECK(cudaMemcpy(hostTransformedImage, devTransformedImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	// Transformed image
	Image transformedImage = {width, height, hostTransformedImage};
	return transformedImage; */




	/* // Test using locallHistogram2D

	// Bloc and grid dimensions
	blocDimensions = dim3(BLOCKDIMX, BLOCKDIMY);
	griDimensions = dim3(2 * (height + BLOCKDIMX - 1) / BLOCKDIMX, (width + BLOCKDIMY - 1) / BLOCKDIMY);

	gpuSharedHistogram2D << < griDimensions, blocDimensions >> >(devImageF, devImageR, width, height, devHistogramFR);

	// Wait for GPU
	CHECK(cudaDeviceSynchronize());

	// Device to host copy
	CHECK(cudaMemcpy(hostHistogramFR, devHistogramFR, 256 * 256 * sizeof(HistogramType), cudaMemcpyDeviceToHost));

	// Test
	for (int binF = 0; binF != 256; binF++) {

		for (int binR = 0; binR != 256; binR++) {

			if (hostHistogramFR[binF + 256 * binR] != 0) {

				cout << "[" << binF << ", " << binR << "] : " << hostHistogramFR[binF + 256 * binR] << endl;
			}
		}
	} */
}