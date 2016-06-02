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