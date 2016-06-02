__global__ void gpuGlobalHistogram1D(const unsigned char* image, const int width, const int height, HistogramType* histogram) {

	// Pixel coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {

		atomicAdd(&histogram[image[x + width * y]], 1);
	}
}