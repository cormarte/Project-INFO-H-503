__global__ void gpuPartialMutualInformation(const HistogramType* histogram1, const HistogramType* histogram2, const HistogramType* histogram2D, const unsigned int width, const unsigned int height, double* partialMutualInformation) {

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