__global__ void gpuApplyTransform(const unsigned char* devOriginalImage, unsigned char* devTransformedImage, const int width, const int height, const int tx, const int ty, const float rz) {

	// Pixel coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {

		float cosrz, sinrz;
		__sincosf(-rz * M_PI / 180.0f, &sinrz, &cosrz);

		int centerX = width / 2 - tx;
		int centerY = height / 2 - ty;

		int originalX = (int)((x - centerX)*cosrz - (y - centerY)*sinrz - tx + centerX);
		int originalY = (int)((x - centerX)*sinrz + (y - centerY)*cosrz - ty + centerY);

		if (originalX >= 0 && originalX < width && originalY >= 0 && originalY < height) {

			devTransformedImage[x + width * y] = devOriginalImage[originalX + width * originalY];
		}

		else {

			devTransformedImage[x + width * y] = 0;
		}
	}
}