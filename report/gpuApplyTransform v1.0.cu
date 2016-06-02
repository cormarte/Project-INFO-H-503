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