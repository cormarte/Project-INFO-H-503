#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <chrono>
#include <iostream>
#include <string>
#include <stb_image.h>
#include <stb_image_write.h>

#include "cpuRegistration.h"
#include "gpuRegistration.h"

using namespace std;


int main(int argc, char* argv[]) {

	// Pixels loading
	int fHeight, fWidth, fBitsPerPixel, rHeight, rWidth, rBitsPerPixel;
	unsigned char* fPixels = stbi_load("..//data//floatingImage.jpg", &fWidth, &fHeight, &fBitsPerPixel, 1);
	unsigned char* rPixels = stbi_load("..//data//referenceImage.jpg", &rWidth, &rHeight, &rBitsPerPixel, 1);

	if (fHeight != rHeight || fWidth != rWidth) {
	
		cout << "Error: both images must have the same dimensions!" << endl;
	}

	else {

		// Images definition
		Image floatingImage = {(unsigned int)fWidth, (unsigned int)fHeight, fPixels};
		Image referenceImage = {(unsigned int)rWidth, (unsigned int)rHeight, rPixels};

		// CPU registration
		auto begin = chrono::high_resolution_clock::now();
		Image cpuRegisteredImage = cpuRegister(floatingImage, referenceImage);
		auto end = chrono::high_resolution_clock::now();
		cout << "CPU: " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "us" << std::endl;

		// GPU registration
		begin = chrono::high_resolution_clock::now();
		Image gpuRegisteredImage = gpuRegister(floatingImage, referenceImage);
		end = chrono::high_resolution_clock::now();
		cout << "GPU: " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "us" << std::endl;

		// Image export
		stbi_write_png("..//data//cpuRegistered.png", cpuRegisteredImage.width, cpuRegisteredImage.height, 1, cpuRegisteredImage.pixels, cpuRegisteredImage.width);
		stbi_write_png("..//data//gpuRegistered.png", gpuRegisteredImage.width, gpuRegisteredImage.height, 1, gpuRegisteredImage.pixels, gpuRegisteredImage.width);

		// Memory free
		stbi_image_free(floatingImage.pixels);
		stbi_image_free(referenceImage.pixels);
		stbi_image_free(cpuRegisteredImage.pixels);
		stbi_image_free(gpuRegisteredImage.pixels);
	}

	// Prompt to end
	int end;
	std::cin >> end;

	return 0;
}