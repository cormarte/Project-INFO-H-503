#define STB_IMAGE_IMPLEMENTATION

#include <iostream>
#include <string>
#include <stb_image.h>

using namespace std;

unsigned int* cpuHistogram1D(const unsigned char* image, const int width, const int height) {

	unsigned int* histogram = new unsigned int[256]();

	for (int y = 0; y != height; y++) {
	
		for (int x = 0; x != width; x++) {

			histogram[+image[x+width*y]]++;
		}
	}

	return histogram;
}

unsigned int* cpuHistogram2D(const unsigned char* image1, const unsigned char* image2, const int width, const int height) {

	unsigned int* histogram = new unsigned int[256 * 256]();

	for (int y = 0; y != height; y++) {

		for (int x = 0; x != width; x++) {

			histogram[ image1[x + width * y] + 256 * image2[x + width * y] ]++;
		}
	}

	return histogram;
}

int main(int argc, char* argv[]) {

	// Image loading
	int height1, width1, bitsPerPixel1, height2, width2, bitsPerPixel2;
	unsigned char* image1 = stbi_load("..//data//test.jpg", &height1, &width1, &bitsPerPixel1, 1);
	unsigned char* image2 = stbi_load("..//data//test2.jpg", &height2, &width2, &bitsPerPixel2, 1);

	cout << "Image 1 : width: " << width1 << ", height: " << height1 << endl;
	cout << "Image 2 : width: " << width2 << ", height: " << height2 << endl;

	// Histogram
	unsigned int* histogram1D = cpuHistogram1D(image1, width1, height1);
	unsigned int* histogram2D = cpuHistogram2D(image1, image2, width1, height1);

	cout << histogram2D[image1[40 + width1 * 0] + 256 * image2[40 + width2 * 0]] << endl;

	// Memory free
	stbi_image_free(image1);
	stbi_image_free(image2);
	delete histogram1D;
	delete histogram2D;

	// Prompt to end
	int end;
	cin >> end;

	return 0;
}