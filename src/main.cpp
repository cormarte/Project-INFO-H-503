#define STB_IMAGE_IMPLEMENTATION

#include <iostream>
#include <string>
#include <stb_image.h>

using namespace std;

struct Image {

	int width;
	int height;
	unsigned char* pixels;
};

template<typename HistogramType, int histogramSize>
HistogramType* cpuHistogram1D(const Image& image) {

	int height = image.height;
	int width = image.width;

	HistogramType* histogram = new HistogramType[histogramSize]();

	for (int y = 0; y != height; y++) {
	
		for (int x = 0; x != width; x++) {

			histogram[ image.pixels[x + width*y] ] += 1;
		}
	}

	return histogram;
}

template<typename HistogramType, int histogramSize>
HistogramType* cpuHistogram2D(const Image& imageF, const Image& imageR) {

	int height = imageF.height;
	int width = imageF.width;

	HistogramType* histogram = new HistogramType[histogramSize * histogramSize]();

	for (int y = 0; y != height; y++) {

		for (int x = 0; x != width; x++) {

			histogram[ imageF.pixels[x + width * y] + histogramSize * imageR.pixels[x + width * y] ] += 1;
		}
	}

	return histogram;
}

template<typename HistogramType, int histogramSize>
double cpuMutualInformation(const HistogramType* histogramF, const HistogramType* histogramR, const HistogramType* histogramFR, const Image& imageF, const Image& imageR){

	double nbPixelsF = imageF.width*imageF.height;
	double nbPixelsR = imageR.width*imageR.height;
	double nbPixelsFR = nbPixelsF + nbPixelsR;
	double mutualInformation = 0;

	for (int binF = 0; binF != histogramSize; binF++) {
	
		for (int binR = 0; binR != histogramSize; binR++) {

			double pF = histogramF[binF] / nbPixelsF;
			double pR = histogramR[binR] / nbPixelsR;
			double pFR = histogramFR[binF + histogramSize * binR] / nbPixelsFR;

			if (pFR != 0) {

				mutualInformation += pFR*log2(pFR / (pF * pR));
			}
		}
	}

	return mutualInformation;
}

int main(int argc, char* argv[]) {

	// Pixels loading
	int heightF, widthF, bitsPerPixelF, heightR, widthR, bitsPerPixelR;
	unsigned char* pixelsF = stbi_load("..//data//testF.jpg", &heightF, &widthF, &bitsPerPixelF, 1);
	unsigned char* pixelsR = stbi_load("..//data//testR.jpg", &heightR, &widthR, &bitsPerPixelR, 1);

	cout << "Image F : width: " << widthF << ", height: " << heightF << endl;
	cout << "Image R : width: " << widthR << ", height: " << heightR << endl;

	// Images definition
	Image imageF = { widthF, heightF, pixelsF };
	Image imageR = { widthF, heightR, pixelsR };

	// Histogram
	unsigned int* histogramF = cpuHistogram1D<unsigned int, 256>(imageF);
	unsigned int* histogramR = cpuHistogram1D<unsigned int, 256>(imageR);
	unsigned int* histogramFR = cpuHistogram2D<unsigned int, 256>(imageF, imageR);

	cout << "HistogramFR[][]: " << histogramFR[0 + 256 * 247] << endl;

	// Mutual information
	double mutualInformation = cpuMutualInformation<unsigned int, 256>(histogramF, histogramR, histogramFR, imageF, imageR);

	cout << "Mutual information: " << mutualInformation << endl;

	// Memory free
	stbi_image_free(imageF.pixels);
	stbi_image_free(imageR.pixels);
	delete histogramF;
	delete histogramR;
	delete histogramFR;

	// Prompt to end
	int end;
	cin >> end;

	return 0;
}