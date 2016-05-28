#define STB_IMAGE_IMPLEMENTATION

#include <iostream>
#include <string>
#include <stb_image.h>

#include "gpu.h"

using namespace std;

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
double cpuMutualInformation2(const HistogramType* histogramF, const HistogramType* histogramR, const HistogramType* histogramFR, const Image& imageF, const Image& imageR){

	double nbPixelsF = imageF.width*imageF.height;
	double nbPixelsR = imageR.width*imageR.height;

	double mutualInformation = 0;

	for (int binF = 0; binF != histogramSize; binF++) {

		for (int binR = 0; binR != histogramSize; binR++) {

			double pF = histogramF[binF] / nbPixelsF;
			double pR = histogramR[binR] / nbPixelsR;
			double pFR = histogramFR[binF + histogramSize * binR] / nbPixelsF;

			if (pFR != 0) {

				mutualInformation += pFR*log2(pFR / (pF * pR));
			}
		}
	}

	return mutualInformation;
}

template<typename HistogramType, int histogramSize>
double cpuMutualInformation(const HistogramType* histogramFR){

	double histogramSum = 0;
	HistogramType histogramF[histogramSize] = {};
	HistogramType histogramR[histogramSize] = {};

	for (int binF = 0; binF != histogramSize; binF++) {

		for (int binR = 0; binR != histogramSize; binR++) {
			
			histogramSum += histogramFR[binF + histogramSize * binR];
			histogramF[binF] += histogramFR[binF + histogramSize * binR];
			histogramR[binR] += histogramFR[binF + histogramSize * binR];
		}
	}

	double mutualInformation = 0;

	for (int binF = 0; binF != histogramSize; binF++) {

		for (int binR = 0; binR != histogramSize; binR++) {

			double pF = histogramF[binF] / histogramSum;
			double pR = histogramR[binR] / histogramSum;
			double pFR = histogramFR[binF + histogramSize * binR] / histogramSum;

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
	unsigned char* pixelsF = stbi_load("..//data//imageF.jpg", &heightF, &widthF, &bitsPerPixelF, 1);
	unsigned char* pixelsR = stbi_load("..//data//imageR.jpg", &heightR, &widthR, &bitsPerPixelR, 1);

	if (heightF != heightR || widthF != widthR) {
	
		cout << "Error: both images must have the same dimensions!" << endl;
	}

	else {

		// Images definition
		Image imageF = { widthF, heightF, pixelsF };
		Image imageR = { widthF, heightR, pixelsR };

		// Types definition
		typedef double HistogramType;
		const int histogramSize = 256;

		// Histogram 2D
		HistogramType* histogramFR = cpuHistogram2D<HistogramType, histogramSize>(imageF, imageR);

		// Mutual information
		double mutualInformation = cpuMutualInformation<HistogramType, histogramSize>(histogramFR);

		cout << "Mutual information: " << mutualInformation << endl;

		gpuRegistration(imageF, imageR);

		// Memory free
		stbi_image_free(imageF.pixels);
		stbi_image_free(imageR.pixels);
		delete histogramFR;
	}

	// Prompt to end
	int end;
	cin >> end;

	return 0;
}