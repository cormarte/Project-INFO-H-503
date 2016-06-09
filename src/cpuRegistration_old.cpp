#define _USE_MATH_DEFINES

#include <chrono>
#include <math.h>
#include <iostream>
#include <string>
#include <stb_image.h>
#include <stb_image_write.h>
#include <vector>

#include "gpuRegistration.h"
#include "transform.h"

using namespace std;

// Types definition
typedef int HistogramType;
const int histogramSize = 256;


Image cpuApplyTransform(const Image& originalImage, const Transform& transform) {

	Image transformedImage = { originalImage.width, originalImage.height, new unsigned char[originalImage.width * originalImage.height]() };

	int centerX = originalImage.width / 2 - transform.tx;
	int centerY = originalImage.height / 2 - transform.ty;

	for (int y = 0; y != originalImage.height; y++) {

		for (int x = 0; x != originalImage.width; x++) {

			int originalX = (int)((x - centerX)*cos(-transform.rz * M_PI / 180.0) - (y - centerY)*sin(-transform.rz * M_PI / 180.0) - transform.tx + centerX);
			int originalY = (int)((x - centerX)*sin(-transform.rz * M_PI / 180.0) + (y - centerY)*cos(-transform.rz * M_PI / 180.0) - transform.ty + centerY);

			if (originalX >= 0 && (unsigned int)originalX < originalImage.width && originalY >= 0 && (unsigned int)originalY < originalImage.height) {

				transformedImage.pixels[x + transformedImage.width * y] = originalImage.pixels[originalX + originalImage.width * originalY];
			}
		}
	}

	return transformedImage;
}


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
HistogramType* cpuHistogram2D(const Image& image1, const Image& image2) {

	int height = image1.height;
	int width = image1.width;

	HistogramType* histogram = new HistogramType[histogramSize * histogramSize]();

	for (int y = 0; y != height; y++) {

		for (int x = 0; x != width; x++) {

			histogram[image1.pixels[x + width * y] + histogramSize * image2.pixels[x + width * y]] += 1;
		}
	}

	return histogram;
}


template<typename HistogramType, int histogramSize>
double cpuMutualInformation(const HistogramType* histogram1, const HistogramType* histogram2, const HistogramType* histogram2D, const Image& image1, const Image& image2){

	double nbPixels1 = image1.width*image1.height;
	double nbPixels2 = image2.width*image2.height;

	double mutualInformation = 0;

	for (int bin1 = 0; bin1 != histogramSize; bin1++) {

		for (int bin2 = 0; bin2 != histogramSize; bin2++) {

			double p1 = histogram1[bin1] / nbfPixels;
			double p2 = histogram2[bin2] / nbrPixels;
			double p12 = histogram2D[bin1 + histogramSize * bin2] / nbfPixels;

			if (p12 != 0) {

				mutualInformation += p12*log2(p12 / (p1 * p2));
			}
		}
	}

	return mutualInformation;
}


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


Image cpuRegister(const Image& floatingImage, const Image& referenceImage) {
	
	vector<int> translationsX;
	vector<int> translationsY;
	vector<float> rotationsZ;

	for (int i = 0; i != 20; i++) {
	
		translationsX.push_back(i + 10);
		translationsY.push_back(i - 20);
		rotationsZ.push_back(i - 30);
	}

	Transform optimalTransform = {0, 0, 0, 0, 0, 0};
	double maxMutualInformation = 0;
	//double progress = 0;
	//double step = 100.0 / (translationsX.size()*translationsY.size()*rotationsZ.size());

	for (int a = 0; a != translationsX.size(); a++) {
		
		for (int b = 0; b != translationsY.size(); b++) {
		
			for (int c = 0; c != rotationsZ.size(); c++) {

				Transform transform = { translationsX[a], translationsY[b], 0, 0, 0, rotationsZ[c] };
				Image transformedImage = cpuApplyTransform(floatingImage, transform);
				HistogramType* histogram2D = cpuHistogram2D<HistogramType, histogramSize>(transformedImage, referenceImage);
				double mutualInformation = cpuMutualInformation<HistogramType, histogramSize>(histogram2D);

				//cout << "CPU mutual information: " << mutualInformation << endl;

				if (mutualInformation > maxMutualInformation) {
				
					maxMutualInformation = mutualInformation;
					optimalTransform = transform;
				}

				delete histogram2D;
				delete transformedImage.pixels;

				//progress += step;
				//cout << progress << "%" << endl;
			}
		}
	}

	cout << "Optimal transform: Tx: " << optimalTransform.tx << ", Ty: " << optimalTransform.ty << ", Rz: " << optimalTransform.rz << endl;

	return cpuApplyTransform(floatingImage, optimalTransform);
}