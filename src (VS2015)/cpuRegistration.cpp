#define _USE_MATH_DEFINES

#include <amp_math.h>
#include <chrono>
#include <math.h>
#include <iostream>
#include <vector>

#include "image.h"
#include "transform.h"

using namespace Concurrency::fast_math;
using namespace std;

// Defininitions
#define HISTOGRAMSIZE 256
#define PI 3.14159265358979f
typedef int HistogramType;


void cpuApplyTransform(const Image& originalImage, Image& transformedImage, const Transform& transform) {

	float cosrz = cosf(-transform.rz * PI / 180.0f);
	float sinrz = sinf(-transform.rz * PI / 180.0f);

	int centerX = originalImage.width / 2 - transform.tx;
	int centerY = originalImage.height / 2 - transform.ty;

	for (int y = 0; y != originalImage.height; y++) {

		for (int x = 0; x != originalImage.width; x++) {

			int originalX = (int)((x - centerX)*cosrz - (y - centerY)*sinrz - transform.tx + centerX);
			int originalY = (int)((x - centerX)*sinrz + (y - centerY)*cosrz - transform.ty + centerY);

			if (originalX >= 0 && (unsigned int)originalX < originalImage.width && originalY >= 0 && (unsigned int)originalY < originalImage.height) {

				transformedImage.pixels[x + transformedImage.width * y] = originalImage.pixels[originalX + originalImage.width * originalY];
			}

			else {

				transformedImage.pixels[x + transformedImage.width * y] = 255;
			}
		}
	}
}


template<typename HistogramType, int histogramSize>
void cpuHistogram1D(const Image& image, HistogramType* histogram) {

	int height = image.height;
	int width = image.width;

	for (int y = 0; y != height; y++) {

		for (int x = 0; x != width; x++) {

			histogram[image.pixels[x + width*y]] += 1;
		}
	}
}


template<typename HistogramType, int histogramSize>
void cpuHistogram2D(const Image& image1, const Image& image2, HistogramType* histogram) {

	int height = image1.height;
	int width = image1.width;

	for (int y = 0; y != height; y++) {

		for (int x = 0; x != width; x++) {

			histogram[image1.pixels[x + width * y] + histogramSize * image2.pixels[x + width * y]] += 1;
		}
	}
}


template<typename HistogramType, int histogramSize>
float cpuMutualInformation(const HistogramType* histogram1, const HistogramType* histogram2, HistogramType* histogram2D, const unsigned int width, const unsigned int height){

	unsigned int histogramSum = width*height;

	float mutualInformation = 0;

	for (int bin1 = 0; bin1 != histogramSize; bin1++) {

		for (int bin2 = 0; bin2 != histogramSize; bin2++) {

			float p1 = (1.0f * histogram1[bin1]) / histogramSum;
			float p2 = (1.0f * histogram2[bin2]) / histogramSum;
			float p12 = (1.0f * histogram2D[bin1 + histogramSize * bin2]) / histogramSum;

			if (p12 != 0) {

				mutualInformation += p12*log2f(p12 / (p1 * p2));
			}
		}
	}

	return mutualInformation;
}


Image cpuRegister(const Image& floatingImage, const Image& referenceImage) {

	// Initialization
	unsigned int width = referenceImage.width;
	unsigned int height = referenceImage.height;
	Image transformedImage = {width, height, new unsigned char[width * height]};
	HistogramType* referenceHistogram = new HistogramType[HISTOGRAMSIZE];
	HistogramType* transformedHistogram = new HistogramType[HISTOGRAMSIZE];
	HistogramType* histogram2D = new HistogramType[HISTOGRAMSIZE * HISTOGRAMSIZE];

	// Registration
	vector<int> translationsX;
	vector<int> translationsY;
	vector<float> rotationsZ;

	for (int i = 0; i != 1; i++) {

		translationsX.push_back(i + 10);
		translationsY.push_back(i - 20);
		rotationsZ.push_back(i - 30);
	}

	Transform optimalTransform = {0, 0, 0, 0, 0, 0};
	float maxMutualInformation = 0;

	memset(referenceHistogram, 0, HISTOGRAMSIZE * sizeof(HistogramType));
	auto begin = chrono::high_resolution_clock::now();
	cpuHistogram1D<HistogramType, HISTOGRAMSIZE>(referenceImage, referenceHistogram);
	auto end = chrono::high_resolution_clock::now();
	cout << "cpuHistogram1D<HistogramType, HISTOGRAMSIZE>(referenceImage, referenceHistogram): " << chrono::duration_cast<chrono::nanoseconds>(end - begin).count() << "ns" << std::endl;
	
	memset(transformedHistogram, 0, HISTOGRAMSIZE * sizeof(HistogramType));
	begin = chrono::high_resolution_clock::now();
	cpuHistogram1D<HistogramType, HISTOGRAMSIZE>(floatingImage, transformedHistogram);
	end = chrono::high_resolution_clock::now();
	cout << "cpuHistogram1D<HistogramType, HISTOGRAMSIZE>(floatingImage, transformedHistogram): " << chrono::duration_cast<chrono::nanoseconds>(end - begin).count() << "ns" << std::endl;

	for (int a = 0; a != translationsX.size(); a++) {

		for (int b = 0; b != translationsY.size(); b++) {

			for (int c = 0; c != rotationsZ.size(); c++) {

				Transform transform = {translationsX[a], translationsY[b], 0, 0, 0, rotationsZ[c]};
				begin = chrono::high_resolution_clock::now();
				cpuApplyTransform(floatingImage, transformedImage, transform);
				end = chrono::high_resolution_clock::now();
				cout << "cpuApplyTransform(floatingImage, transformedImage, transform): " << chrono::duration_cast<chrono::nanoseconds>(end - begin).count() << "ns" << std::endl;

				memset(histogram2D, 0, HISTOGRAMSIZE * HISTOGRAMSIZE * sizeof(HistogramType));
				begin = chrono::high_resolution_clock::now();
				cpuHistogram2D<HistogramType, HISTOGRAMSIZE>(transformedImage, referenceImage, histogram2D);
				end = chrono::high_resolution_clock::now();
				cout << "cpuHistogram2D<HistogramType, HISTOGRAMSIZE>(transformedImage, referenceImage, histogram2D): " << chrono::duration_cast<chrono::nanoseconds>(end - begin).count() << "ns" << std::endl;
				
				begin = chrono::high_resolution_clock::now();
				float mutualInformation = cpuMutualInformation<HistogramType, HISTOGRAMSIZE>(transformedHistogram, referenceHistogram, histogram2D, width, height);
				end = chrono::high_resolution_clock::now();
				cout << "cpuMutualInformation<HistogramType, HISTOGRAMSIZE>(transformedHistogram, referenceHistogram, histogram2D, width, height): " << chrono::duration_cast<chrono::nanoseconds>(end - begin).count() << "ns" << std::endl;

				//cout << "CPU mutual information: " << mutualInformation << endl;

				if (mutualInformation > maxMutualInformation) {

					maxMutualInformation = mutualInformation;
					optimalTransform = transform;
				}
			}
		}
	}

	cout << "Optimal transform: Tx: " << optimalTransform.tx << ", Ty: " << optimalTransform.ty << ", Rz: " << optimalTransform.rz << endl;

	begin = chrono::high_resolution_clock::now();
	cpuApplyTransform(floatingImage, transformedImage, optimalTransform);
	end = chrono::high_resolution_clock::now();
	cout << "cpuApplyTransform(floatingImage, transformedImage, optimalTransform): " << chrono::duration_cast<chrono::nanoseconds>(end - begin).count() << "ns" << std::endl;


	return transformedImage;
}
