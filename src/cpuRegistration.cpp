#define _USE_MATH_DEFINES

#include <amp_math.h>
#include <chrono>
#include <math.h>
#include <iostream>
#include <vector>

#include "image.h"
#include "powell.h"
#include "transform.h"

using namespace Concurrency::fast_math;
using namespace std;

// Defininitions
#define HISTOGRAMSIZE 256
#define PI 3.14159265358979f
typedef int HistogramType;

// Variables
unsigned int width;
unsigned int height;
Image floatingImage;
Image referenceImage;
Image transformedImage;
HistogramType* referenceHistogram;
HistogramType* transformedHistogram;
HistogramType* histogram2D;

void cpuApplyTransform(const Image& originalImage, Image& transformedImage, const Transform& transform) {

	float cosrz = cosf(-transform.rz * PI / 180.0f);
	float sinrz = sinf(-transform.rz * PI / 180.0f);

	float centerX = originalImage.width / 2 - transform.tx;
	float centerY = originalImage.height / 2 - transform.ty;

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


float cpuPowellFunction(float* transformVector) {

	/* Calculates the mutual information of the reference image
	and the transformed image for a transform vector 'transformVector' */

	Transform transform = { transformVector[0], transformVector[1], 0, 0, 0, transformVector[2] };

	//cout << transformVector[0] << ", " << transformVector[1] << ", " << transformVector[2] << endl;

	cpuApplyTransform(floatingImage, transformedImage, transform);

	memset(transformedHistogram, 0, HISTOGRAMSIZE * sizeof(HistogramType));
	cpuHistogram1D<HistogramType, HISTOGRAMSIZE>(transformedImage, transformedHistogram);

	memset(histogram2D, 0, HISTOGRAMSIZE * HISTOGRAMSIZE * sizeof(HistogramType));
	cpuHistogram2D<HistogramType, HISTOGRAMSIZE>(transformedImage, referenceImage, histogram2D);

	// Because Powell search is written for minimization of function, MI is multiplied by -1 to search max
	float mutualInformation = (-1.0f)*cpuMutualInformation<HistogramType, HISTOGRAMSIZE>(transformedHistogram, referenceHistogram, histogram2D, width, height);
	
	//cout << -1.0f*mutualInformation << endl;
	
	return mutualInformation;
}

Image cpuRegister(const Image& image1, const Image& image2) {

	// Initialization
	floatingImage = image1;
	referenceImage = image2;
	width = referenceImage.width;
	height = referenceImage.height;
	transformedImage = {width, height, new unsigned char[width * height]};
	referenceHistogram = new HistogramType[HISTOGRAMSIZE];
	transformedHistogram = new HistogramType[HISTOGRAMSIZE];
	histogram2D = new HistogramType[HISTOGRAMSIZE * HISTOGRAMSIZE];

	memset(referenceHistogram, 0, HISTOGRAMSIZE * sizeof(HistogramType));
	cpuHistogram1D<HistogramType, HISTOGRAMSIZE>(referenceImage, referenceHistogram);
	
	/*memset(transformedHistogram, 0, HISTOGRAMSIZE * sizeof(HistogramType));
	cpuHistogram1D<HistogramType, HISTOGRAMSIZE>(floatingImage, transformedHistogram);*/
	
	// Powell registration
	float* maxMutualInformation = new float(FLT_MAX);
	float transformVector[3] = { 0.0f, 0.0f, 0.0f };
	
	powell(maxMutualInformation, transformVector, 3, 2.0e-4f, cpuPowellFunction);
	
	// Final transform
	Transform transform = { transformVector[0], transformVector[1], 0, 0, 0, transformVector[2] };
	cpuApplyTransform(floatingImage, transformedImage, transform);

	cout << "CPU optimal transform: Tx: " << transform.tx << ", Ty: " << transform.ty << ", Rz: " << transform.rz << endl;
	cout << "CPU max mutual information: " << (-1.0f) * *maxMutualInformation << endl;

	return transformedImage;
}
