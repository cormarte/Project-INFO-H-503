#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
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

Image cpuApplyTransform(const Image& originalImage, const Transform& transform) {

	Image transformedImage = { originalImage.width, originalImage.height, new unsigned char[originalImage.width * originalImage.height]() };

	double centerX = originalImage.width / 2 - transform.tx;
	double centerY = originalImage.height / 2 - transform.ty;

	/* for (int y = 0; y != originalImage.height; y++) {

		for (int x = 0; x != originalImage.width; x++) {

			int newX = (int)((x + 0.5 - centerX)*cos(transform.rz * M_PI / 180.0) - (y + 0.5 - centerY)*sin(transform.rz * M_PI / 180.0) + transform.tx + centerX - 0.5);
			int newY = (int)((x + 0.5 - centerX)*sin(transform.rz * M_PI / 180.0) + (y + 0.5 - centerY)*cos(transform.rz * M_PI / 180.0) + transform.ty + centerY - 0.5);

			if (newX >= 0 && newX < transformedImage.width && newY >= 0  && newY < transformedImage.height) {

				transformedImage.pixels[newX + transformedImage.width * newY] = originalImage.pixels[x + originalImage.width * y];
			}
		}
	} */

	for (int y = 0; y != originalImage.height; y++) {

		for (int x = 0; x != originalImage.width; x++) {

			int originalX = (int)((x - centerX)*cos(-transform.rz * M_PI / 180.0) - (y - centerY)*sin(-transform.rz * M_PI / 180.0) - transform.tx + centerX);
			int originalY = (int)((x - centerX)*sin(-transform.rz * M_PI / 180.0) + (y - centerY)*cos(-transform.rz * M_PI / 180.0) - transform.ty + centerY);

			if (originalX >= 0 && originalX < originalImage.width && originalY >= 0 && originalY < originalImage.height) {

				transformedImage.pixels[x + transformedImage.width * y] = originalImage.pixels[originalX + originalImage.width * originalY];
			}
		}
	}

	return transformedImage;
}

Image cpuRegister(const Image& floatingImage, const Image& referenceImage) {
	
	vector<double> translationsX;
	vector<double> translationsY;
	vector<double> rotationsZ;

	for (int i = 0; i != 20; i++) {
	
		translationsX.push_back(i + 10);
		translationsY.push_back(i - 20);
		rotationsZ.push_back(i - 30);
	}

	Transform optimalTransform = {0, 0, 0, 0, 0, 0};
	double maxMutualInformation = 0;
	/*double progress = 0;
	double step = 100.0 / (translationsX.size()*translationsY.size()*rotationsZ.size());*/

	for (int a = 0; a != translationsX.size(); a++) {
		
		for (int b = 0; b != translationsY.size(); b++) {
		
			for (int c = 0; c != rotationsZ.size(); c++) {

				Transform transform = { translationsX[a], translationsY[b], 0, 0, 0, rotationsZ[c] };
				Image transformedImage = cpuApplyTransform(floatingImage, transform);
				HistogramType* histogram2D = cpuHistogram2D<HistogramType, histogramSize>(transformedImage, referenceImage);
				double mutualInformation = cpuMutualInformation<HistogramType, histogramSize>(histogram2D);

				if (mutualInformation > maxMutualInformation) {
				
					maxMutualInformation = mutualInformation;
					optimalTransform = transform;
				}

				delete histogram2D;
				delete transformedImage.pixels;

				/*progress += step;
				cout << progress << "%" << endl;*/
			}
		}
	}

	cout << "Optimal transform: Tx: " << optimalTransform.tx << ", Ty: " << optimalTransform.ty << ", Rz: " << optimalTransform.rz << endl;

	return cpuApplyTransform(floatingImage, optimalTransform);

}

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
		Image floatingImage = { fWidth, fHeight, fPixels };
		Image referenceImage = { fWidth, rHeight, rPixels };

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

	
		/*// Image transformation
		Transform transform = { -20, 10, 0, 0, 0, 20 };
		Image transformedImage = cpuApplyTransform(referenceImage, transform);
		cout << stbi_write_png("..//data//transformed.png", transformedImage.width, transformedImage.height, 1, transformedImage.pixels, transformedImage.width) << endl;

		// Histogram 2D
		HistogramType* histogram2D = cpuHistogram2D<HistogramType, histogramSize>(floatingImage, referenceImage);

		// Mutual information
		double mutualInformation = cpuMutualInformation<HistogramType, histogramSize>(histogram2D);

		cout << "Mutual information: " << mutualInformation << endl;

		// Memory free
		delete transformedImage.pixels;
		delete histogram2D;*/
	}

	// Prompt to end
	int end;
	cin >> end;

	return 0;
}