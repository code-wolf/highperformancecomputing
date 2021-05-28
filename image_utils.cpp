#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <stdlib.h>
#else
#include "CL/cl.h"
#endif

#include "image_utils.h"
#include <iostream>
#include <cmath>
#include "globals.h"


using namespace std;

tga::TGAImage loadImage(const char *path)
{
	tga::TGAImage image;
	tga::LoadTGA(&image, path);
	cout << "load image " << path << ". width=" << image.width << "; height=" << image.height << endl;
	cout << "image data: " << image.imageData.size() << endl;
	
	
	return image;
}

double* setupGaussFilterKernel()
{
	double gauss[smooth_kernel_size][smooth_kernel_size];
	double sum = 0;
	int i, j;

	for (i = 0; i < smooth_kernel_size; i++) {
		for (j = 0; j < smooth_kernel_size; j++) {
			double x = i - (smooth_kernel_size - 1) / 2.0;
			double y = j - (smooth_kernel_size - 1) / 2.0;
			gauss[i][j] = 1.0 / (2.0 * CL_M_PI * pow(sigma, 2.0)) * exp(-(pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2)));
			sum += gauss[i][j];
		}
	}

	for (i = 0; i < smooth_kernel_size; i++) {
		for (j = 0; j < smooth_kernel_size; j++) {
			gauss[i][j] /= sum;
		}
	}
	
	double gaussSeparated[smooth_kernel_size];

	for (i = 0; i < smooth_kernel_size; i++) {
		gaussSeparated[i] = sqrt(gauss[i][i]);
	}

	printf("1D Separated Gaussian filter kernel:\n");
	for (i = 0; i < smooth_kernel_size; i++) {
		printf("%f, ", gaussSeparated[i]);
	}
	printf("\n");

	return gaussSeparated;
}

void convertPixelsToImage(PixelValue **pixels, tga::TGAImage &image)
{
	vector<unsigned char> outData;
	for(unsigned int y = 0; y < image.height; y++) {
		for(unsigned int x = 0; x < image.width; x++) {
			PixelValue pixel = pixels[y][x];

			outData.push_back(pixel.r * 255);
			outData.push_back(pixel.g * 255);
			outData.push_back(pixel.b * 255);
		}
	}
	image.imageData = outData;
}

PixelValue **convertImageToPixels(tga::TGAImage image)
{
	int w = image.width;
	int h = image.height;
	
	PixelValue **pixels;
	pixels = new PixelValue* [h];
	for(int i = 0; i < h; i++) {
		pixels[i] = new PixelValue [w];
	}
	
	int pos = 0;
	for(int y = 0; y < h; y++) {
		for(int x = 0; x < w; x++) {
			PixelValue pixelValue;
			pixelValue.r = image.imageData[pos] / 255.f;
			pixelValue.g = image.imageData[pos + 1] / 255.f;
			pixelValue.b = image.imageData[pos + 2] / 255.f;
			pixels[y][x] = pixelValue;

			pos += 3;
		}
	}

	return pixels;
}
void printPixel(PixelValue value) {
	cout << "(" << value.r << "," << value.g << "," << value.b << ")" << endl;
}