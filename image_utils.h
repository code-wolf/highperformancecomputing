#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include "tga.h"

typedef struct PixelValue {
	float r;
	float g;
	float b;
} _PixelValue;


tga::TGAImage loadImage(const char *path);

/**
 * Sets up the gauss kernel (filter) for a given radius
 * @return: an array containing the filter values
 **/
double* setupGaussFilterKernel(double *gauss);

/**
 * Converts a two-dimensional image array (y,x) to a TGA image
 * @param pixels: The two-dimensional image pixel values
 * @param image: Out-parameter containing the image
 **/
void convertPixelsToImage(PixelValue **pixels, tga::TGAImage &image);

/**
 * Converts a TGA image to a two-dimensional array (y,x)
 * @param: the image to be converted
 * @return: The image as two-dimensional Pixel array
 **/
PixelValue **convertImageToPixels(tga::TGAImage image);

/**
 * Convenience Method to print the r,g,b values of a pixel
 **/
void printPixel(PixelValue value);

#endif