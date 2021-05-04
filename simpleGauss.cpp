#import "image_utils.h"

PixelValue** apply(double **filter, int radius, PixelValue **pixels, int imageWidth, int imageHeight)
{
	int i, j, h, w, pos;
	int filterHeight = 2 * radius + 1;
	int filterWidth = 2 * radius + 1;
	int newImageHeight = imageHeight;
    int newImageWidth = imageWidth;
	
	PixelValue **outPixels;
	outPixels = new PixelValue*[imageHeight];
	for(int i = 0; i < imageHeight; i++) {
		outPixels[i] = new PixelValue[imageWidth];
	}
	for (int y = radius ; y < imageHeight - radius; y++) {
		for (int x = radius ; x < imageWidth - radius; x++) {
			PixelValue newPixelValue;
			PixelValue oldPixelValue = pixels[y][x];

			double r = 0, g = 0, b = 0;

			for (h = -radius; h <= radius; h++) {
				for (w = -radius ; w <= radius ; w++) {
					double kernelValue = filter[h + radius][w + radius];
					PixelValue pixelValue = pixels[y + h][x + w];
					
					r += kernelValue * pixelValue.r;
					g += kernelValue * pixelValue.g;
					b += kernelValue * pixelValue.b;
				}
			}
			
			newPixelValue.r = r;
			newPixelValue.g = g;
			newPixelValue.b = b;

			printf("(%d,%d): (%f,%f,%f)\n", y, x, oldPixelValue.r, oldPixelValue.g, oldPixelValue.b);

			outPixels[y][x] = newPixelValue;
		}
	}

	return outPixels;
}

void simpleGauss(int radius)
{
	tga::TGAImage image = loadImage("lena_small.tga");
	PixelValue **pixels = convertImageToPixels(image);

	double **kernel = setupGaussKernel(radius);
	

	PixelValue **filteredPixels;
	filteredPixels = apply(kernel, radius, pixels, image.width, image.height);

	tga::TGAImage outImage;
	outImage.height = image.height;
	outImage.width = image.width;
	outImage.bpp = image.bpp;
	outImage.type = image.type;

	convertPixelsToImage(filteredPixels, outImage);

	tga::saveTGA(outImage, ("lena_small_out_" + std::to_string(radius) + ".tga").c_str());
}