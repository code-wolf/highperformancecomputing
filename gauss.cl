typedef struct _PixelValue
{
	float r;
	float g;
	float b;
} PixelValue;

__kernel void gauss(
	__global const PixelValue *imageData,
	__global const double *filter,
	__global PixelValue *outputBuffer,
	__private int radius
)
{	
	size_t outputHeight = get_global_size(0);
	size_t outputWidth = get_global_size(1);
	size_t filterWidth = 2 * radius + 1;
	size_t y = get_global_id(0);
	size_t x = get_global_id(1);
	
	PixelValue newPixelValue;

	int imagePos = y * outputWidth + x;
	PixelValue oldPixelValue = imageData[imagePos];
	float r = 0, g = 0, b = 0;
	
	for (int h = -radius; h <= radius; h++) {
		for (int w = -radius ; w <= radius ; w++) {
			int filterPos = (h + radius) * filterWidth + (w + radius);
			int pixelPos = (y + h) * outputWidth + (x + w);
			
			double kernelValue = filter[filterPos];			
			PixelValue pixelValue = imageData[pixelPos];

			r += kernelValue * pixelValue.r;
			g += kernelValue * pixelValue.g;
			b += kernelValue * pixelValue.b;
		}
	}

	newPixelValue.r = r;
	newPixelValue.g = g;
	newPixelValue.b = b;

	outputBuffer[imagePos] = newPixelValue;	
}

