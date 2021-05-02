typedef struct _PixelValue
{
	float r;
	float g;
	float b;
} PixelValue;

__kernel void gauss(
	__global const PixelValue *imageData,
	__global const double *filter,
	__global PixelValue *outputBuffer
)
{	
	size_t outputHeight = get_global_size(0);
	size_t outputWidth = get_global_size(1);
	int radius = 11;
	size_t filterWidth = 2 * radius + 1;
	size_t y = get_global_id(0);
	size_t x = get_global_id(1);


	/*
	float r = imageData[y * outputWidth + x].r;
	float g = imageData[y * outputWidth + x].g;
	float b = imageData[y * outputWidth + x].b;
	*/
	
	PixelValue newPixelValue;

	if(y >= radius && y <= outputHeight - radius &&
		x >= radius && x <= outputWidth - radius) {
		
		int imagePos = y * outputWidth + x;
		PixelValue oldPixelValue = imageData[imagePos];
		float r = 0, g = 0, b = 0;
		
		for (int h = -radius; h <= radius; h++) {
			for (int w = -radius ; w <= radius ; w++) {
				int _y = h + radius;
				int _x = w + radius;
				
				int filterPos = _y * filterWidth + _x;
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
	/*
	PixelValue oldPixelValue = imageData[y * outputWidth + x];

	float r = 0, g = 0, b = 0;
	
	if(y < outputHeight - radius && x < outputWidth - radius) {
		for (int h = -radius; h <= radius; h++) {
			for (int w = -radius ; w <= radius ; w++) {
				//PixelValue pixelValue = imageData[_y * outputWidth + _x];
			}
		}
	}

	newPixelValue.r = r;
	newPixelValue.g = g;
	newPixelValue.b = b;
	*/
	//outputBuffer[y * outputWidth + x] = newPixelValue;
	
    //printf("|%d|%d|: (%f,%f,%f)\n", x, y, r, g, b);
}

