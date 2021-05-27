typedef struct _PixelValue
{
	float r;
	float g;
	float b;
} PixelValue;

__kernel void gauss(
	__global const PixelValue *imageData,
	__local PixelValue *localBuffer,
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
	
	size_t l_y = get_local_id(0);
	size_t l_x = get_local_id(1);

	size_t size_y = get_local_size(0);
	size_t size_x = get_local_size(1);
	//printf("(%d,%d),(%d,%d), (%d,%d)\n", y, x, l_y, l_x, size_y, size_x);
	
	bool isColumns = size_x > 1;
	size_t local_index = (isColumns) ? l_y : l_x;
	PixelValue newPixelValue;

	int imagePos = y * outputWidth + x;
	PixelValue oldPixelValue = imageData[imagePos];
	
	localBuffer[local_index] = oldPixelValue;
	mem_fence(CLK_LOCAL_MEM_FENCE);

	float r = 0, g = 0, b = 0;
	
	for (int h = -radius; h <= radius; h++) {
			int filterPos = (h + radius) * filterWidth;
			int p = (isColumns) ? y : x;
			int pixelPos = (p + h) * outputWidth;

			
			double kernelValue = filter[filterPos];			
			//PixelValue pixelValue = imageData[pixelPos];
			PixelValue pixelValue = localBuffer[pixelPos];
			//printf("(%d,%d,%d)\n", pixelValue.r, pixelValue.g, pixelValue.b);

			/*
			r += kernelValue * pixelValue.r;
			g += kernelValue * pixelValue.g;
			b += kernelValue * pixelValue.b;
			*/
			r = pixelValue.r;
			g = pixelValue.g;
			b = pixelValue.b;
	}

	newPixelValue.r = r;
	newPixelValue.g = g;
	newPixelValue.b = b;

	outputBuffer[imagePos] = newPixelValue;	
}

