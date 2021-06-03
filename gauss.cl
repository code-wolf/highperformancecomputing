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

	size_t y = get_global_id(0);
	size_t x = get_global_id(1);
	
	size_t l_y = get_local_id(0);
	size_t l_x = get_local_id(1);

	size_t size_x = get_local_size(1);
	
	bool isColumns = size_x > 1;
	size_t local_index = (isColumns) ? l_x : l_y;
	size_t image_size = (isColumns) ? outputWidth : outputHeight;
	
	int imagePos = y * outputWidth + x;
	PixelValue oldPixelValue = imageData[imagePos];
	
	localBuffer[local_index] = oldPixelValue;
	mem_fence(CLK_LOCAL_MEM_FENCE);

	float r = 0, g = 0, b = 0;

	for (int h = -radius; h <= radius; h++) {
		int filterPos = clamp((h + radius), 0, 2 * radius); 
		double kernelValue = filter[filterPos];
			
		int buffer_index = clamp((int)(local_index + h), 0, (int)image_size);

		PixelValue pixelValue = localBuffer[buffer_index];
			
		r += kernelValue * pixelValue.r;
		g += kernelValue * pixelValue.g;
		b += kernelValue * pixelValue.b;
	}

	PixelValue newPixelValue;
	newPixelValue.r = r;
	newPixelValue.g = g;
	newPixelValue.b = b;

	outputBuffer[imagePos] = newPixelValue;	
}

