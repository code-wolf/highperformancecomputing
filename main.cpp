#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <stdlib.h>
#else
#include "CL/cl.h"
#endif

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <assert.h>
#include <cmath>

#include "image_utils.h"
#include "globals.h"

using namespace std;


std::string cl_errorstring(cl_int err)
{
	switch (err)
	{
	case CL_SUCCESS:									return std::string("Success");
	case CL_DEVICE_NOT_FOUND:							return std::string("Device not found");
	case CL_DEVICE_NOT_AVAILABLE:						return std::string("Device not available");
	case CL_COMPILER_NOT_AVAILABLE:						return std::string("Compiler not available");
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:				return std::string("Memory object allocation failure");
	case CL_OUT_OF_RESOURCES:							return std::string("Out of resources");
	case CL_OUT_OF_HOST_MEMORY:							return std::string("Out of host memory");
	case CL_PROFILING_INFO_NOT_AVAILABLE:				return std::string("Profiling information not available");
	case CL_MEM_COPY_OVERLAP:							return std::string("Memory copy overlap");
	case CL_IMAGE_FORMAT_MISMATCH:						return std::string("Image format mismatch");
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:					return std::string("Image format not supported");
	case CL_BUILD_PROGRAM_FAILURE:						return std::string("Program build failure");
	case CL_MAP_FAILURE:								return std::string("Map failure");
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:				return std::string("Misaligned sub buffer offset");
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:	return std::string("Exec status error for events in wait list");
	case CL_INVALID_VALUE:                    			return std::string("Invalid value");
	case CL_INVALID_DEVICE_TYPE:              			return std::string("Invalid device type");
	case CL_INVALID_PLATFORM:                 			return std::string("Invalid platform");
	case CL_INVALID_DEVICE:                   			return std::string("Invalid device");
	case CL_INVALID_CONTEXT:                  			return std::string("Invalid context");
	case CL_INVALID_QUEUE_PROPERTIES:         			return std::string("Invalid queue properties");
	case CL_INVALID_COMMAND_QUEUE:            			return std::string("Invalid command queue");
	case CL_INVALID_HOST_PTR:                 			return std::string("Invalid host pointer");
	case CL_INVALID_MEM_OBJECT:               			return std::string("Invalid memory object");
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  			return std::string("Invalid image format descriptor");
	case CL_INVALID_IMAGE_SIZE:               			return std::string("Invalid image size");
	case CL_INVALID_SAMPLER:                  			return std::string("Invalid sampler");
	case CL_INVALID_BINARY:                   			return std::string("Invalid binary");
	case CL_INVALID_BUILD_OPTIONS:            			return std::string("Invalid build options");
	case CL_INVALID_PROGRAM:                  			return std::string("Invalid program");
	case CL_INVALID_PROGRAM_EXECUTABLE:       			return std::string("Invalid program executable");
	case CL_INVALID_KERNEL_NAME:              			return std::string("Invalid kernel name");
	case CL_INVALID_KERNEL_DEFINITION:        			return std::string("Invalid kernel definition");
	case CL_INVALID_KERNEL:                   			return std::string("Invalid kernel");
	case CL_INVALID_ARG_INDEX:                			return std::string("Invalid argument index");
	case CL_INVALID_ARG_VALUE:                			return std::string("Invalid argument value");
	case CL_INVALID_ARG_SIZE:                 			return std::string("Invalid argument size");
	case CL_INVALID_KERNEL_ARGS:             			return std::string("Invalid kernel arguments");
	case CL_INVALID_WORK_DIMENSION:          			return std::string("Invalid work dimension");
	case CL_INVALID_WORK_GROUP_SIZE:          			return std::string("Invalid work group size");
	case CL_INVALID_WORK_ITEM_SIZE:           			return std::string("Invalid work item size");
	case CL_INVALID_GLOBAL_OFFSET:            			return std::string("Invalid global offset");
	case CL_INVALID_EVENT_WAIT_LIST:          			return std::string("Invalid event wait list");
	case CL_INVALID_EVENT:                    			return std::string("Invalid event");
	case CL_INVALID_OPERATION:                			return std::string("Invalid operation");
	case CL_INVALID_GL_OBJECT:                			return std::string("Invalid OpenGL object");
	case CL_INVALID_BUFFER_SIZE:              			return std::string("Invalid buffer size");
	case CL_INVALID_MIP_LEVEL:                			return std::string("Invalid mip-map level");
	case CL_INVALID_GLOBAL_WORK_SIZE:         			return std::string("Invalid gloal work size");
	case CL_INVALID_PROPERTY:                 			return std::string("Invalid property");
	default:                                  			return std::string("Unknown error code");
	}
}

void checkStatus(cl_int err) 
{
	if (err != CL_SUCCESS) {
		printf("OpenCL Error: %s \n", cl_errorstring(err).c_str());
		exit(EXIT_FAILURE);
	}
}

void printCompilerError(cl_program program, cl_device_id device)
{
	cl_int status;
	size_t logSize;
	char *log;

	// get log size
	status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
	checkStatus(status);

	// allocate space for log
	log = new char[logSize];
	if (!log)
	{
		exit(EXIT_FAILURE);
	}

	// read the log
	status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
	checkStatus(status);

	// print the log
	printf("Build Error: %s\n", log);
}

void printVector(int32_t* vector, unsigned int elementSize, const char* label)
{
	printf("%s:\n", label);

	for (unsigned int i = 0; i < elementSize; ++i)
	{
		printf("%d ", vector[i]);
	}
	
	printf("\n");
}


void printPixels(PixelValue **pixels, int width, int height, const char *filename)
{
	ofstream file1;
	file1.open(filename);
	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			PixelValue value = pixels[y][x];
			file1 << "(" << value.r << "," << value.g << "," << value.b << ")" << endl;
		}
	}
	file1.close();
}

PixelValue** applyOnGPU(double * filterVector,
						PixelValue **pixels,
						size_t imageWidth,
						size_t imageHeight,
						cl_context context,
						cl_command_queue commandQueue,
						cl_kernel kernel)
{
	int pos; 
	int filterHeight = 2 * smooth_kernel_size + 1;
	int filterWidth = 2 * smooth_kernel_size + 1;
	cl_int status;

	size_t num_elements = (imageHeight * imageWidth);
	size_t vector_size = (num_elements * sizeof(PixelValue));
	size_t filter_size = (filterHeight * filterWidth) * sizeof(double);
	size_t row_size = (imageWidth * sizeof(PixelValue));
	size_t column_size = (imageHeight * sizeof(PixelValue));

	size_t globalWorkSize[2] = {imageHeight, imageWidth};
	size_t localWorkSizeHeight[2] = {imageHeight, 1};
	size_t localWorkSizeWidth[2] = {1, imageWidth};

	// allocate memory and convert 2D array to 1D vector for the actual image data
	PixelValue *pixelVector = new PixelValue[vector_size]();
	pos = 0;
	for(size_t y = 0; y < imageHeight; y++) {
		for(size_t x = 0; x < imageWidth; x++) {
			pixelVector[pos] = pixels[y][x];
			pos ++;
		}
	}

	// memory for the resulting image
	PixelValue* outPixels = new PixelValue[vector_size];

	// create OpenCL Buffers
	cl_mem pixelBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, vector_size, NULL, &status);
	checkStatus(status);
	cl_mem filterBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, filter_size, NULL, &status);
	checkStatus(status);
	cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, vector_size, NULL, &status);
	checkStatus(status);
	
	// PROCESS ROWS
	// send memory to device
	checkStatus(clEnqueueWriteBuffer(commandQueue, pixelBuffer, CL_TRUE, 0, vector_size, pixelVector, 0, NULL, NULL));
	checkStatus(clEnqueueWriteBuffer(commandQueue, filterBuffer, CL_TRUE, 0, filter_size, filterVector, 0, NULL, NULL));
	
	// set kernel arguments
	checkStatus(clSetKernelArg(kernel, 0, sizeof(cl_mem), &pixelBuffer));
	checkStatus(clSetKernelArg(kernel, 1, column_size, NULL));
	checkStatus(clSetKernelArg(kernel, 2, sizeof(cl_mem), &filterBuffer));
	checkStatus(clSetKernelArg(kernel, 3, sizeof(cl_mem), &outputBuffer));
	checkStatus(clSetKernelArg(kernel, 4, sizeof(int), &smooth_kernel_size));
	checkStatus(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSizeHeight, 0, NULL, NULL));
	checkStatus(clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, vector_size, outPixels, 0, NULL, NULL));

	// PROCESS COLUMN
	// send memory to device
	/*
	checkStatus(clEnqueueWriteBuffer(commandQueue, pixelBuffer, CL_TRUE, 0, vector_size, pixelVector, 0, NULL, NULL));
	checkStatus(clEnqueueWriteBuffer(commandQueue, filterBuffer, CL_TRUE, 0, filter_size, filterVector, 0, NULL, NULL));
	// set kernel arguments
	checkStatus(clSetKernelArg(kernel, 0, sizeof(cl_mem), &pixelBuffer));
	checkStatus(clSetKernelArg(kernel, 1, sizeof(row_size), NULL));
	checkStatus(clSetKernelArg(kernel, 2, sizeof(cl_mem), &filterBuffer));
	checkStatus(clSetKernelArg(kernel, 3, sizeof(cl_mem), &outputBuffer));
	checkStatus(clSetKernelArg(kernel, 4, sizeof(int), &smooth_kernel_size));
	checkStatus(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSizeWidth, 0, NULL, NULL));
	checkStatus(clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, vector_size, outPixels, 0, NULL, NULL));
*/

	// allocate memory for the result image and convert the 1D array to a nice 2D array
	PixelValue **result;
	result = new PixelValue*[imageHeight];
	for(size_t i = 0; i < imageHeight; i++) {
		result[i] = new PixelValue[imageWidth];
	}

	pos = 0;
	for (size_t y = 0; y < imageHeight; y++) {
		for(size_t x = 0; x < imageWidth; x++) {
			result[y][x] = outPixels[pos];
			pos++;
		}
	}

	// free memory
	checkStatus(clReleaseMemObject(pixelBuffer));
	checkStatus(clReleaseMemObject(filterBuffer));
	checkStatus(clReleaseMemObject(outputBuffer));

	return result;
}

void gaussianBlur(cl_context context, cl_command_queue command_queue, cl_kernel kernel)
{
	tga::TGAImage image = loadImage("lena.tga");
	
	PixelValue **pixels = convertImageToPixels(image);
	double *gaussKernel = setupGaussFilterKernel();
	PixelValue **filteredPixels = applyOnGPU(gaussKernel, pixels, image.width, image.height, context, command_queue, kernel);
	
	tga::TGAImage outImage;
	outImage.height = image.height;
	outImage.width = image.width;
	outImage.bpp = image.bpp;
	outImage.type = image.type;

	convertPixelsToImage(filteredPixels, outImage);

	tga::saveTGA(outImage, ("lena_out_gpu_" + std::to_string(smooth_kernel_size) + ".tga").c_str());
}

int main(int argc, char **argv) 
{

	
	// used for checking error status of api calls
	cl_int status;

	// retrieve the number of platforms
	cl_uint numPlatforms = 0;
	checkStatus(clGetPlatformIDs(0, NULL, &numPlatforms));

	if (numPlatforms == 0)
	{
		printf("Error: No OpenCL platform available!\n");
		exit(EXIT_FAILURE);
	}

	// select the platform
	cl_platform_id platform;
	checkStatus(clGetPlatformIDs(1, &platform, NULL));

	// retrieve the number of devices
	cl_uint numDevices = 0;
	checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices));

	if (numDevices == 0)
	{
		printf("Error: No OpenCL device available for platform!\n");
		exit(EXIT_FAILURE);
	}

	// select the device
	cl_device_id device;
	checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));

	// create context
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
	checkStatus(status);

	// create command queue
	cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, &status);
	checkStatus(status);
	
	// read the kernel source
	const char* kernelFileName = "gauss.cl";
	std::ifstream ifs(kernelFileName);
	if (!ifs.good())
	{
		printf("Error: Could not open kernel with file name %s!\n", kernelFileName);
		exit(EXIT_FAILURE);
	}

	std::string programSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	const char* programSourceArray = programSource.c_str();
	size_t programSize = programSource.length();

	// create the program
	cl_program program = clCreateProgramWithSource(context, 1, static_cast<const char**>(&programSourceArray), &programSize, &status);
	checkStatus(status);

	// build the program
	status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		printCompilerError(program, device);
		exit(EXIT_FAILURE);
	}

	// create the gauss kernel
	cl_kernel kernel = clCreateKernel(program, "gauss", &status);
	checkStatus(status);
	
	// output device capabilities
	size_t maxWorkGroupSize;
	checkStatus(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL));
	printf("Device Capabilities: Max work items in single group: %zu\n", maxWorkGroupSize);
	
	cl_uint maxWorkItemDimensions;
	checkStatus(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &maxWorkItemDimensions, NULL));
	printf("Device Capabilities: Max work item dimensions: %u\n", maxWorkItemDimensions);
	
	size_t* maxWorkItemSizes = new size_t[maxWorkItemDimensions];
	checkStatus(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, maxWorkItemDimensions * sizeof(size_t), maxWorkItemSizes, NULL));
	printf("Device Capabilities: Max work items in group per dimension:");
	for (cl_uint i = 0; i < maxWorkItemDimensions; ++i)
		printf(" %u:%zu", i, maxWorkItemSizes[i]);
	printf("\n");
	delete[] (maxWorkItemSizes);

	gaussianBlur(context, commandQueue, kernel);

	// release opencl objects
	checkStatus(clReleaseKernel(kernel));
	checkStatus(clReleaseProgram(program));
	checkStatus(clReleaseCommandQueue(commandQueue));
	checkStatus(clReleaseContext(context));
	
	exit(EXIT_SUCCESS);
}
