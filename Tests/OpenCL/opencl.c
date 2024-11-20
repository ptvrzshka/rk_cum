#include <stdio.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (1048576) 


static cl_device_id device_id;
static cl_int status;
static cl_context context; 
static cl_command_queue command_queue;
static cl_kernel kernel;
static cl_mem memobj_in = NULL;
static cl_mem memobj_out = NULL;
static size_t numNames = 0;

static int memLenth = 640 * 512;
static size_t global_work_size[1] = { 640 * 512 };


void get_device_id(cl_device_id *device_id) 
{
    cl_platform_id platform_id;
    cl_uint numPlatforms;
    cl_uint ret_num_devices;

    status = clGetPlatformIDs(1, &platform_id, &numPlatforms);
    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, device_id, &ret_num_devices);

    status = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 0, NULL, &numNames);
    char Name[numNames];
    status = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sizeof(Name), Name, NULL);
    printf("Name of platform: %s\n", Name);
}

cl_kernel create_kernel(cl_context context, cl_device_id device_id) 
{
    cl_program program = NULL;
    cl_kernel kernel = NULL;

    FILE *fp;
    const char fileName[] = "../OpenCL/tpc.cl";
    size_t source_size;
    char *source_str;

    fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char *)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &status);

    status = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    kernel = clCreateKernel(program, "sobel", &status);

    return kernel;
}

void create_buffer(cl_context context, cl_command_queue command_queue, cl_kernel kernel, int memLenth) 
{
    memobj_in = clCreateBuffer(context, CL_MEM_READ_ONLY, memLenth * sizeof(unsigned short), NULL, &status);
    memobj_out = clCreateBuffer(context, CL_MEM_READ_WRITE, memLenth * sizeof(unsigned short), NULL, &status);
}

void init_image_processor(unsigned short *membuffer) 
{
    get_device_id(&device_id);
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &status);
    command_queue = clCreateCommandQueue(context, device_id, 0, &status);
    kernel = create_kernel(context, device_id);    
    create_buffer(context, command_queue, kernel, memLenth);

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj_in);

    status = clEnqueueWriteBuffer(command_queue, memobj_out, CL_TRUE, 0, memLenth * sizeof(unsigned short), membuffer, 0, NULL, NULL);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memobj_out);
}

void process_image(unsigned short* inputImage, unsigned short* outputImage) 
{
    status = clEnqueueWriteBuffer(command_queue, memobj_in, CL_TRUE, 0, memLenth * sizeof(unsigned short), inputImage, 0, NULL, NULL);

    status = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
    status = clEnqueueReadBuffer(command_queue, memobj_out, CL_TRUE, 0, memLenth * sizeof(unsigned short), outputImage, 0, NULL, NULL);
}
