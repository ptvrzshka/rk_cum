#include <stdio.h>
#include <CL/cl.h>

#include <math.h>
#include <cstring>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define MAX_SOURCE_SIZE (1048576) 


static cl_device_id device_id;
static cl_int status;
static cl_context context; 
static cl_command_queue command_queue;

static cl_kernel fs_kernel = NULL;
static cl_kernel base_proc_kernel = NULL;
static cl_kernel freq_sep_kernel = NULL;
static cl_kernel local_contrast_kernel = NULL;
static cl_kernel sharpen_kernel = NULL;
static cl_kernel bilateral_kernel = NULL;

// static cl_kernel freq_sum_kernel = NULL;

static cl_mem memobj_in = NULL;
static cl_mem memobj_temp = NULL;
static cl_mem memobj_out = NULL;

static cl_mem memobj_k = NULL;
static cl_mem memobj_fs = NULL;
static cl_mem memobj_defects = NULL;
static cl_mem memobj_defectsCnt = NULL;
static cl_mem memobj_statsPrev = NULL;
static cl_mem memobj_statsCurrent = NULL;
static cl_mem memobj_rangeCnt = NULL;
static cl_mem memobj_lowFreq = NULL;
static cl_mem memobj_lowFreqTemp= NULL;
static cl_mem memobj_highFreq = NULL;

static cl_mem memobj_lcMeansPrev = NULL;
static cl_mem memobj_lcMeansNew = NULL;
static cl_mem memobj_lcStdsPrev = NULL;
static cl_mem memobj_lcStdsNew = NULL;

static const int memLenth = 640 * 512;
static size_t global_work_size[1] = { memLenth };
static size_t global_work_size_1[2] = { 512, 640 };
static size_t local_work_size_1[2] = { 32, 32 };
static int defectsLenth = 0;

static cl_int bytesDivider = 8;
static cl_float contrast = 1.0;
static cl_float localContrastLimit = 64.0;
static cl_float localContrastMultiplecative = 1.0;
static cl_int localContrastDim = 31; // 35

static cl_float sigma_s = 15.0;
static cl_float sigma_r = 15.0;

static float* K = new float[memLenth];
static unsigned short* Fs = new unsigned short[memLenth];
static int* defects = nullptr;
static int* defectsCnt = new int[memLenth];
static int* statsPrev = new int[2] {0 , 0};
static int* statsCurrent = new int[2] {0 , 0};
static int* hist = new int[65536];

static int* lowFreq = new int[memLenth];
static int* highFreq = new int[memLenth];

static float* lcMeansNew = new float[memLenth];
static float* lcStdsNew = new float[memLenth];

static unsigned short* tempImage = new unsigned short[memLenth];


void restore_fs() 
{
    memset(Fs, 0, memLenth * sizeof(unsigned short));
}

void load_calibration_data() 
{
    FILE *fp;
    const char fileName[] = "/home/ubuntu/projects/Embedded/Tests/OpenCL/K_new.data";
    size_t source_size;

    fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load K data.\n");
		exit(1);
	}
	source_size = fread(K, sizeof(float), memLenth, fp);
	fclose(fp);
}

void load_defects_data() 
{
    FILE *fp;
    const char fileName[] = "/home/ubuntu/projects/Embedded/Tests/OpenCL/defects_coord.data";
    size_t source_size;

    fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load Defects data.\n");
		exit(1);
	}
    fseek(fp, 0L, SEEK_END);
    defectsLenth = ftell(fp) / sizeof(int);
    rewind(fp);
    defects = new int[defectsLenth];
	source_size = fread(defects, sizeof(int), defectsLenth, fp);
	fclose(fp);

    const char fileName_cnt[] = "/home/ubuntu/projects/Embedded/Tests/OpenCL/defects_cnt.data";
    fp = fopen(fileName_cnt, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load Defects Counter data.\n");
		exit(1);
	}
    source_size = fread(defectsCnt, sizeof(int), memLenth, fp);
	fclose(fp);
}

void get_device_id(cl_device_id *device_id) 
{
    cl_platform_id platform_id;
    cl_uint numPlatforms;
    cl_uint ret_num_devices;
    size_t numNames = 0;
    size_t versNames = 0;

    status = clGetPlatformIDs(1, &platform_id, &numPlatforms);
    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, device_id, &ret_num_devices);

    status = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 0, NULL, &numNames);
    char Name[numNames];
    status = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sizeof(Name), Name, NULL);
    printf("Name of platform: %s\n", Name);
}

cl_program create_program(cl_context context, cl_device_id device_id) 
{
    FILE *fp;
    const char fileName[] = "/home/ubuntu/projects/Embedded/Tests/OpenCL/tpc.cl";
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

    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &status);

    status = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    return program;
}

cl_kernel create_kernel(cl_program program, const char *kernelName) 
{
    cl_kernel kernel = clCreateKernel(program, kernelName, &status);
    return kernel;
}

void create_buffers(cl_context context, cl_command_queue command_queue, int memLenth) 
{
    memobj_in = clCreateBuffer(context, CL_MEM_READ_ONLY, memLenth * sizeof(unsigned short), NULL, &status);
    memobj_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, memLenth * sizeof(unsigned char), NULL, &status);
    memobj_temp = clCreateBuffer(context, CL_MEM_READ_WRITE, memLenth * sizeof(unsigned short), NULL, &status);
    memobj_fs = clCreateBuffer(context, CL_MEM_READ_WRITE, memLenth * sizeof(unsigned short), NULL, &status);
    memobj_k = clCreateBuffer(context, CL_MEM_READ_ONLY, memLenth * sizeof(float), NULL, &status);
    memobj_defects = clCreateBuffer(context, CL_MEM_READ_ONLY, defectsLenth * sizeof(int), NULL, &status);
    memobj_defectsCnt = clCreateBuffer(context, CL_MEM_READ_ONLY, memLenth * sizeof(int), NULL, &status);
    memobj_statsPrev = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 2 * sizeof(int), NULL, &status);
    memobj_statsCurrent = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * sizeof(int), NULL, &status);
    memobj_rangeCnt = clCreateBuffer(context, CL_MEM_READ_WRITE, 65536 * sizeof(int), NULL, &status);

    memobj_lowFreq = clCreateBuffer(context, CL_MEM_READ_WRITE, memLenth * sizeof(int), NULL, &status);
    memobj_lowFreqTemp = clCreateBuffer(context, CL_MEM_READ_WRITE, memLenth * sizeof(int), NULL, &status);
    memobj_highFreq = clCreateBuffer(context, CL_MEM_READ_WRITE, memLenth * sizeof(int), NULL, &status);

    memobj_lcMeansPrev = clCreateBuffer(context, CL_MEM_READ_ONLY, memLenth * sizeof(float), NULL, &status);
    memobj_lcMeansNew = clCreateBuffer(context, CL_MEM_WRITE_ONLY, memLenth * sizeof(float), NULL, &status);
    memobj_lcStdsPrev = clCreateBuffer(context, CL_MEM_READ_ONLY, memLenth * sizeof(float), NULL, &status);
    memobj_lcStdsNew = clCreateBuffer(context, CL_MEM_WRITE_ONLY, memLenth * sizeof(float), NULL, &status);

    // memobj_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &status);
}

void init_image_processor() 
{
    get_device_id(&device_id);
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &status);

#if defined(CL_VERSION_2_0)
    command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &status);
#else
    command_queue = clCreateCommandQueue(context, device_id, 0, &status);
#endif

    cl_program program = create_program(context, device_id);

    fs_kernel = create_kernel(program, "calc_fs");  
    base_proc_kernel = create_kernel(program, "calibration_and_agc");  
    freq_sep_kernel = create_kernel(program, "separate_frequences");  
    local_contrast_kernel = create_kernel(program, "local_contrast"); 
    sharpen_kernel = create_kernel(program, "sharpen");
    bilateral_kernel = create_kernel(program, "bilateral_filter");
    // freq_sum_kernel = create_kernel(program, "summary_frequences");   

    create_buffers(context, command_queue, memLenth);

    status = clSetKernelArg(fs_kernel, 0, sizeof(cl_mem), (void *)&memobj_in);
    status = clSetKernelArg(fs_kernel, 1, sizeof(cl_mem), (void *)&memobj_fs);

    status = clSetKernelArg(base_proc_kernel, 0, sizeof(cl_mem), (void *)&memobj_in);
    status = clSetKernelArg(base_proc_kernel, 1, sizeof(cl_mem), (void *)&memobj_k);
    status = clSetKernelArg(base_proc_kernel, 2, sizeof(cl_mem), (void *)&memobj_fs);
    status = clSetKernelArg(base_proc_kernel, 3, sizeof(cl_mem), (void *)&memobj_defects);
    status = clSetKernelArg(base_proc_kernel, 4, sizeof(cl_mem), (void *)&memobj_defectsCnt);
    status = clSetKernelArg(base_proc_kernel, 5, sizeof(cl_mem), (void *)&memobj_statsPrev);
    status = clSetKernelArg(base_proc_kernel, 6, sizeof(cl_mem), (void *)&memobj_statsCurrent);
    status = clSetKernelArg(base_proc_kernel, 7, sizeof(cl_int), &bytesDivider);
    status = clSetKernelArg(base_proc_kernel, 8, sizeof(cl_float), &contrast);
    status = clSetKernelArg(base_proc_kernel, 9, sizeof(cl_mem), (void *)&memobj_rangeCnt);
    status = clSetKernelArg(base_proc_kernel, 10, sizeof(cl_mem), (void *)&memobj_temp);

    status = clSetKernelArg(freq_sep_kernel, 0, sizeof(cl_mem), (void *)&memobj_temp);
    status = clSetKernelArg(freq_sep_kernel, 1, sizeof(cl_mem), (void *)&memobj_lowFreq);
    status = clSetKernelArg(freq_sep_kernel, 2, sizeof(cl_mem), (void *)&memobj_highFreq);

    status = clSetKernelArg(local_contrast_kernel, 0, sizeof(cl_mem), (void *)&memobj_lowFreq);
    status = clSetKernelArg(local_contrast_kernel, 1, sizeof(cl_mem), (void *)&memobj_lcMeansPrev);
    status = clSetKernelArg(local_contrast_kernel, 2, sizeof(cl_mem), (void *)&memobj_lcStdsPrev);
    status = clSetKernelArg(local_contrast_kernel, 3, sizeof(cl_mem), (void *)&memobj_lcMeansNew);
    status = clSetKernelArg(local_contrast_kernel, 4, sizeof(cl_mem), (void *)&memobj_lcStdsNew);
    status = clSetKernelArg(local_contrast_kernel, 5, sizeof(cl_float), &localContrastLimit);
    status = clSetKernelArg(local_contrast_kernel, 6, sizeof(cl_float), &localContrastMultiplecative);
    status = clSetKernelArg(local_contrast_kernel, 7, sizeof(cl_mem), (void *)&memobj_lowFreqTemp);

    status = clSetKernelArg(sharpen_kernel, 0, sizeof(cl_mem), (void *)&memobj_lowFreqTemp);
    status = clSetKernelArg(sharpen_kernel, 1, sizeof(cl_mem), (void *)&memobj_lowFreq);

    status = clSetKernelArg(bilateral_kernel, 0, sizeof(cl_mem), (void *)&memobj_lowFreq);
    status = clSetKernelArg(bilateral_kernel, 1, sizeof(cl_float), &sigma_s);
    status = clSetKernelArg(bilateral_kernel, 2, sizeof(cl_float), &sigma_r);
    status = clSetKernelArg(bilateral_kernel, 3, sizeof(cl_mem), (void *)&memobj_out);

    // status = clSetKernelArg(freq_sum_kernel, 0, sizeof(cl_mem), (void *)&memobj_lowFreq);
    // status = clSetKernelArg(freq_sum_kernel, 1, sizeof(cl_mem), (void *)&memobj_highFreq);
    // status = clSetKernelArg(freq_sum_kernel, 2, sizeof(cl_mem), (void *)&memobj_out);

    restore_fs();
    load_calibration_data();
    load_defects_data();
}

// void exec_summury_frequences_kernel(int* lowFreq, int* highFreq, unsigned char* outputImage) 
// {
//     status = clEnqueueWriteBuffer(command_queue, memobj_lowFreq, CL_TRUE, 0, memLenth * sizeof(int), lowFreq, 0, NULL, NULL);
//     status = clEnqueueWriteBuffer(command_queue, memobj_highFreq, CL_TRUE, 0, memLenth * sizeof(int), highFreq, 0, NULL, NULL);

//     status = clEnqueueNDRangeKernel(command_queue, freq_sum_kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

//     status = clEnqueueReadBuffer(command_queue, memobj_out, CL_TRUE, 0, memLenth * sizeof(unsigned char), outputImage, 0, NULL, NULL);
// }

void exec_bilateral_kernel(int* inputImage, unsigned char* outputImage) 
{
    status = clEnqueueWriteBuffer(command_queue, memobj_lowFreq, CL_TRUE, 0, memLenth * sizeof(int), inputImage, 0, NULL, NULL);

    status = clEnqueueNDRangeKernel(command_queue, bilateral_kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

    status = clEnqueueReadBuffer(command_queue, memobj_out, CL_TRUE, 0, memLenth * sizeof(unsigned char), outputImage, 0, NULL, NULL);
}

void exec_sharpen_kernel(int* inputImage, int* outputImage)  
{
    status = clEnqueueWriteBuffer(command_queue, memobj_lowFreqTemp, CL_TRUE, 0, memLenth * sizeof(int), inputImage, 0, NULL, NULL);

    status = clEnqueueNDRangeKernel(command_queue, sharpen_kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

    status = clEnqueueReadBuffer(command_queue, memobj_lowFreq, CL_TRUE, 0, memLenth * sizeof(int), outputImage, 0, NULL, NULL);
}

void exec_local_contrast_kernel(int* inputImage, int* outputImage) 
{
    status = clEnqueueWriteBuffer(command_queue, memobj_lowFreq, CL_TRUE, 0, memLenth * sizeof(int), inputImage, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(command_queue, memobj_lcMeansPrev, CL_TRUE, 0, memLenth * sizeof(float), lcMeansNew, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(command_queue, memobj_lcStdsPrev, CL_TRUE, 0, memLenth * sizeof(float), lcStdsNew, 0, NULL, NULL);

    status = clEnqueueNDRangeKernel(command_queue, local_contrast_kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

    status = clEnqueueReadBuffer(command_queue, memobj_lowFreqTemp, CL_TRUE, 0, memLenth * sizeof(int), outputImage, 0, NULL, NULL);
    status = clEnqueueReadBuffer(command_queue, memobj_lcMeansNew, CL_TRUE, 0, memLenth * sizeof(float), lcMeansNew, 0, NULL, NULL);
    status = clEnqueueReadBuffer(command_queue, memobj_lcStdsNew, CL_TRUE, 0, memLenth * sizeof(float), lcStdsNew, 0, NULL, NULL);
}

void exec_separate_frequences(unsigned short* inputImage, int* lowFreq, int* highFreq) 
{
    status = clEnqueueWriteBuffer(command_queue, memobj_temp, CL_TRUE, 0, memLenth * sizeof(unsigned short), inputImage, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(command_queue, memobj_lowFreq, CL_TRUE, 0, memLenth * sizeof(int), lowFreq, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(command_queue, memobj_highFreq, CL_TRUE, 0, memLenth * sizeof(int), highFreq, 0, NULL, NULL);

    status = clEnqueueNDRangeKernel(command_queue, freq_sep_kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

    status = clEnqueueReadBuffer(command_queue, memobj_lowFreq, CL_TRUE, 0, memLenth * sizeof(int), lowFreq, 0, NULL, NULL);
    status = clEnqueueReadBuffer(command_queue, memobj_highFreq, CL_TRUE, 0, memLenth * sizeof(int), highFreq, 0, NULL, NULL);

    // printf("Mean: %d ", statsContrast[0]);
    // printf("Std: %d\n", statsContrast[1]);
}

static const float p_agc_black = 0.001;
static const float p_agc_white = 0.001;
static float agc_limit = 12.0; // 8 for local

void exec_firts_kernel(unsigned short* inputImage, unsigned short* outputImage) 
{
    memset(hist, 0, memLenth);

    status = clEnqueueWriteBuffer(command_queue, memobj_in, CL_TRUE, 0, memLenth * sizeof(unsigned short), inputImage, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(command_queue, memobj_k, CL_TRUE, 0, memLenth * sizeof(float), K, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(command_queue, memobj_fs, CL_TRUE, 0, memLenth * sizeof(unsigned short), Fs, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(command_queue, memobj_defects, CL_TRUE, 0, defectsLenth * sizeof(int), defects, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(command_queue, memobj_defectsCnt, CL_TRUE, 0, memLenth * sizeof(int), defectsCnt, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(command_queue, memobj_statsPrev, CL_TRUE, 0, 2 * sizeof(int), statsPrev, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(command_queue, memobj_statsCurrent, CL_TRUE, 0, 2 * sizeof(int), statsCurrent, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(command_queue, memobj_rangeCnt, CL_TRUE, 0, 65536 * sizeof(int), hist, 0, NULL, NULL);
    status = clSetKernelArg(base_proc_kernel, 8, sizeof(cl_float), &contrast);

    status = clEnqueueNDRangeKernel(command_queue, base_proc_kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

    status = clEnqueueReadBuffer(command_queue, memobj_statsCurrent, CL_TRUE, 0, 2 * sizeof(int), statsCurrent, 0, NULL, NULL);
    status = clEnqueueReadBuffer(command_queue, memobj_rangeCnt, CL_TRUE, 0, 65536 * sizeof(int), hist, 0, NULL, NULL);
    status = clEnqueueReadBuffer(command_queue, memobj_temp, CL_TRUE, 0, memLenth * sizeof(unsigned short), outputImage, 0, NULL, NULL);

    int N_black = 0; int N_white = 0;
    int l_tail = 0; int r_tail = 65535;
    bool l_flag = false; bool r_flag = false;
    for (int i = 0; i < 65536; ++i) 
    {
        // printf("%d\n", hist[i]);
        if (!l_flag) 
        {
            N_black += hist[i];
            if (N_black > memLenth * p_agc_black) 
            {
                l_tail = i;
                l_flag = true;
            }
        }

        if (!r_flag) 
        {
            N_white += hist[65535 - i];
            if (N_white > memLenth * p_agc_white) 
            {
                r_tail = 65535 - i;
                r_flag = true;
            }
        }

        if (l_flag && r_flag)
            break;
    }

    contrast = 65535.0 / (r_tail - l_tail + 1.0);
    // printf("%f\n", contrast);

    if (contrast > agc_limit)
        contrast = agc_limit;

    statsPrev[0] = statsCurrent[0] / memLenth * bytesDivider;
    statsCurrent[0] = 0;

    // int N = rangeCnt[0] + rangeCnt[1];
    // if (N <= p_agc_black * memLenth && contrast < agc_limit)
    //     contrast = contrast * (1 + 0.03125) + 1e-6;
    // if (N >= p_agc_white * memLenth && contrast > 0.0)
    //     contrast = contrast * (1 - 0.03125) + 1e-6;
    // statsPrev[0] = statsCurrent[0] / memLenth * bytesDivider;
    // statsCurrent[0] = 0;
    // rangeCnt[0] = 0; rangeCnt[1] = 0;
}

void calc_fs(unsigned short* inputImage, int iter, bool start) 
{
    if (iter < 1) 
    {
        printf("Error: iter value must be >= 1\n");
        return;
    }
    
    if (start) restore_fs();

    status = clEnqueueWriteBuffer(command_queue, memobj_in, CL_TRUE, 0, memLenth * sizeof(unsigned short), inputImage, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(command_queue, memobj_fs, CL_TRUE, 0, memLenth * sizeof(unsigned short), Fs, 0, NULL, NULL);
    status = clSetKernelArg(fs_kernel, 2, sizeof(cl_int), (cl_int*)&iter);

    status = clEnqueueNDRangeKernel(command_queue, fs_kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
    status = clEnqueueReadBuffer(command_queue, memobj_fs, CL_TRUE, 0, memLenth * sizeof(unsigned short), Fs, 0, NULL, NULL);
}

void process_image(unsigned short* inputImage, unsigned char* outputImage) 
{    
    exec_firts_kernel(inputImage, tempImage);
    exec_separate_frequences(tempImage, lowFreq, highFreq);
    exec_local_contrast_kernel(lowFreq, lowFreq);
    exec_sharpen_kernel(lowFreq, lowFreq);
    exec_bilateral_kernel(lowFreq, outputImage);
}

void global_contrast_changed(unsigned char value) 
{
    agc_limit = (float)value;
}

void local_contrast_changed(unsigned char value) 
{
    localContrastMultiplecative = (cl_float)value / 100.0;
    status = clSetKernelArg(local_contrast_kernel, 6, sizeof(cl_float), &localContrastMultiplecative);
}

void denoise_1_changed(unsigned char value) 
{
    sigma_s = (cl_float)value;
    status = clSetKernelArg(bilateral_kernel, 1, sizeof(cl_float), &sigma_s);
}

void denoise_2_changed(unsigned char value) 
{
    sigma_r = (cl_float)value;
    status = clSetKernelArg(bilateral_kernel, 2, sizeof(cl_float), &sigma_r);
}
