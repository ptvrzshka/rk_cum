
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics:enable


__kernel void mean(__global unsigned short* inputImage, __global int *stats, int div)
{
    int index = get_global_id(0);
    int size = get_global_size(0);
    int value = inputImage[index];
    atomic_add(&stats[0], value / div);
}

int median3(int a1, int a2, int a3) 
{
    int m = min(max(min(a1, a2), a3), max(a1, a2));
    return m;
}

int median5(int a1, int a2, int a3, int a4, int a5) 
{
    int m1 = min(max(min(a1, a2), a3), max(a1, a2));
    int m2 = min(max(min(a2, a3), a4), max(a2, a3));
    int m3 = min(max(min(a3, a4), a5), max(a3, a4));
    int m = min(max(min(m1, m2), m3), max(m1, m2));
    return m;
}

__kernel void sobel(__global int* inputImage, __global int* outputImage)
{
    int index = get_global_id(0);

    int x = index / 640;
    int y = index % 640;

    int i1 = (x - 1) * 640 + (y - 1);
    int i2 = (x - 1) * 640 + y;
    int i3 = (x - 1) * 640 + (y + 1);

    int i4 = x * 640 + (y - 1);
    int i6 = x * 640 + (y + 1);

    int i7 = (x + 1) * 640 + (y - 1);
    int i8 = (x + 1) * 640 + y;
    int i9 = (x + 1) * 640 + (y + 1);

    float Gx = (inputImage[i7] + 2 * inputImage[i8] + inputImage[i9]) - (inputImage[i1] + 2 * inputImage[i2] + inputImage[i3]);
    float Gy = (inputImage[i3] + 2 * inputImage[i6] + inputImage[i9]) - (inputImage[i1] + 2 * inputImage[i4] + inputImage[i7]);

    outputImage[index] = convert_ushort_sat(sqrt((pown(Gx, 2) + pown(Gy, 2))));
}

__kernel void calc_fs(__global unsigned short* inputImage, __global unsigned short* outputImage, int div)
{
    int index = get_global_id(0);
    int value = (outputImage[index] * (div - 1) + inputImage[index]) / div;
    outputImage[index] = convert_ushort_sat(value);
}

__kernel void calibration_and_agc(__global unsigned short* inputImage, 
                                 __global float* K, __global unsigned short* Fs,
                                 __global int* defects, __global int* defects_cnt,
                                 __global int* statsPrev, __global int* statsCurrent,
                                 int div,
                                 float contrast, __global int* N,
                                 __global unsigned short* outputImage)
{
    int index = get_global_id(0);

    // Calib 
    int value = (int)((((int)inputImage[index] - Fs[index]) * K[index]) + 16384);

    // Defect swap
    if (K[index] < 0.0001) 
    {
        float s = 0;
        int defect_index = defects_cnt[index] * 14;
        for (int i = defect_index; i < defect_index + 14; ++i) 
            s += (float)inputImage[i] / 14;
        value = (int)s;
        //printf("%d\n", value);
    }
    
    atomic_add(&statsCurrent[0], value / div); // For next calibration_and_agc iteration

    // Global contrast
    int contrastValue = (int)(((int)value - statsPrev[0]) * contrast + 24576);

    if (contrastValue <= 0) 
    {
        contrastValue = 0;
        atomic_inc(&N[0]);
    }

    if (contrastValue >= 65535) 
    {
        contrastValue = 65535;
        atomic_inc(&N[1]);
    }

    outputImage[index] = contrastValue;
}

__kernel void separate_frequences(__global unsigned short* inputImage, 
                                __global int* lowFreqImage, __global int* highFreqImage)
{
    int index = get_global_id(0);  
    int size = get_global_size(0);

    int value = inputImage[index]; 

    // Get LF
    int x = index / 640;
    int y = index % 640;    

    int i1 = (x - 1) * 640 + (y - 1);
    int i2 = (x - 1) * 640 + y;
    int i3 = (x - 1) * 640 + (y + 1);

    int i4 = x * 640 + (y - 1);
    int i6 = x * 640 + (y + 1);

    int i7 = (x + 1) * 640 + (y - 1);
    int i8 = (x + 1) * 640 + y;
    int i9 = (x + 1) * 640 + (y + 1);

    int m1 = median3(inputImage[i1], inputImage[i2], inputImage[i3]);
    int m2 = median3(inputImage[i4], value, inputImage[i6]);
    int m3 = median3(inputImage[i7], inputImage[i8], inputImage[i9]);
    lowFreqImage[index] = median3(m1, m2, m3);
    // lowFreqImage[index] = inputImage[index];
    
    // Get HF
    highFreqImage[index] = value - lowFreqImage[index];
}

__kernel void local_contrast(__global int* inputImage, 
                            __global float* meansPrev, __global float* stdsPrev, __global float* meansNew, __global float* stdsNew,
                            float limit, float multiplecative,
                            __global int* outputImage) 
{  
    int index = get_global_id(0); 
    int x = index / 640;
    int y = index % 640;
    float x_max = 0.0; float x_min = 65535.0;
    float meanFrame = 0.0; float stdFrame = 0.0;
    int dim = 31; int step = 4;
    int dim2 = dim * dim / step; 
    float meanPrev = meansPrev[index]; 
    float cnt = 0.0;
    for (int i = (-dim / 2); i < dim / 2 + 1; i+=step)
    {
        for (int j = (-dim / 2); j < dim / 2 + 1; j+=step)
        {
            int index_x = abs(x + i);
            int index_y = abs(y + j);
            if (x + i > 511)
                index_x = x - i;
            if (y + j > 640)
                index_y = y - j;
            int index_new = index_x * 640 + index_y;
            float elem = inputImage[index_new];

            // float k_interp = 1.0 - (pown((float)i, 2) + pown((float)j, 2)) / (float)((dim2 * step) / 2);
            // k_interp = sqrt(k_interp);
            cnt += 1.0;
            float newMean = meanFrame + (elem - meanFrame) / cnt;
            stdFrame += ((elem - meanFrame) * (elem - newMean)); //* k_interp;
            meanFrame = newMean; 

            // float k_interp = 1.0 - (pown((float)i, 2) + pown((float)j, 2)) / (float)((dim2 * step) / 2);
            // meanFrame += elem / dim2;
            // k_interp = pown(k_interp, 4);
            // stdFrame += pown(meanPrev - elem, 2) * k_interp;

            // // k_interp = sqrt(k_interp);
            // // k_interp = 1.0;
            // float elem_interp = elem * k_interp;
            // if (elem_interp > x_max)
            //     x_max = elem_interp;
            // if (elem_interp < x_min)
            //     x_min = elem_interp;
        }
    }

    // meanFrame *= step;
    stdFrame = sqrt(stdFrame / (cnt - 1));

    meansNew[index] = meanFrame;
    stdsNew[index] = stdFrame;

    // float dim_max = 1.5; float dim_min = 1.0;
    // float a = (-dim_max + dim_min) / pown(65535.0, 4);
    // float c = dim_max;
    // float mc = a * pown(stdsPrev[index], 4) + dim_max;
    float k = 65535.0 / (stdFrame + 1.0) * multiplecative / 1.0; 
    // k = sqrt(k);
    k = rootn(k, 3);
    // float k = (65535.0 - stdFrame) / 65535.0 * (limit - 1) + 1;
    // float k_min = 1; k_max = limit;
    // float k = 
    if (k > limit)
        k = limit;
    if (k <= 1.0)
        k = 1.0;
    int contrastValue = (inputImage[index] - meanFrame) * k + meanFrame;  

    outputImage[index] = convert_ushort_sat(contrastValue);



    // int index = get_global_id(0); 
    // int x = index / 640;
    // int y = index % 640;
    // float x_max = 0.0; float x_min = 65535.0;
    // float meanFrame = 0.0; float stdFrame = 0.0;
    // int dim = 43; 
    // int dim2 = dim * dim / 4; 
    // float meanPrev = meansPrev[index]; 
    // for (int l = 0; l < dim2; ++l) 
    // {
    //     if (l % 4 != 0)
    //         continue;
    //     int i = l / dim - dim / 2;
    //     int j = l % dim - dim / 2;
    //     int index_x = abs(x + i);
    //     int index_y = abs(y + j);
    //     if (x + i > 511)
    //         index_x = x - i;
    //     if (y + j > 640)
    //         index_y = y - j;
    //     int index_new = index_x * 640 + index_y;
    //     float elem = inputImage[index_new];
    //     meanFrame += elem / dim2;
    //     stdFrame += pown(meanPrev - elem, 2);
    //     float k_interp = 1.0 - (pown((float)i, 2) + pown((float)j, 2)) / (float)(dim2 / 2);
    //     k_interp = sqrt(k_interp);
    //     float elem_interp = elem * k_interp;
    //     if (elem_interp > x_max)
    //         x_max = elem_interp;
    //     if (elem_interp < x_min)
    //         x_min = elem_interp;
    // }
    // meanFrame *= 4;
    // meansNew[index] = meanFrame;
    // stdsNew[index] = sqrt(stdFrame / dim2);

    // float dim_max = 1.5; float dim_min = 0.1;
    // float a = (-dim_max + dim_min) / pown(65535.0, 4);
    // float c = dim_max;
    // float mc = a * pown(stdsPrev[index], 4) + dim_max;
    // float k = 65535.0 / (x_max - x_min + 1.0) * multiplecative; 
    // if (k > limit)
    //     k = limit;
    // if (k <= 1.0)
    //     k = 1.0;
    // int contrastValue = (inputImage[index] - meanFrame) * k + meanFrame;  
    // outputImage[index] = convert_ushort_sat(contrastValue);
}
 
__kernel void summary_frequences(__global int* lowFreqImage, __global int* highFreqImage, __global unsigned char* outputImage) 
{
    int index = get_global_id(0);  
    // int value = ((float)lowFreqImage[index] + (float)highFreqImage[index] * 4) / 256;
    int value = lowFreqImage[index] / 256;
    outputImage[index] = convert_uchar_sat(value);
}
