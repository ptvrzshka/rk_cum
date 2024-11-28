
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

__kernel void sobel(__global unsigned short* inputImage, __global unsigned short* outputImage)
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

    outputImage[index] = sqrt((pown(Gx, 2) + pown(Gy, 2)));
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
                                 __global int* statsPrev, __global int* statsCurrent, __global int* statsContrast,
                                 int div,
                                 float contrast, __global int* N,
                                 __global unsigned short* outputImage)
{
    int index = get_global_id(0);

    // Calib 
    unsigned short value = (unsigned short)((((int)inputImage[index] - Fs[index]) * K[index]) + 16384);

    // Defect swap
    if (K[index] < 0.0001) 
    {
        float s = 0;
        int defect_index = defects_cnt[index] * 14;
        for (int i = defect_index; i < defect_index + 14; ++i) 
            s += (float)inputImage[i] / 14;
        value = (unsigned short)s;
        //printf("%d\n", value);
    }
    
    atomic_add(&statsCurrent[0], value / div); // For next calibration_and_agc iteration

    // Global contrast
    unsigned short contrastValue = (unsigned short)(((int)value - statsPrev[0]) * contrast + 24678);

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

    atomic_add(&statsContrast[0], contrastValue / div); // For CLAAGC

    outputImage[index] = contrastValue;
}

__kernel void separate_frequences(__global unsigned short* inputImage, __global int* stats, 
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

    atomic_add(&stats[1], pown((float)((int)lowFreqImage[index] - stats[0]), 2) / size); 

    // Get HF
    highFreqImage[index] = value - lowFreqImage[index];
}

__kernel void local_contrast(__global int* inputImage, 
                            __global int* stats, float limit, float multiplecative, int dim,
                            __global int* outputImage) 
{
    int index = get_global_id(0);   

    int x = index / 640;
    int y = index % 640;  

    float x_max = 0.0; float x_min = 65535.0;
    int N_black = 0; int N_white = 0; float N = 0;
    float meanFrame = 0.0;
    float meanFrame1 = 0.0;
    // int dim2 = (dim / 2 + 1) * (dim / 2 + 1); 
    int dim2 = dim * dim; 
    float mean = stats[0];
    float std = stats[1];
    int cnt = 0;
    for (int l = 0; l < dim2; ++l) 
    {
        int i = l / dim - dim / 2;
        int j = l % dim - dim / 2;
        int index_x = abs(x + i);
        int index_y = abs(y + j);
        if (x + i > 511)
            index_x = x - i;
        if (y + j > 640)
            index_y = y - j;
        float k_interp = (1.0 - (pown((float)i, 2) + pown((float)j, 2)) / (float)(dim2 / 2));
        // k_interp = 1;
        // k_interp = pown(k_interp, 2);
        k_interp = rootn(k_interp, 4);
        // printf("%f %f\n", k_interp, pown(k_interp, 4));
        float elem = (float)inputImage[index_x * 640 + index_y];
        meanFrame1 += elem / dim2;
        if (elem * k_interp > x_max)
            x_max = elem * k_interp;
        if (elem * k_interp < x_min)
            x_min = elem * k_interp;
        if (fabs(elem - mean) * k_interp > 2.5 * std) 
            N += 1;
        // else 
        // {
        //     if (elem * k_interp > x_max)
        //         x_max = elem * k_interp;
        //     if (elem * k_interp < x_min)
        //         x_min = elem * k_interp;
        // }
        // if (elem > x_max)
        //     x_max = elem;
        // if (elem < x_min)
        //     x_min = elem;
        // if (mean - elem >= 2.0 * std) 
        // {
        //     N_black += 1;
        // }
        // if (elem - mean <= - 2.0 * std) 
        // {
        //     N_white += 1;     
        // }
        // else 
        // {
        //     cnt += 1;
        //     meanFrame += elem; 
        //     if (elem > x_max)
        //         x_max = elem;
        //     if (elem < x_min)
        //         x_min = elem;
        // }
    }
    // meanFrame = meanFrame / cnt;
    float k = 65535.0 / (x_max - x_min) * multiplecative * (1 - (fabs(mean - meanFrame1) / mean)) * pown((1.0 - N / dim2), 4);
    // printf("%f\n", pown((1.0 - N / dim2), 2));
    if (k > limit)
        k = limit;
    if (k <= 1.0)
        k = 1.0;
    // float b = meanFrame * sqrt(1.0 + (float)mean / meanFrame);
    // float b = mean * sqrt(1.0 + fabs(meanFrame - mean + 1) / mean);
    int contrastValue = (inputImage[index] - meanFrame1) * k + meanFrame1;
    outputImage[index] = convert_ushort_sat(contrastValue);
}
 
__kernel void summary_frequences(__global int* lowFreqImage, __global int* highFreqImage, __global unsigned short* outputImage) 
{
    int index = get_global_id(0);  
    int value = (float)lowFreqImage[index] + (float)highFreqImage[index] * 4;
    outputImage[index] = convert_ushort_sat(value);
}
