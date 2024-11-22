
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics:enable

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

__kernel void mean(__global unsigned short* inputImage, __global int *stats, int div)
{
    int index = get_global_id(0);
    int size = get_global_size(0);
    int value = inputImage[index];
    atomic_add(&stats[0], value / div);
}

__kernel void calc_fs(__global unsigned short* inputImage, __global unsigned short* outputImage, int div)
{
    int index = get_global_id(0);
    int value = (outputImage[index] * (div - 1) + inputImage[index]) / div;
    if (value > 65535)
        value = 65535;
    outputImage[index] = value;
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
    }
    
    atomic_add(&statsCurrent[0], value / div); // For next calibration_and_agc iteration

    // Global contrast
    unsigned short contrastValue = (unsigned short)(((int)value - statsPrev[0]) * contrast + 16384);

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

__kernel void separate_frequences(__global unsigned short* inputImage, __global int* lowFreqImage, __global int* highFreqImage)
{
    int index = get_global_id(0);   

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
    int m2 = median3(inputImage[i4], inputImage[index], inputImage[i6]);
    int m3 = median3(inputImage[i7], inputImage[i8], inputImage[i9]);
    lowFreqImage[index] = median3(m1, m2, m3);

    // Get HF
    highFreqImage[index] = (int)inputImage[index] - lowFreqImage[index];
}
