
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
                                 float contrast, __global int* hist,
                                 __global int* outputImage)
{
    int index = get_global_id(0);

    // Calib 
    int value = (int)((((int)inputImage[index] - Fs[index]) * K[index]) + 16384);

    int x = index / 640;
    int y = index % 640 - 1;    
    int index_1 = x * 640 + y;
    // Defect swap
    if (K[index_1] < 0.0001) 
    {
        float s = 0;
        int defect_index = defects_cnt[index_1] * 14;
        for (int i = defect_index; i < defect_index + 14; ++i) 
            s += (float)inputImage[i] / 14;
        value = (int)s;
    }
    
    atomic_add(&statsCurrent[0], value / div); // For the next iteration

    atomic_inc(&hist[convert_ushort_sat(value)]);

    int contrastValue = (int)(((int)value - statsPrev[0]) * contrast + statsPrev[0]);

    outputImage[index] = convert_ushort_sat(contrastValue);
}

__kernel void separate_frequences(__global int* inputImage, 
                                __global int* lowFreqImage, __global int* highFreqImage)
{
    int index = get_global_id(0);  

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

    // int m1 = median3(inputImage[i1], inputImage[i2], inputImage[i3]);
    // int m2 = median3(inputImage[i4], value, inputImage[i6]);
    // int m3 = median3(inputImage[i7], inputImage[i8], inputImage[i9]);
    // lowFreqImage[index] = median3(m1, m2, m3);

    int min1 = min(min(inputImage[i1], inputImage[i2]), inputImage[i3]);
    int min2 = min(min(inputImage[i4], value), inputImage[i6]);
    int min3 = min(min(inputImage[i7], inputImage[i8]), inputImage[i9]);
    int min4 = min(min(min1, min2), min3);

    int max1 = max(max(inputImage[i1], inputImage[i2]), inputImage[i3]);
    int max2 = max(max(inputImage[i4], value), inputImage[i6]);
    int max3 = max(max(inputImage[i7], inputImage[i8]), inputImage[i9]);
    int max4 = max(max(max1, max2), max3);

    lowFreqImage[index] = min4 / 2 + max4 / 2;

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
    float meanFrame = 0.0; float stdFrame = 0.0; float meanFrame4 = 0.0; float meanFrame8 = 0.0; float stdFrame4 = 0.0; float stdFrame8 = 0.0;
    int dim = 25; int step = 4;
    int dim2 = dim * dim / step; 
    float cnt = 0.0; float cnt4 = 0.0; float cnt8 = 0.0;
    for (int i = (-dim / 2); i < dim / 2 + 1; i+=step)
    {
        for (int j = (-dim / 2); j < dim / 2 + 1; j+=step)
        {
            int index_x = abs(x + i);
            int index_y = abs(y + j);
            if (x + i > 511)
                index_x = x - i;
            if (y + j > 639)
                index_y = y - j;
            int index_new = index_x * 640 + index_y;

            float elem = inputImage[index_new];

            cnt += 1.0;
            float newMean = meanFrame + (elem - meanFrame) / cnt;
            stdFrame += ((elem - meanFrame) * (elem - newMean));
            meanFrame = newMean; 

            if (i >= -4 && i <= 4 && j >= -4 && j <= 4) 
            {
                cnt4 += 1.0;
                float newMean4 = meanFrame4 + (elem - meanFrame4) / cnt4;
                stdFrame4 += ((elem - meanFrame4) * (elem - newMean4));
                meanFrame4 = newMean4; 
            }

            if (i >= -8 && i <= 8 && j >= -8 && j <= 8) 
            {
                cnt8 += 1.0;
                float newMean8 = meanFrame8 + (elem - meanFrame8) / cnt8;
                stdFrame8 += ((elem - meanFrame8) * (elem - newMean8));
                meanFrame8 = newMean8; 
            }

        }
    }

    stdFrame = sqrt(stdFrame / (cnt - 1));
    stdFrame4 = sqrt(stdFrame4 / (cnt4 - 1));
    stdFrame8 = sqrt(stdFrame8 / (cnt8 - 1));

    float t = 1024.0;
    if (stdFrame >= t && stdFrame < t * 2) 
    {
        meanFrame = meanFrame8;
        stdFrame = stdFrame8;
    }
    if (stdFrame >= t * 2) 
    {
        meanFrame = meanFrame4;
        stdFrame = stdFrame4;
    }
    // if (stdFrame >= t * 3)
    //     meanFrame = inputImage[index];

    meansNew[index] = meanFrame;
    stdsNew[index] = stdFrame;

    float k = 65535.0 / (stdFrame + 1.0); 
    // k = half_powr(k, 2.0 / 3.0)
    k = rootn(k, 3) * multiplecative;
    if (k > limit)
        k = limit;
    // if (k <= 1.0)
    //     k = 1.0;

    int contrastValue = (inputImage[index] - meanFrame) * k + meanFrame;  

    if (contrastValue < 0)
        contrastValue = 0;
    if (contrastValue > 65535)
        contrastValue = 65535;

    outputImage[index] = convert_ushort_sat(contrastValue);
}
 
__kernel void summary_frequences(__global int* lowFreqImage, __global int* highFreqImage, __global int* outputImage) 
{
    int index = get_global_id(0);  
    int value = ((float)lowFreqImage[index] + (float)highFreqImage[index]);
    if (value < 0)
        value = 0;
    if (value > 65535)
        value = 65535;
    // int value = lowFreqImage[index] / 256;
    outputImage[index] = convert_ushort_sat(value);
}

__kernel void sharpen(__global int* inputImage, __global int* outputImage) 
{
    int index = get_global_id(0);

    int x = index / 640;
    int y = index % 640;

    // outputImage[index] = inputImage[index];

    if (x < 1 || y < 1 || x > 510 || x > 638)
        outputImage[index] = inputImage[index];
    else
        // outputImage[index] = inputImage[index] * 9 - inputImage[(x - 1) * 640 + y] - inputImage[(x + 1) * 640 + y] - inputImage[x * 640 + y + 1] - inputImage[x * 640 + y - 1] - inputImage[(x - 1) * 640 + (y - 1)] - inputImage[(x - 1) * 640 + (y + 1)] - inputImage[(x + 1) * 640 + (y - 1)] - inputImage[(x + 1) * 640 + (y + 1)];
        outputImage[index] = inputImage[index] * 5 - inputImage[(x - 1) * 640 + y] - inputImage[(x + 1) * 640 + y] - inputImage[x * 640 + y + 1] - inputImage[x * 640 + y - 1];
}

__kernel void dde(__global int* inputImage, float hs, __global int* hs_new, int k, int div, __global int* outputImage) 
{
    int index = get_global_id(0);

    float value = inputImage[index]; 

    int x = index / 640;
    int y = index % 640;    

    int h = 0;
    for (int i = -1; i < 2; ++i) 
    {
        for (int j = -1; j < 2; ++j) 
        {
            h += abs_diff(inputImage[(x + i) * 640 + (y + j)], value);
        }
    }
    h = h / 9;
    atomic_add(&hs_new[0], h / div);

    int m = 3;
    float elem = 65535.0;
    if (value < - hs * m || value > hs * m) 
    {
        elem = 65536 / (1 + exp(-k * value / 65536)) - 32768;
    }
    else 
    {
        float koef = (65536 / (1 + exp(-k * hs * m / 65536)) - 32768) / pown(hs * m + 1, 2);
        if (value < 0)
            elem = -koef * pown(value, 2);
        else
            elem = koef * pown(value, 2);
    }

    outputImage[index] = elem;

}

__kernel void bilateral_filter(__global int* inputImage, 
                               float sigma_s, float sigma_r,
                               __global unsigned char* outputImage) 
{
    int index = get_global_id(0);

    // int x = index / 640;
    // int y = index % 640;

    // float sigma_s_2 = 2 * pown(sigma_s, 2);
    // float sigma_r_2 = 2 * pown(sigma_r, 2);

    // int dim = 7;
    // float s1 = 0.0; float s2 = 0.0;
    // float currentElem = inputImage[index] / 256;
    // for (int i = (-dim / 2); i < dim / 2 + 1; ++i)
    // {
    //     for (int j = (-dim / 2); j < dim / 2 + 1; ++j)
    //     {
    //         int index_x = abs(x + i);
    //         int index_y = abs(y + j);
    //         if (x + i > 511)
    //             index_x = x - i;
    //         if (y + j > 640)
    //             index_y = y - j;
    //         float elem = inputImage[index_x * 640 + index_y] / 256;
    //         // float w_s = exp(-((pown((float)(x - index_x), 2) + pown((float)(y - index_y), 2)) / sigma_s_2));
    //         // float w_r = exp(-(pown(currentElem - elem, 2) / sigma_r_2));
    //         float w = exp(-((pown((float)(x - index_x), 2) + pown((float)(y - index_y), 2)) / sigma_s_2) -(pown(currentElem - elem, 2) / sigma_r_2));
    //         // float w_s = exp(-1.0);
    //         // float w_r = exp(1.0);
    //         s1 += w * elem;
    //         s2 += w;
    //     }
    // }

    // outputImage[index] = convert_uchar_sat(s1 / s2);

    outputImage[index] = convert_uchar_sat(inputImage[index] / 256);
}
