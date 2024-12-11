#include <iostream>

#include "../OpenCL/opencl.cpp"
#include "../Ethernet/ethernet.cpp"

#include <mutex>
#include <condition_variable>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <gst/gst.h>


std::mutex procMtx;
std::condition_variable procCond;
std::mutex visMtx;
std::condition_variable visCond;
bool stop_flag = false;

unsigned short* frame = nullptr;
unsigned short* frameOut = new unsigned short[WIDTH * HEIGHT];

std::chrono::time_point<std::chrono::system_clock> startTime;
std::chrono::time_point<std::chrono::system_clock> endTime;
int framesCnt = 0;

void CalcFps() 
{
	endTime = std::chrono::system_clock::now();
	std::cout << "Fps: " << (float)framesCnt / std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() * 1000.0 << std::endl;
	framesCnt = 0;
	startTime = std::chrono::system_clock::now();
}

void ReaderLoop()
{
	while (!stop_flag)
	{
		frame = GetFramebuffer(); 
		procCond.notify_one();
	}
}

void ProcessLoop() 
{
	while (!stop_flag)
	{
		std::unique_lock<std::mutex> lock(procMtx);
		procCond.wait(lock);
		process_image(frame, frameOut);
		visCond.notify_one();
	}
}

int main(int argc, char *argv[]) 
{

    InitializeSocket(new int[4] { 192, 168, 0, 1 }, 50016);
	InitializeCamera(new int[4] { 192, 168, 0, 15 }, 50000);

    std::thread readerThread(&ReaderLoop);
	// std::thread processorThread(&ProcessLoop);

	// init_image_processor(frameOut);

    std::cout << "Start reading" << std::endl;

	int key = 0;
	int fs_cnt = 0;
	startTime = std::chrono::system_clock::now();
	while (true)
	{
		// std::unique_lock<std::mutex> lock(visMtx);
		// visCond.wait(lock);

		std::unique_lock<std::mutex> lock(procMtx);
		procCond.wait(lock);

		if (fs_cnt != 0) 
		{
			if (fs_cnt == 1)
				calc_fs(frame, fs_cnt, true);
			calc_fs(frame, fs_cnt, false);
			fs_cnt += 1;
			if (fs_cnt > 16)
				fs_cnt = 0;
		}

		framesCnt += 1;
		if (framesCnt == 30)
			CalcFps();

		cv::Mat image_cv(HEIGHT, WIDTH, CV_16UC1, frame);
		cv::Mat image_cv_out(HEIGHT, WIDTH, CV_16UC1, frameOut);

		imshow("Original", image_cv);
		imshow("Processed", image_cv_out);

		key = cv::waitKey(1);
		if (key == 99) {
			int packet_data[6] = { 0x05, 0x5C, 0xCF, 0x00, 0x00, 0x00 };
			SendData(packet_data, 6); // Команда на калибровку 
		}
		if (key == 98) {
			int packet_data[6] = { 0x05, 0x5C, 0x00, 0x00, 0x0E, 0x80 };
			SendData(packet_data, 6); // Команда на Прямой проход 
		}
		if (key == 97) {
			SendFrequence(25);
			SendIntTime(25000);
		}
		if (key == 100) {
			fs_cnt = 1;
		}
		else if (key == 27) break;
	}

	stop_flag = true;

	std::cout << "Wait..." << std::endl;

	readerThread.join();
	CloseSocket();
	delete[] frame;

	std::cout << "Complited" << std::endl;

    return 0;
}
