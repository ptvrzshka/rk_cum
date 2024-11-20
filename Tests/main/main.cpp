#include <iostream>

#include "../OpenCL/opencl.c"
#include "../Ethernet/ethernet.cpp"

#include <mutex>
#include <condition_variable>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>


std::mutex mtx;
std::condition_variable cond;
bool stop_flag = false;

unsigned short* frame = nullptr;
unsigned short* frameOut = new unsigned short[WIDTH * HEIGHT];

void ReaderLoop()
{
	while (!stop_flag)
	{
		frame = GetFramebuffer(); 
		cond.notify_one();
	}
}


int main() 
{
    InitializeSocket(new int[4] { 192, 168, 0, 1 }, 50016);
	InitializeCamera(new int[4] { 192, 168, 0, 15 }, 50000);

    std::thread readerThread(&ReaderLoop);

	init_image_processor(frameOut);

    std::cout << "Start reading" << std::endl;

	int key = 0;
	while (true)
	{
		std::unique_lock<std::mutex> lock(mtx);
		cond.wait(lock);

		process_image(frame, frameOut);

		std::cout << frame[10000] << " " << frameOut[10000] << std::endl;

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
