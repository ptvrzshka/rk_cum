#define PLATFORM_WINDOWS  1
#define PLATFORM_MAC      2
#define PLATFORM_UNIX     3

#if defined(_WIN32)
#define PLATFORM PLATFORM_WINDOWS
#elif defined(__APPLE__)
#define PLATFORM PLATFORM_MAC
#else
#define PLATFORM PLATFORM_UNIX
#endif


#include <iostream>
#include <cstring>
#include <unistd.h>
#include <algorithm>

#if PLATFORM == PLATFORM_WINDOWS

#include <winsock2.h>

#elif PLATFORM == PLATFORM_MAC || PLATFORM == PLATFORM_UNIX

#include <sys/socket.h>
#include <netinet/in.h>
#include <fcntl.h>

#endif

#if PLATFORM == PLATFORM_WINDOWS
#pragma comment( lib, "wsock32.lib" )
#endif


static int handle;
static sockaddr_in address;
static sockaddr_in camera;

static const int WIDTH = 640;
static const int HEIGHT = 512;

static int rowSize = sizeof(unsigned short) * 642;
#if PLATFORM == PLATFORM_WINDOWS
static int cameraLength;
#else
socklen_t cameraLength;
#endif

static const float baseFreq = 29.5;
static const int lineWidth = 944;

static unsigned short* frameBuffer = new unsigned short[WIDTH * HEIGHT];
static unsigned short* outFrame = new unsigned short[WIDTH * HEIGHT];


bool InitializeSocket(int* ip, int port)
{	
#if PLATFORM == PLATFORM_WINDOWS
	WSADATA WsaData;
	if (WSAStartup(MAKEWORD(2, 2), &WsaData) != NO_ERROR) 
		return false;
#endif

	handle = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

	if (handle <= 0)
		return false;

	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(((uint32_t)ip[0] << 24)
		| ((uint32_t)ip[1] << 16)
		| ((uint32_t)ip[2] << 8)
		| ip[3]);
	address.sin_port = htons(port);


	if (bind(handle, (const sockaddr*)&address, sizeof(sockaddr_in)) < 0)
		return false;

	return true;
}


void InitializeCamera(int* ip, int port)
{
	camera.sin_family = AF_INET;
	camera.sin_addr.s_addr = ntohl(((uint32_t)ip[0] << 24)
		| ((uint32_t)ip[1] << 16)
		| ((uint32_t)ip[2] << 8)
		| ip[3]);
	camera.sin_port = ntohs(port);
	cameraLength = sizeof(camera);
}


void CloseSocket() 
{
#if PLATFORM == PLATFORM_WINDOWS
	closesocket(handle);
	WSACleanup();
#else
	close(handle);
#endif
}

int SendData(int* message, int size)
{
	char buffer[size];
	for (int i = 0; i < size; ++i) buffer[i] = message[i];
	int bytesOut = sendto(handle, buffer, size, 0, (sockaddr*)&camera, sizeof(sockaddr));
#if PLATFORM == PLATFORM_WINDOWS
	Sleep(1);
#else
	usleep(1000);
#endif
	return bytesOut;
}


void SendFrequence(int value) 
{
	int data[6] = { 0x05, 0x5C, 0x00, 0x00, 0x00, 0x00 };
	int lines = (int)(1000000.0 * baseFreq / (float)value / lineWidth);
	data[4] = 0x0C;
	data[5] = (lines / 256) | 0x80; // 0x80 - отключить автомат, 0x00 - включить автомат
	SendData(data, 6);
	data[4] = 0x0D;
	data[5] = lines % 256;
	SendData(data, 6);
}


void SendIntTime(int value)
{
	int data[6] = { 0x05, 0x5C, 0x00, 0x00, 0x00, 0x00 };
	int lines = (int)(1000000.0 * baseFreq / (float)value / lineWidth);
	int outTime = (int)(lines - baseFreq * (float)value / lineWidth);
	data[4] = 0x04;
	data[5] = (outTime / 256) | 0x80; // 0x80 - отключить автомат, 0x00 - включить автомат
	SendData(data, 6);
	data[4] = 0x05;
	data[5] = outTime % 256;
	SendData(data, 6);
}


unsigned short* GetFramebuffer() 
{
	char buffer[1284];
	unsigned short ushortsIn[WIDTH + 2];

	while (true)
	{
		int bytesIn = recvfrom(handle, buffer, 1284, 0, (sockaddr*)&camera, &cameraLength);

		std::memcpy(ushortsIn, buffer, rowSize);

		if (ushortsIn[0] <= HEIGHT - 1) 
			std::copy_n(ushortsIn + 2, WIDTH, frameBuffer + WIDTH * ushortsIn[0]);

		if (ushortsIn[0] == HEIGHT - 1) 
			break;
	}

	std::copy_n(frameBuffer, WIDTH * HEIGHT, outFrame);
	return outFrame;
}
