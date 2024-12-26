#include <iostream>

#include "../OpenCL/opencl.cpp"
#include "../Ethernet/ethernet.cpp"
#include "../Ethernet/tcp_proxy.cpp"

#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>

// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc.hpp>

#include <rk_mpi.h>
#include <rk_type.h>

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>  

#define RTP_PORT 5004
#define HOST "192.168.88.21"

std::mutex procMtx;
std::condition_variable procCond;
std::mutex visMtx;
std::condition_variable visCond;
bool stop_flag = false;

std::queue<unsigned short *> rawQueue;
std::queue<unsigned char *> procQueue;
const int maxQueueSize = 3;

std::chrono::time_point<std::chrono::system_clock> startTime;
std::chrono::time_point<std::chrono::system_clock> endTime;
int framesCnt = 0;
const int frameRate = 50;


void ReaderLoop()
{
	while (!stop_flag)
	{
		if (rawQueue.size() >= maxQueueSize) 
		{
			delete[] rawQueue.front();
			rawQueue.pop();
		}
		unsigned short* frame = new unsigned short[WIDTH * HEIGHT];
		memcpy(frame, GetFramebuffer(), HEIGHT * WIDTH * 2);
		rawQueue.push(frame);
		procCond.notify_one();
	}
}

int calib_cnt = 1;
void ProcessLoop() 
{
	while (!stop_flag)
	{
		std::unique_lock<std::mutex> lock(procMtx);
		procCond.wait(lock);

		if (rawQueue.empty())
			return;

		if (procQueue.size() >= maxQueueSize) 
		{
			delete[] procQueue.front();
			procQueue.pop();
		}
		unsigned char* frameOut = new unsigned char[WIDTH * HEIGHT];
		unsigned short* frame = rawQueue.front();
		if (calib_cnt < 25) 
		{
			if (calib_cnt == 1)
				calc_fs(rawQueue.front(), calib_cnt, true);
			else
				calc_fs(rawQueue.front(), calib_cnt, false);
			calib_cnt += 1;
		}
		process_image(frame, frameOut);
		procQueue.push(frameOut);

		// framesCnt += 1;
		// if (framesCnt == 30)
		// {
		// 	endTime = std::chrono::system_clock::now();
		// 	std::cout << "Fps: " << (float)framesCnt / std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() * 1000.0 << " Queue: " << procQueue.size() << std::endl;
		// 	framesCnt = 0;
		// 	startTime = std::chrono::system_clock::now();
		// }
	}
}

void ProxyLoop() 
{
	while (true)
	{
		unsigned char* data = GetDataFromProxyServer();
		if (data == nullptr)
			continue;
		if (data[0] == 0) 
		{
			calib_cnt = 1;
			continue;
		}
		if (data[0] == 1) 
		{
			global_contrast_changed(data[1]);
			continue;
		}
		if (data[0] == 2) 
		{
			local_contrast_changed(data[1]);
			continue;
		}
		if (data[0] == 3) 
		{
			denoise_1_changed(data[1]);
			continue;
		}
		if (data[0] == 4) 
		{
			denoise_2_changed(data[1]);
			continue;
		}
		SendData(data, 6);
	}	
}

// Глобальная структура данных
typedef struct
{
	GstBuffer *buffer;
	GstClockTime timestamp;
} MyContext;

// Callback для предоставления новых данных
static void need_data(GstElement * appsrc, guint unused, MyContext * ctx) {

	// std::unique_lock<std::mutex> lock(visMtx);
	// visCond.wait(lock);

	// if (procQueue.empty()) 
	// {
	// 	std::cout << "procQueue is empty" << std::endl;
	// 	return;
	// }

	if (procQueue.empty())
		return;
	
	unsigned char* frame = procQueue.front();

	guint size;
	GstFlowReturn ret;

	ctx->buffer = gst_buffer_new_allocate(NULL, WIDTH*HEIGHT, NULL);
	GstMapInfo map;
    gst_buffer_map(ctx->buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, frame, WIDTH * HEIGHT);
    gst_buffer_unmap(ctx->buffer, &map);
		
	/* increment the timestamp every 1/2 second */
	// GST_BUFFER_PTS(ctx->buffer) = ctx->timestamp;
    // GST_BUFFER_DTS(ctx->buffer) = ctx->timestamp;
	// ctx->timestamp += gst_util_uint64_scale_int (1, GST_SECOND, 50);
	GST_BUFFER_PTS (ctx->buffer) = ctx->timestamp;
	GST_BUFFER_DURATION (ctx->buffer) = gst_util_uint64_scale_int (1, GST_SECOND, frameRate);
	ctx->timestamp += GST_BUFFER_DURATION (ctx->buffer);

	g_signal_emit_by_name(appsrc, "push-buffer", ctx->buffer, &ret);
	gst_buffer_unref(ctx->buffer);
}


void push_data_to_appsrc(GstElement* appsrc, guint unused)
 {

	if (procQueue.empty())
		return;

	// std::cout << "kek" << std::endl;

	GstFlowReturn ret;
	
	unsigned char* frame = procQueue.front();

    GstBuffer *buffer = gst_buffer_new_allocate(nullptr, WIDTH * HEIGHT, nullptr);
	GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, frame, WIDTH * HEIGHT);
    gst_buffer_unmap(buffer, &map);

	g_signal_emit_by_name(appsrc, "push-buffer", buffer, &ret);
	gst_buffer_unref(buffer);

	framesCnt += 1;
	if (framesCnt == 30)
	{
		endTime = std::chrono::system_clock::now();
		std::cout << "Fps: " << (float)framesCnt / std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() * 1000.0 << " Queue: " << procQueue.size() << std::endl;
		framesCnt = 0;
		startTime = std::chrono::system_clock::now();
	}
}

GstElement* create_udp_pipeline(const char* host, guint port, GstElement *appsrc) {
    GstElement *pipeline, *convert, *encoder, *payloader, *sink;

    pipeline = gst_pipeline_new("rtp-pipeline");
    convert = gst_element_factory_make("videoconvert", "convert");
    encoder = gst_element_factory_make("x264enc", "encoder");
    payloader = gst_element_factory_make("rtph264pay", "payloader");
    sink = gst_element_factory_make("udpsink", "sink");

    if (!pipeline || !convert || !encoder || !payloader || !sink) {
        std::cerr << "Failed to create one or more elements." << std::endl;
        return nullptr;
    }

    g_object_set(sink, "host", host, "port", port, nullptr);

    gst_bin_add_many(GST_BIN(pipeline), appsrc, convert, encoder, payloader, sink, nullptr);

    if (!gst_element_link_many(appsrc, convert, encoder, payloader, sink, nullptr)) {
        std::cerr << "Failed to link elements." << std::endl;
        return nullptr;
    }
	
	gst_element_set_state(pipeline, GST_STATE_PLAYING);

    return pipeline;
}

int main(int argc, char *argv[]) {

	InitializeSocket(new int[4] { 192, 168, 0, 1}, 50016);
	InitializeCamera(new int[4] { 192, 168, 0, 15}, 50000);
	InitializeProxyServer(50032);

	init_image_processor();

	std::thread readerThread(&ReaderLoop);
	std::thread processorThread(&ProcessLoop);
	std::thread proxyThread(&ProxyLoop);

	// SendData(new unsigned char[6] {0x5, 0x5c, 0x00, 0x00, 0x37, 0x1}, 6);
	// SendData(new unsigned char[6] {0x5, 0x5c, 0x00, 0x00, 0xe, 0x80}, 6);

	GstElement *pipeline, *source, *sink, *convert, *encoder, *payloader;
	GstStateChangeReturn ret;

    gst_init(&argc, &argv);

	source = gst_element_factory_make ("appsrc", "source");
	convert = gst_element_factory_make ("videoconvert", "convert");
    encoder = gst_element_factory_make ("x264enc", "encoder");
    payloader = gst_element_factory_make ("rtph264pay", "payloader");
  	sink = gst_element_factory_make ("udpsink", "sink");

	pipeline = gst_pipeline_new ("udp-pipeline");

	g_object_set (G_OBJECT (source), "caps",
		gst_caps_new_simple(
		"video/x-raw",
		"format", G_TYPE_STRING, "GRAY8",
		"width", G_TYPE_INT, 640,                      
		"height", G_TYPE_INT, 512,                    
		"framerate", GST_TYPE_FRACTION, frameRate, 1,      
		NULL), NULL);

	gst_bin_add_many (GST_BIN (pipeline), source, sink, convert, encoder, payloader, NULL);
	if (gst_element_link (source, sink) != TRUE) 
	{
		g_printerr ("Elements could not be linked.\n");
		gst_object_unref (pipeline);
		return -1;
	}

	// g_object_set (source, "pattern", 0, NULL);

	ret = gst_element_set_state (pipeline, GST_STATE_PLAYING);
	if (ret == GST_STATE_CHANGE_FAILURE) {
		g_printerr ("Unable to set the pipeline to the playing state.\n");
		gst_object_unref (pipeline);
		return -1;
	}

	g_signal_connect(source, "need-data", G_CALLBACK(push_data_to_appsrc), source);

	std::cout << "UDP Live streaming started..." << std::endl;

    GMainLoop *loop = g_main_loop_new(nullptr, FALSE);
    g_main_loop_run(loop);

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    g_main_loop_unref(loop);


    // GstElement *appsrc = gst_element_factory_make("appsrc", "source");
    // if (!appsrc) {
    //     std::cerr << "Failed to create appsrc element." << std::endl;
    //     return -1;
    // }

    // g_object_set(GST_OBJECT(appsrc), "is-live", TRUE, "format", GST_FORMAT_TIME, nullptr);

    // GstElement *pipeline = create_udp_pipeline(HOST, RTP_PORT, appsrc);
    // if (!pipeline) {
    //     std::cerr << "Failed to create UDP pipeline." << std::endl;
    //     return -1;
    // }

    // g_signal_connect(appsrc, "need-data", G_CALLBACK(push_data_to_appsrc), appsrc);

    // GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    // if (ret == GST_STATE_CHANGE_FAILURE) {
    //     std::cerr << "Failed to set pipeline to PLAYING state." << std::endl;
    //     gst_object_unref(pipeline);
    //     return -1;
    // }

    // std::cout << "UDP Live streaming started..." << std::endl;

    // GMainLoop *loop = g_main_loop_new(nullptr, FALSE);
    // g_main_loop_run(loop);

    // gst_element_set_state(pipeline, GST_STATE_NULL);
    // gst_object_unref(pipeline);
    // g_main_loop_unref(loop);

    return 0;
}
