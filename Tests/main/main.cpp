#include <iostream>
#include <csignal>

#include "../OpenCL/opencl.cpp"
#include "../Ethernet/ethernet.cpp"
#include "../Ethernet/tcp_proxy.cpp"

#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>

#include <rk_mpi.h>
// #include <rk_type.h>

#include <gst/gst.h>
// #include <gst/app/gstappsrc.h>  

#define RTP_PORT 5004
#define HOST "127.0.0.1"

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

typedef struct {
    GstClockTime timestamp;
} StreamContext;
GstElement *source, *convert, *encoder, *muxer, *payloader, *sink;
GstElement *pipeline;
gulong needDataHandlerId;
StreamContext *ctx;


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
// cv::VideoWriter video = cv::VideoWriter("asas41.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 50.0, cv::Size(640, 512), false);
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

		// video.write(cv::Mat(512, 640, CV_8UC1, frameOut));

		framesCnt += 1;
		if (framesCnt == 30)
		{
			endTime = std::chrono::system_clock::now();
			std::cout << "Fps: " << (float)framesCnt / std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() * 1000.0 << " Queue: " << procQueue.size() << std::endl;
			framesCnt = 0;
			startTime = std::chrono::system_clock::now();
		}
	}
}

void need_data(GstElement* appsrc, guint unused, StreamContext* ctx)
{

	if (procQueue.empty())
		return;

	// std::cout << "kek" << std::endl;

	GstFlowReturn ret;
	
	unsigned char* frame = procQueue.front();

	// GstBuffer *buffer = gst_buffer_new_wrapped(frame, WIDTH * HEIGHT);

    GstBuffer *buffer = gst_buffer_new_allocate(nullptr, WIDTH * HEIGHT, nullptr);
	GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, frame, WIDTH * HEIGHT);
    gst_buffer_unmap(buffer, &map);

	GST_BUFFER_PTS (buffer) = ctx->timestamp;
	GST_BUFFER_DURATION (buffer) = gst_util_uint64_scale_int (1, GST_SECOND, frameRate);
	ctx->timestamp += GST_BUFFER_DURATION (buffer);

	// gst_app_src_push_buffer((GstAppSrc*)ctx->appsrc, buffer);

	g_signal_emit_by_name (appsrc, "push-buffer", buffer, &ret);

	gst_buffer_unref(buffer);

	// framesCnt += 1;
	// if (framesCnt == 30)
	// {
	// 	endTime = std::chrono::system_clock::now();
	// 	std::cout << "Fps: " << (float)framesCnt / std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() * 1000.0 << " Queue: " << procQueue.size() << std::endl;
	// 	framesCnt = 0;
	// 	startTime = std::chrono::system_clock::now();
	// }
}

void StopRtpLoop() 
{
	std::cout << "Stoppp" << std::endl;
	g_signal_handler_disconnect(source, needDataHandlerId);
	gst_element_set_state (pipeline, GST_STATE_PAUSED);
	// gst_element_set_state(pipeline, GST_STATE_NULL);
    // g_free(ctx);
    // g_free(loop);
	// rtpThread->join();
}

void RunRtpLoop()
{
	std::cout << "Starttt" << std::endl;
	gst_element_set_state (pipeline, GST_STATE_PLAYING);
    needDataHandlerId = g_signal_connect (source, "need-data", (GCallback) need_data, ctx);
}

static StreamContext *stream_context_new()
{
    StreamContext *ctx = g_new0(StreamContext, 1);
    ctx->timestamp = 0;
    return ctx;
}

void ProxyLoop() 
{
	while (true)
	{
		unsigned char* data = GetDataFromProxyServer();
		if (data == nullptr)
			continue;
		if (data[0] == 0xff) 
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
		if (data[0] == 6) 
		{
			if (data[1] == 1)
				StopRtpLoop();
			if (data[1] == 0)
				RunRtpLoop();
			continue;
		}
		if (data[0] == 7) 
		{
			dde_changed(data[1]);
			continue;
		}
		if (data[0] == 8) 
		{
			adaptive_changed(data[1]);
			continue;
		}
		SendData(data, 6);
	}	
}

void cleanup(int signal) 
{
	std::cout << "Exit..." << std::endl;

	CloseSocket();
	CloseProxy();

	StopRtpLoop();
    g_free(ctx);

	std::cout << "Completed" << std::endl;
}

int main(int argc, char *argv[]) 
{
	struct sigaction sigIntHandler;
   	sigIntHandler.sa_handler = cleanup;
   	sigemptyset(&sigIntHandler.sa_mask);
   	sigIntHandler.sa_flags = 0;
   	sigaction(SIGINT, &sigIntHandler, NULL);

	InitializeSocket(new int[4] { 192, 168, 0, 1}, 50016);
	InitializeCamera(new int[4] { 192, 168, 0, 15}, 50000);
	InitializeProxyServer(50032);

	init_image_processor();

	std::thread readerThread(&ReaderLoop);
	std::thread processorThread(&ProcessLoop);
	std::thread proxyThread(&ProxyLoop);

	SendData(new unsigned char[6] {0x5, 0x5c, 0x00, 0x00, 0x37, 0x1}, 6);
	SendData(new unsigned char[6] {0x5, 0x5c, 0x00, 0x00, 0xe, 0x80}, 6);

	gst_init(&argc, &argv);

	// GstElement *source, *filter, *convert, *encoder, *payloader, *sink;

	source = gst_element_factory_make ("appsrc", "source");
	// filter = gst_element_factory_make("capsfilter", "filter");
	convert = gst_element_factory_make ("videoconvert", "convert");
    encoder = gst_element_factory_make ("mpph264enc", "encoder");
	muxer = gst_element_factory_make ("mpegtsmux", "muxer");
    payloader = gst_element_factory_make ("rtpmp2tpay", "payloader");
  	sink = gst_element_factory_make ("udpsink", "sink");
	// queue = gst_element_factory_make ("queue", "queue");

	pipeline = gst_pipeline_new ("test-pipeline");

	g_object_set(G_OBJECT(source), "stream-type", 0, "format", GST_FORMAT_TIME, NULL);
	g_object_set(encoder, "header-mode", 1, NULL);
	// g_object_set(payloader, "config-interval", -1, NULL);
	// g_object_set(payloader, "source-info", true, NULL);
	// g_object_set(payloader, "pt", 96, NULL);
	g_object_set(sink, "host", "192.168.88.255", "port", 5004, NULL);

	g_object_set (G_OBJECT (source), "caps",
		gst_caps_new_simple(
		"video/x-raw",
		"format", G_TYPE_STRING, "GRAY8",
		"width", G_TYPE_INT, 640,                      
		"height", G_TYPE_INT, 512,                    
		"framerate", GST_TYPE_FRACTION, frameRate, 1,      
		NULL), NULL);

    // g_object_set(filter, "caps", 
	// gst_caps_new_simple(
	// 	"video/x-raw",
	// 	"format", G_TYPE_STRING, "GRAY8",
	// 	"width", G_TYPE_INT, 640,                      
	// 	"height", G_TYPE_INT, 512,                    
	// 	"framerate", GST_TYPE_FRACTION, frameRate, 1,      
	// 	NULL), NULL);

	gst_bin_add_many (GST_BIN (pipeline), source, convert, encoder, muxer, payloader, sink, NULL);
	gst_element_link_many(source, convert, encoder, muxer, payloader, sink, NULL);

    ctx = stream_context_new();

	// gst_element_set_state (pipeline, GST_STATE_PLAYING);
    // g_signal_connect (source, "need-data", (GCallback) need_data, ctx);
	RunRtpLoop();

	while (1)
	{
		// std::cout << "lul" << std::endl;
		sleep(1);
	}
 
    // gst_element_set_state(pipeline, GST_STATE_NULL);
 
    // g_free(ctx);
    // g_free(loop);
 
    return 0;
}



















// // Callback для предоставления новых данных
// static void need_data(GstElement * appsrc, guint unused, MyContext * ctx) {

// 	// std::unique_lock<std::mutex> lock(visMtx);
// 	// visCond.wait(lock);

// 	// if (procQueue.empty()) 
// 	// {
// 	// 	std::cout << "procQueue is empty" << std::endl;
// 	// 	return;
// 	// }

// 	if (procQueue.empty())
// 		return;
	
// 	unsigned char* frame = procQueue.front();

// 	guint size;
// 	GstFlowReturn ret;

// 	ctx->buffer = gst_buffer_new_allocate(NULL, WIDTH*HEIGHT, NULL);
// 	GstMapInfo map;
//     gst_buffer_map(ctx->buffer, &map, GST_MAP_WRITE);
//     memcpy(map.data, frame, WIDTH * HEIGHT);
//     gst_buffer_unmap(ctx->buffer, &map);
		
// 	/* increment the timestamp every 1/2 second */
// 	// GST_BUFFER_PTS(ctx->buffer) = ctx->timestamp;
//     // GST_BUFFER_DTS(ctx->buffer) = ctx->timestamp;
// 	// ctx->timestamp += gst_util_uint64_scale_int (1, GST_SECOND, 50);
// 	GST_BUFFER_PTS (ctx->buffer) = ctx->timestamp;
// 	GST_BUFFER_DURATION (ctx->buffer) = gst_util_uint64_scale_int (1, GST_SECOND, frameRate);
// 	ctx->timestamp += GST_BUFFER_DURATION (ctx->buffer);

// 	g_signal_emit_by_name (appsrc, "push-buffer", ctx->buffer, &ret);
// 	gst_buffer_unref (ctx->buffer);
// }

// static void media_configure (GstRTSPMediaFactory * factory, GstRTSPMedia * media,
//     gpointer user_data)
// {
// 	GstElement *element, *appsrc;
// 	MyContext *ctx;

// 	element = gst_rtsp_media_get_element (media);

// 	appsrc = gst_bin_get_by_name_recurse_up (GST_BIN (element), "mysrc");

// 	gst_util_set_object_arg (G_OBJECT (appsrc), "format", "time");
// 	g_object_set (G_OBJECT (appsrc), "caps",
// 		 gst_caps_new_simple(
// 			"video/x-raw",
// 			"format", G_TYPE_STRING, "GRAY8",
// 			"width", G_TYPE_INT, 640,                      
// 			"height", G_TYPE_INT, 512,                    
// 			"framerate", GST_TYPE_FRACTION, frameRate, 1,      
// 			NULL), NULL);

// 	ctx = g_new0 (MyContext, 1);
// 	ctx->buffer = nullptr;
// 	ctx->timestamp = 0;
// 	g_object_set_data_full (G_OBJECT (media), "my-extra-data", ctx,
// 		(GDestroyNotify) g_free);

// 	g_signal_connect (appsrc, "need-data", (GCallback) need_data, ctx);
// 	gst_object_unref (appsrc);
// 	gst_object_unref (element);
// }

// int main(int argc, char *argv[]) {

// 	InitializeSocket(new int[4] { 192, 168, 0, 1}, 50016);
// 	InitializeCamera(new int[4] { 192, 168, 0, 15}, 50000);
// 	InitializeProxyServer(50032);

// 	init_image_processor();

// 	// SendData(new unsigned char[6] {0x5, 0x5c, 0x00, 0x00, 0x37, 0x1}, 6);
// 	// SendData(new unsigned char[6] {0x5, 0x5c, 0x00, 0x00, 0xe, 0x80}, 6);

// 	std::thread readerThread(&ReaderLoop);
// 	std::thread processorThread(&ProcessLoop);
// 	std::thread proxyThread(&ProxyLoop);

// 	GMainLoop *loop;
// 	GstRTSPServer *server;
// 	GstRTSPMountPoints *mounts;
// 	GstRTSPMediaFactory *factory;

// 	gst_init (&argc, &argv);

// 	loop = g_main_loop_new (NULL, FALSE);

// 	server = gst_rtsp_server_new ();
//     gst_rtsp_server_set_address(server, "192.168.88.21"); // Set the server IP

// 	mounts = gst_rtsp_server_get_mount_points (server);

// 	factory = gst_rtsp_media_factory_new ();
// 	gst_rtsp_media_factory_set_launch (factory,
// 		"( appsrc name=mysrc max-latency=0 max-lateness=0 is-live=true buffer-mode=none ! videoconvert ! mpph264enc ! rtph264pay name=pay0 pt=96 )");

// 	g_signal_connect (factory, "media-configure", (GCallback) media_configure,
// 		NULL);

// 	gst_rtsp_mount_points_add_factory (mounts, "/test", factory);

// 	g_object_unref (mounts);

// 	gst_rtsp_server_attach (server, NULL);

// 	g_print ("stream ready at rtsp://192.168.88.26:8554/test\n");
// 	g_main_loop_run (loop);

// 	return 0;
// }

