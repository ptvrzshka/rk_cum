#include <iostream>

#include "../OpenCL/opencl.cpp"
#include "../Ethernet/ethernet.cpp"

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
#include <gst/rtsp-server/rtsp-server.h>

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
		visCond.notify_one();

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

// Глобальная структура данных
typedef struct
{
	GstBuffer *buffer;
	GstClockTime timestamp;
} MyContext;

// Callback для предоставления новых данных
static void need_data(GstElement * appsrc, guint unused, MyContext * ctx) {

	std::unique_lock<std::mutex> lock(visMtx);
	visCond.wait(lock);

	// if (procQueue.empty()) 
	// {
	// 	std::cout << "procQueue is empty" << std::endl;
	// 	return;
	// }

	guint size;
	GstFlowReturn ret;

	unsigned char* frame = procQueue.front();

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

	g_signal_emit_by_name (appsrc, "push-buffer", ctx->buffer, &ret);
	gst_buffer_unref (ctx->buffer);
}

static void media_configure (GstRTSPMediaFactory * factory, GstRTSPMedia * media,
    gpointer user_data)
{
	calib_cnt = 1;

	GstElement *element, *appsrc;
	MyContext *ctx;

	element = gst_rtsp_media_get_element (media);

	appsrc = gst_bin_get_by_name_recurse_up (GST_BIN (element), "mysrc");

	gst_util_set_object_arg (G_OBJECT (appsrc), "format", "time");
	g_object_set (G_OBJECT (appsrc), "caps",
		 gst_caps_new_simple(
			"video/x-raw",
			"format", G_TYPE_STRING, "GRAY8",
			"width", G_TYPE_INT, 640,                      // Øèðèíà âèäåî
			"height", G_TYPE_INT, 512,                     // Âûñîòà âèäåî
			"framerate", GST_TYPE_FRACTION, frameRate, 1,         // ×àñòîòà êàäðîâ (30 FPS)
			NULL), NULL);

	ctx = g_new0 (MyContext, 1);
	ctx->buffer = nullptr;
	ctx->timestamp = 0;
	g_object_set_data_full (G_OBJECT (media), "my-extra-data", ctx,
		(GDestroyNotify) g_free);

	g_signal_connect (appsrc, "need-data", (GCallback) need_data, ctx);
	gst_object_unref (appsrc);
	gst_object_unref (element);
}

int main(int argc, char *argv[]) {

	InitializeSocket(new int[4] { 192, 168, 0, 1}, 50016);
	InitializeCamera(new int[4] { 192, 168, 0, 15}, 50000);

	init_image_processor();

	std::thread readerThread(&ReaderLoop);
	std::thread processorThread(&ProcessLoop);

	GMainLoop *loop;
	GstRTSPServer *server;
	GstRTSPMountPoints *mounts;
	GstRTSPMediaFactory *factory;

	gst_init (&argc, &argv);

	loop = g_main_loop_new (NULL, FALSE);

	server = gst_rtsp_server_new ();
    gst_rtsp_server_set_address(server, "192.168.88.26"); // Set the server IP

	mounts = gst_rtsp_server_get_mount_points (server);

	factory = gst_rtsp_media_factory_new ();
	gst_rtsp_media_factory_set_launch (factory,
		"( appsrc name=mysrc max-latency=0 max-lateness=0 is-live=true buffer-mode=none ! videoconvert ! mpph264enc ! rtph264pay name=pay0 pt=96 )");

	g_signal_connect (factory, "media-configure", (GCallback) media_configure,
		NULL);

	gst_rtsp_mount_points_add_factory (mounts, "/test", factory);

	g_object_unref (mounts);

	gst_rtsp_server_attach (server, NULL);

	g_print ("stream ready at rtsp://192.168.88.26:8554/test\n");
	g_main_loop_run (loop);

	return 0;
}
