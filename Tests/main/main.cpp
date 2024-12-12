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
#include <gst/rtsp-server/rtsp-server.h>

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


typedef struct
{
  gboolean white;
  GstClockTime timestamp;
} MyContext;

/* called when we need to give data to appsrc */
static void
need_data (GstElement * appsrc, guint unused, MyContext * ctx)
{
	std::unique_lock<std::mutex> lock(procMtx);
	procCond.wait(lock);

	GstBuffer *buffer;
  	guint size;
  	GstFlowReturn ret;

  	size = WIDTH * HEIGHT * 2;

  	buffer = gst_buffer_new_allocate (NULL, size, NULL);

  /* this makes the image black/white */
  	gst_buffer_memcmp (buffer, 0, frame, size);
	//gst_buffer_memset (buffer, 0, ctx->white ? 0xff : 0x00, size);

  	ctx->white = !ctx->white;

  /* increment the timestamp every 1/2 second */
	GST_BUFFER_PTS (buffer) = ctx->timestamp;
	GST_BUFFER_DURATION (buffer) = gst_util_uint64_scale_int (1, GST_SECOND, 50);
	ctx->timestamp += GST_BUFFER_DURATION (buffer);

	g_signal_emit_by_name (appsrc, "push-buffer", buffer, &ret);
	gst_buffer_unref (buffer);

	// cv::Mat image_cv(HEIGHT, WIDTH, CV_16UC1, frame);
	// imshow("Original", image_cv);
	// cv::waitKey(1);
	framesCnt += 1;
	if (framesCnt == 30)
		CalcFps();
}


static void media_configure (GstRTSPMediaFactory * factory, GstRTSPMedia * media,
    gpointer user_data)
{
	GstElement *element, *appsrc;
	MyContext *ctx;

	/* get the element used for providing the streams of the media */
	element = gst_rtsp_media_get_element (media);

	/* get our appsrc, we named it 'mysrc' with the name property */
	appsrc = gst_bin_get_by_name_recurse_up (GST_BIN (element), "mysrc");

	/* this instructs appsrc that we will be dealing with timed buffer */
	gst_util_set_object_arg (G_OBJECT (appsrc), "format", "time");
	/* configure the caps of the video */
	g_object_set (G_OBJECT (appsrc), "caps",
		gst_caps_new_simple ("video/x-raw",
			"format", G_TYPE_STRING, "GRAY16_BE",
			"width", G_TYPE_INT, WIDTH,
			"height", G_TYPE_INT, HEIGHT,
			"framerate", GST_TYPE_FRACTION, 50, 1, NULL), NULL);

	ctx = g_new0 (MyContext, 1);
	ctx->white = FALSE;
	ctx->timestamp = 0;
	/* make sure ther datais freed when the media is gone */
	g_object_set_data_full (G_OBJECT (media), "my-extra-data", ctx,
		(GDestroyNotify) g_free);

	/* install the callback that will be called when a buffer is needed */
	g_signal_connect (appsrc, "need-data", (GCallback) need_data, ctx);
	gst_object_unref (appsrc);
	gst_object_unref (element);
}

int main(int argc, char *argv[]) 
{
	InitializeSocket(new int[4] { 192, 168, 0, 1 }, 50016);
	InitializeCamera(new int[4] { 192, 168, 0, 15 }, 50000);

    std::thread readerThread(&ReaderLoop);

	  // Initialize GStreamer
    gst_init(&argc, &argv);

    // Create an RTSP server
    GstRTSPServer *server = gst_rtsp_server_new();
    gst_rtsp_server_set_address(server, "127.0.0.1"); // Set the server IP

    // Create a mount point for the RTSP stream
    GstRTSPMountPoints *mounts = gst_rtsp_server_get_mount_points(server);
    GstRTSPMediaFactory *factory = gst_rtsp_media_factory_new();

    // Configure the media factory with a pipeline to loop a video
    gst_rtsp_media_factory_set_launch(factory,
        "( appsrc name=mysrc ! videoconvert ! x264enc ! rtph264pay name=pay0 pt=96  )");

	g_signal_connect (factory, "media-configure", (GCallback) media_configure, NULL);

    // Add the mount point
    gst_rtsp_mount_points_add_factory(mounts, "/test", factory);
    g_object_unref(mounts);

    // Start the RTSP server
    gst_rtsp_server_attach(server, NULL);
    g_print("Stream ready at rtsp://127.0.0.1:8554/test\n");

    // Run the main loop
    g_main_loop_run(g_main_loop_new(NULL, FALSE));
    return 0;

	// GMainLoop *loop;
	// GstRTSPServer *server;
	// GstRTSPMountPoints *mounts;
	// GstRTSPMediaFactory *factory;

	// gst_init (&argc, &argv);

	// loop = g_main_loop_new (NULL, FALSE);

	// /* create a server instance */
	// server = gst_rtsp_server_new ();

	// /* get the mount points for this server, every server has a default object
	// * that be used to map uri mount points to media factories */
	// mounts = gst_rtsp_server_get_mount_points (server);

	// /* make a media factory for a test stream. The default media factory can use
	// * gst-launch syntax to create pipelines.
	// * any launch line works as long as it contains elements named pay%d. Each
	// * element with pay%d names will be a stream */
	// factory = gst_rtsp_media_factory_new ();
	// gst_rtsp_media_factory_set_launch (factory,
	// 	"( appsrc name=mysrc ! videoconvert ! x264enc ! rtph264pay name=pay0 pt=96 )");

	// /* notify when our media is ready, This is called whenever someone asks for
	// * the media and a new pipeline with our appsrc is created */
	// g_signal_connect (factory, "media-configure", (GCallback) media_configure,
	// 	NULL);

	// /* attach the test factory to the /test url */
	// gst_rtsp_mount_points_add_factory (mounts, "/test", factory);

	// /* don't need the ref to the mounts anymore */
	// g_object_unref (mounts);

	// /* attach the server to the default maincontext */
	// gst_rtsp_server_attach (server, NULL);

	// /* start serving */
	// g_print ("stream ready at rtsp://127.0.0.1:8554/test\n");
	// g_main_loop_run (loop);

	return 0;

	// // std::thread processorThread(&ProcessLoop);

	// // init_image_processor(frameOut);

    // std::cout << "Start reading" << std::endl;

	// int key = 0;
	// int fs_cnt = 0;
	// startTime = std::chrono::system_clock::now();
	// while (true)
	// {
	// 	// std::unique_lock<std::mutex> lock(visMtx);
	// 	// visCond.wait(lock);

	// 	std::unique_lock<std::mutex> lock(procMtx);
	// 	procCond.wait(lock);

	// 	if (fs_cnt != 0) 
	// 	{
	// 		if (fs_cnt == 1)
	// 			calc_fs(frame, fs_cnt, true);
	// 		calc_fs(frame, fs_cnt, false);
	// 		fs_cnt += 1;
	// 		if (fs_cnt > 16)
	// 			fs_cnt = 0;
	// 	}

	// 	framesCnt += 1;
	// 	if (framesCnt == 30)
	// 		CalcFps();

	// 	// cv::Mat image_cv(HEIGHT, WIDTH, CV_16UC1, frame);
	// 	// cv::Mat image_cv_out(HEIGHT, WIDTH, CV_16UC1, frameOut);

	// 	// imshow("Original", image_cv);
	// 	// imshow("Processed", image_cv_out);

	// 	// key = cv::waitKey(1);
	// 	// if (key == 99) {
	// 	// 	int packet_data[6] = { 0x05, 0x5C, 0xCF, 0x00, 0x00, 0x00 };
	// 	// 	SendData(packet_data, 6); // Команда на калибровку 
	// 	// }
	// 	// if (key == 98) {
	// 	// 	int packet_data[6] = { 0x05, 0x5C, 0x00, 0x00, 0x0E, 0x80 };
	// 	// 	SendData(packet_data, 6); // Команда на Прямой проход 
	// 	// }
	// 	// if (key == 97) {
	// 	// 	SendFrequence(25);
	// 	// 	SendIntTime(25000);
	// 	// }
	// 	// if (key == 100) {
	// 	// 	fs_cnt = 1;
	// 	// }
	// 	// else if (key == 27) break;
	// }

	// stop_flag = true;

	// std::cout << "Wait..." << std::endl;

	// readerThread.join();
	// CloseSocket();
	// delete[] frame;

	// std::cout << "Complited" << std::endl;

    // return 0;
}
