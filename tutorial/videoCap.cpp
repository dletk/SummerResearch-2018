#include<opencv2/core.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>
#include<stdio.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat frame;
	 	const char* gst =  "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,\
			format=(string)I420, framerate=(fraction)120/1 ! \
			nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! \
			videoconvert ! video/x-raw, format=(string)BGR ! \
			appsink";

	VideoCapture vidCap(gst);

	while(true) {
		vidCap.read(frame);
		imshow("Live", frame);
		waitKey(5);
	}
}

