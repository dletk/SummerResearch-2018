#include<opencv2/opencv.hpp>
#include<stdio.h>
#include<iostream>
#include<chrono>

using namespace cv;
using namespace std;


int main(int argc, char** argv) {
	//Get the onboard webcam
	const char* gst =  "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,\
			format=(string)I420, framerate=(fraction)120/1 ! \
			nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! \
			videoconvert ! video/x-raw, format=(string)BGR ! \
			appsink";

	VideoCapture vidCap(gst);

	// Create face classifier
	CascadeClassifier faceClassifier = CascadeClassifier();
	faceClassifier.load("/home/nvidia/opencv/data/haarcascades/haarcascade_frontalface_default.xml");
	Mat frame, grayImage;
	vector<Rect> faces;
	
	while(true) {
		vidCap.read(frame);

	
		cvtColor(frame, grayImage, COLOR_BGR2GRAY);
		
	
		// Detect the faces
		auto start = chrono::system_clock::now();
		faceClassifier.detectMultiScale(grayImage, faces, 1.3);
	
		for(Rect rect: faces) {
			rectangle(frame, Point(rect.x, rect.y), Point(rect.x+rect.width, rect.y+rect.height), Scalar(255,0,0));
		}
		auto end = chrono::system_clock::now();
		chrono::duration<double> elapsed_time = end - start;
		
		cout << "Runtime: " << elapsed_time.count() << endl;	
		
		imshow("Image", frame);
		waitKey(1);
	}
	
	return 0;
}
