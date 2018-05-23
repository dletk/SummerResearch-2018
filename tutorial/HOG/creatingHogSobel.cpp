#include<opencv2/opencv.hpp>
#include<iostream>
#include<stdio.h>

using namespace cv;
using namespace std;

void drawPeople(Mat image, vector<Rect> foundLocations) {
	for(Rect rect: foundLocations) {
		rectangle(image, Point(rect.x, rect.y), Point(rect.x+rect.width, rect.y+rect.height), Scalar(255,0,0));
	}
	
	imshow("Image", image);
	waitKey(1);
}

int main(int argc, char** argv) {
	Mat image = imread(argv[1]);
	// image.convertTo(image, CV_32F);
	
	const char* gst =  "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,\
			format=(string)I420, framerate=(fraction)120/1 ! \
			nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! \
			videoconvert ! video/x-raw, format=(string)BGR ! \
			appsink";

	VideoCapture vidCap(gst);
	
	Mat gx, gy;
	
	// Calculate the gradient in x and y direction
	Sobel(image, gx, CV_32F, 1, 0, 1);
	Sobel(image, gy, CV_32F, 0, 1, 1);
	
	// Conver the gradient to mangitude and direction, which means its is from cartesian to polar coordinate
	Mat magnitude, direction;
	
	cartToPolar(gx, gy, magnitude, direction, true);
	
	imshow("Magnitude", magnitude);
	imshow("GX", gx);
	imshow("GY", gy);
	waitKey(0);
	destroyAllWindows();
	
	while(true) {
		vidCap.read(image);
	
		// Create the dectector and result vector
		HOGDescriptor detector = HOGDescriptor();
		vector<Rect> foundLocations;
	
		detector.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	
		// Detect the object and store the location of objects into vector<rect>
		detector.detectMultiScale(image, foundLocations, 0, Size(8,8), Size(32,32), 1.05, 2, false);
	
		drawPeople(image, foundLocations);
	}
	
	return 0;
}
