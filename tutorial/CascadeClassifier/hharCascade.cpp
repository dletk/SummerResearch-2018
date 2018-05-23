#include<opencv2/opencv.hpp>
#include<opencv2/cudaobjdetect.hpp>

#include<stdio.h>
#include<iostream>
#include<chrono>
#include<string>

using namespace cv;
using namespace std;


void drawFaces(Mat frame, vector<Rect> faces) {
	for(Rect rect: faces) {
		rectangle(frame, Point(rect.x, rect.y), Point(rect.x+rect.width, rect.y+rect.height), Scalar(255,0,0));
	}
	
	imshow("Image", frame);
	imwrite("./exampleOut.jpg", frame);
	waitKey(1);
}

int main(int argc, char** argv) {
	//Get the onboard webcam
	const char* gst =  "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,\
			format=(string)I420, framerate=(fraction)120/1 ! \
			nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! \
			videoconvert ! video/x-raw, format=(string)BGR ! \
			appsink";

	VideoCapture vidCap(gst);
	vector<Rect> faces;	
	Mat frame, grayImage;
	
	frame = imread("./example.jpg");
	
	
	if (argc == 2) {

		cuda::GpuMat frameCuda, resultDetected;
		
		cout << "CUDA SELECTED" << endl;

		// Initialize cuda
		cuda::GpuMat test;
		test.create(1, 1, CV_8U);
		test.release();
		
		Ptr<cuda::CascadeClassifier> faceClassifier = cuda::CascadeClassifier::create("/home/nvidia/opencv/data/haarcascades_cuda/haarcascade_fullbody.xml");
		
		cout << "CHECKED" << endl;
		cout << "Number of device: " << cuda::getCudaEnabledDeviceCount() << endl;
		
		while(true) {
			// vidCap.read(frame);
			cvtColor(frame, grayImage, COLOR_BGR2GRAY);
			
			// Upload the frame to cuda
			frameCuda.upload(grayImage);
			
			auto start = chrono::system_clock::now();
			
			// Detect the faces and store into a GpuMat
			faceClassifier->detectMultiScale(frameCuda, resultDetected);
			
			auto end = chrono::system_clock::now();
			
			// Convert the data in the GpuMat to Rectangle to show on screen if there is any
			faceClassifier->convert(resultDetected, faces);
			
				

			chrono::duration<double> elapsed_time = end - start;

			cout << "Runtime: " << elapsed_time.count() << endl;	
			
			drawFaces(frame, faces);
		}
	
	} else {
		// Create face classifier
		CascadeClassifier faceClassifier = CascadeClassifier();
		faceClassifier.load("/home/nvidia/opencv/data/haarcascades/haarcascade_fullbody.xml");

		while(true) {
			// vidCap.read(frame);
			
			cvtColor(frame, grayImage, COLOR_BGR2GRAY);
		
		
			auto start = chrono::system_clock::now();
			
			faceClassifier.detectMultiScale(grayImage, faces, 1.3);
					
			auto end = chrono::system_clock::now();
			chrono::duration<double> elapsed_time = end - start;
			
			cout << "Runtime: " << elapsed_time.count() << endl;	
			
			drawFaces(frame, faces);
		}
	}
	
	return 0;
}
