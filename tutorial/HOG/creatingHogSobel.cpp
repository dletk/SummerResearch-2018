#include<opencv2/opencv.hpp>
#include<opencv2/cudaobjdetect.hpp>

#include<iostream>
#include<string>
#include<stdio.h>
#include<chrono>

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
	bool isUsingCuda = false;
	list<double> totalRunningTime;
	// image.convertTo(image, CV_32F);
	
	if (argc == 3) {
		if (strcmp(argv[2], "cuda") == 0) {
			isUsingCuda = true;
			cout << "===> Using CUDA" << endl;
		}
	}
	
	const char* gst =  "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,\
			format=(string)I420, framerate=(fraction)30/1 ! \
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
	
	Ptr<cuda::HOG> cudaDetector;
	HOGDescriptor detector;
	cuda::GpuMat cudaImage;
	
	// Create the dectector and result vector
	if (!isUsingCuda) {
		detector = HOGDescriptor(Size(64,128), // winSize
		 Size(16,16), // blockSize
		 Size(8,8), // blockStride
		 Size(8,8), // cellSize
		 9, // nbins
		 1 ); //deriveAperture
	} else {	
		// Initialize cuda
		cuda::GpuMat test;
		test.create(1, 1, CV_8U);
		test.release();

		cudaDetector = cuda::HOG::create();
		
		
	}
	
	
	vector<float> svmDetector;	
	
	if (isUsingCuda) {
		svmDetector = cudaDetector->getDefaultPeopleDetector();
		cudaDetector->setSVMDetector(svmDetector);
		
		// Set up cuda detector parameters
		cudaDetector->setNumLevels(64);
        cudaDetector->setHitThreshold(0);
        cudaDetector->setWinStride(Size(8,8));
        cudaDetector->setScaleFactor(1.05);
        //cudaDetector->setGroupThreshold(8);
		
	} else {
		 svmDetector = HOGDescriptor::getDefaultPeopleDetector();
		 detector.setSVMDetector(svmDetector);
	}
		
	vector<Rect> foundLocations;
	
	Mat tempImage;
	Mat grayImage;

	while(totalRunningTime.size() < 100) {
		vidCap.read(tempImage);
		
		// Begin timing
		auto time1 = chrono::system_clock::now();
		
		// resize the image so the winSize 64,128 can be used effectively
		resize(tempImage, image, Size(), 0.5, 0.5);
		auto time2 = chrono::system_clock::now();
		
		// Detect the object and store the location of objects into vector<rect>
		if (isUsingCuda) {
			cvtColor(image, grayImage, COLOR_BGR2GRAY);
			cudaImage.upload(grayImage);
			cudaDetector->detectMultiScale(cudaImage, foundLocations);		
		} else {
			detector.detectMultiScale(image, foundLocations, 0, Size(8,8), Size(32,32), 1.05, 2, false);
		}
		auto time3 = chrono::system_clock::now();
		
		drawPeople(image, foundLocations);
		auto time4 = chrono::system_clock::now();
		
		chrono::duration<double> totalRuntime = time4 - time1;
		chrono::duration<double> resizeTime = time2 - time1;
		chrono::duration<double> detectingTime = time3 - time2;
		chrono::duration<double> drawingTime = time4 - time3;
		
		cout << "============" << endl;
		cout << "Total runtime for a frame: " << totalRuntime.count() << endl;
		cout << "Resize time: " << resizeTime.count() << endl;
		cout << "Detecting time: " << detectingTime.count() << endl;
		cout << "Drawing time: " << drawingTime.count() << endl;
		totalRunningTime.push_back(detectingTime.count());
	}
	
	double avg = 0;
    list<double>::iterator it;
    for(it = totalRunningTime.begin(); it != totalRunningTime.end(); it++) avg += *it;
    avg /= totalRunningTime.size();
	
	cout << "============" << endl;
	cout << "AVERAGE DETECTING TIME: " << avg << endl;
	
	return 0;
}
