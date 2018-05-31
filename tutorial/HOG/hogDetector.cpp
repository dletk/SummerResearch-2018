#include<opencv2/opencv.hpp>
#include<opencv2/ml.hpp>
#include<opencv2/cudaobjdetect.hpp>

#include<algorithm>
#include<iostream>
#include<string>
#include<stdio.h>
#include<chrono>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include<dlib/opencv.h>


using namespace cv;
using namespace std;

dlib::image_window win;
dlib::shape_predictor pose_model;

void findLandmarks(Mat image, vector<Rect> faces) {
	// Convert the openCV image to dlib image
	dlib::cv_image<dlib::bgr_pixel> cvImage(image);
	vector<dlib::full_object_detection> shapes;
	
	for (Rect rect: faces) {
		// Convert the bounding box in cv::Rect to dlib::rectangle
		dlib::rectangle face(rect.x, rect.y, rect.x+rect.width, rect.y+rect.height);
		shapes.push_back(pose_model(cvImage, face));
	}
	
	 // Display it all on the screen
    win.clear_overlay();
    win.set_image(cvImage);
    win.add_overlay(render_face_detections(shapes));
}

// Method to draw a rectangle box around the detected object on the image
void drawPeople(Mat image, vector<Rect> foundLocations, vector<double> confidences, double max) {
	int i = 0;
	for(Rect rect: foundLocations) {
		if (!confidences.empty()) {
			cout << confidences.at(i) << endl;
			// Using 0.05 to have some tolerance
			if (confidences.at(i) == max) {
				rectangle(image, Point(rect.x, rect.y), Point(rect.x+rect.width, rect.y+rect.height), Scalar(255,0,0));
			}
			i++;
		} else {
			rectangle(image, Point(rect.x, rect.y), Point(rect.x+rect.width, rect.y+rect.height), Scalar(255,0,0));
		}
	}
	
	findLandmarks(image, foundLocations);
	
	// imshow("Image", image);
	// We are using camera stream, so the waitTime is set to 1. Change it to 0 for a static image.
	// waitKey(1);
}

int main(int argc, char** argv) {
	// The flag to trigger CUDA if needed
	bool isUsingCuda = false;
	
	double scaleFactor = 1.2;
	
	// Dlib facial landmark model
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

	
	// The list to keep track of all the recorded runtime to calculate average runtime if needed
	list<double> totalRunningTime;
	
	// Check if CUDA is being used
	if (argc == 3) {
		if (strcmp(argv[1], "cuda") == 0) {
			isUsingCuda = true;
			cout << "===> Using CUDA" << endl;
		} else {
			cout << "Wrong input, usage: ./hogDetector <cuda>" << endl;
			return 0;
		}
	} else if (argc > 3 || argc == 1) {
		cout << "Wrong input, usage: ./hogDetector <cuda>" << endl;
		return 0;
	}
	
	// Setting up the onboard webcam of Jetson TX2
	const char* gst =  "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,\
			format=(string)I420, framerate=(fraction)30/1 ! \
			nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! \
			videoconvert ! video/x-raw, format=(string)BGR ! \
			appsink";
	VideoCapture vidCap(1);
	
	// Prepare different versions of the HOG detector because CUDA and sequential code have different types
	Ptr<cuda::HOG> cudaDetector;
	HOGDescriptor detector;
	
	// Create the dectector and result vector
	if (!isUsingCuda) {
		detector = HOGDescriptor(Size(128,192), // winSize
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

		cudaDetector = cuda::HOG::create(Size(128,192));		
	}
	
	
	// Prepare the detector for the HOG feature descriptor

	if (isUsingCuda) {
    	detector.load(argv[2]);
    } else {
    	detector.load(argv[1]);
    }
	vector<float> svmDetector = detector.svmDetector;
	
	cout << "LOADED SVM" << endl;
	
	if (isUsingCuda) {
		//svmDetector = cudaDetector->getDefaultPeopleDetector();
		cudaDetector->setSVMDetector(svmDetector);
		
		// Set up cuda detector parameters
		cudaDetector->setNumLevels(64);
        cudaDetector->setHitThreshold(0);
        cudaDetector->setWinStride(Size(8,8));
        cudaDetector->setScaleFactor(scaleFactor);
        
        // Setting the group threshold means that the detector will group overlapping regions if detected.
        // In order to find confidences for each detection, group threshold needs to be 0
        cudaDetector->setGroupThreshold(8);
		
	} else {
		// Load the custom SVM detector
		 //svmDetector = HOGDescriptor::getDefaultPeopleDetector();
		 //detector.setSVMDetector(svm->getSupportVectors());
	}
		
	// Vector of Rect containing the positions of detected objects	
	vector<Rect> foundLocations;
	
	
	// CUDA needs to store the input image in GpuMat, not Mat
	cuda::GpuMat cudaImage;
	Mat image;
	Mat tempImage;
	// ATTENTION: The CUDA version only works with grayImage???
	Mat grayImage;

	while(totalRunningTime.size() < 1000) {
		vidCap.read(tempImage);
		
		vector<double> confidences;
		
		// Begin timing
		auto time1 = chrono::system_clock::now();
		
		// resize the image so the winSize 64,128 can be used effectively
		resize(tempImage, image, Size(), 1, 1);
		auto time2 = chrono::system_clock::now();
		
		// Detect the object and store the location of objects into vector<rect>
		if (isUsingCuda) {
			// Convert the image into grayImage for the CUDA detector version
			cvtColor(image, grayImage, COLOR_BGR2GRAY);
			cudaImage.upload(grayImage);
			cudaDetector->detectMultiScale(cudaImage, foundLocations);
			// In order to find the confidences for all detections, set groupThreshold to 0
			// cudaDetector->detectMultiScale(cudaImage, foundLocations, &confidences);				
		} else {
			detector.detectMultiScale(image, foundLocations, 0, Size(8,8), Size(32,32), scaleFactor, 2, false);
		}
		auto time3 = chrono::system_clock::now();
		
		if (!confidences.empty()) { 
			auto max = max_element(begin(confidences), end(confidences));
			cout << "MAX VALUE IS: " << *max << endl;
			drawPeople(image, foundLocations, confidences, *max);
		} else {
			drawPeople(image, foundLocations, confidences, 0.0);
		}
		

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
		
		// Add the runtime to the recording lists in order to calculate average later
		totalRunningTime.push_back(detectingTime.count());
	}
	
	// Calculate average runtime to compare between 2 versions
	double avg = 0;
    list<double>::iterator it;
    for(it = totalRunningTime.begin(); it != totalRunningTime.end(); it++) avg += *it;
    avg /= totalRunningTime.size();
	
	cout << "============" << endl;
	cout << "AVERAGE DETECTING TIME: " << avg << endl;
	
	return 0;
}
