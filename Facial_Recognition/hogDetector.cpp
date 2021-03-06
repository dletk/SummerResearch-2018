#include<opencv2/opencv.hpp>
#include<opencv2/ml.hpp>
#include<opencv2/cudaobjdetect.hpp>
#include<opencv2/dnn.hpp>

#include<algorithm>
#include<iostream>
#include<string>
#include<stdio.h>
#include<chrono>
#include<fstream>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include<dlib/opencv.h>


using namespace cv;
using namespace std;

String exportedDir;

dlib::image_window win, win_Reference;
dlib::shape_predictor pose_model;
vector<Point2f> landmarksPositions;

bool isUsingCuda, isUsingImage, isCreatingData, displayOutput;
bool isDetecting;

int currentFrame;
Mat originalImageGray;
// Define the size of a face
dlib::rectangle faceSize(0, 0, 176, 192);

// Variable to make prediction for face recognition
dnn::Net neuralNet;
Ptr< ml::SVM > svmFaceRecognition;
vector<String> labels;
vector<int> detectedLabels;


void prepareFacialRecognition(String model, String modeltxt, String svmFaceRegPath, String labelsFile) {
	cout << "Begin to load models and neural network for facial recognition" << endl;
	// Load the neural network
	neuralNet = dnn::readNetFromTensorflow(model, modeltxt);
	// Load the SVM for facial recognition
	svmFaceRecognition = ml::SVM::load(svmFaceRegPath);
	cout << "Loaded neural network and facial recognition SVM" << endl;
	
	// Load the labels for the SVM
	// The labels from our SVM is 1-216, so make the 0 element be an empty string.
	labels.push_back("");
	// Open the file
	ifstream file(labelsFile);
	string aLabel;
	if (file.is_open()) {
		while (getline(file, aLabel)) {
			cout << aLabel << endl;
			labels.push_back(String(aLabel));
		}
		file.close();
	}
	
	for(int i=0;i<labels.size();i++) {
		cout << labels[i] << endl;
	}
}

// Method to predict the individual
int predictIndividual(Mat image) {
	Mat inputBlob = dnn::blobFromImage(image);   //Convert Mat to dnn::Blob image batch
    neuralNet.setInput(inputBlob);        //set the network input
    
    auto timeBegin = chrono::system_clock::now();
	Mat outMeasurement = neuralNet.forward();
	auto timeEnd = chrono::system_clock::now();
	chrono::duration<double> timeNetwork = timeEnd - timeBegin;
	cout << "--- Network time: " << timeNetwork.count() << endl;
	
	Mat results;
	svmFaceRecognition->predict(outMeasurement, results);
	return (int) results.at<float>(0,0);
}


// NOTE: ORDER OF FACIAL LANDMARKS IN DLIB
//Points 0 to 16 is the Jawline
//Points 17 to 21 is the Right Eyebrow
//Points 22 to 26 is the Left Eyebrow
//Points 27 to 35 is the Nose
//Points 36 to 41 is the Right Eye
//Points 42 to 47 is the Left Eye
//Points 48 to 60 is Outline of the Mouth
//Points 61 to 67 is the Inner line of the Mouth

// Method to create the reference face aligment from the reference image
void createReferenceFace(Mat referenceImage) {
	// Create a dlib image from the openCV reference image
	dlib::cv_image<unsigned char> image(referenceImage);
	
	// Detect the facial landmarks in the current bounding box
	dlib::full_object_detection shape = pose_model(image, faceSize);
	
	Point2f point1((float) shape.part(36).x(), (float) shape.part(36).y());
	Point2f point2((float) shape.part(47).x(), (float) shape.part(47).y());
	Point2f point3((float) shape.part(34).x(), (float) shape.part(34).y());
	
	cout << point1 << endl;
	cout << point2 << endl;
	cout << point3 << endl;
	
	landmarksPositions.push_back(point1);
	landmarksPositions.push_back(point2);
	landmarksPositions.push_back(point3);
	
  	win_Reference.clear_overlay();	
    win_Reference.set_image(image);
    win_Reference.add_overlay(render_face_detections(shape));
    
    win_Reference.wait_until_closed();
}

// Method to align facial landmarks to match the positions in reference image using affine transformation 
Mat alignImageAndPredict(Mat image, dlib::full_object_detection detectedMarks, String file) {
	vector<Point2f> fromPoints;
	
	Point2f point1((float) detectedMarks.part(36).x(), (float) detectedMarks.part(36).y());
	Point2f point2((float) detectedMarks.part(47).x(), (float) detectedMarks.part(47).y());
	Point2f point3((float) detectedMarks.part(34).x(), (float) detectedMarks.part(34).y());
	
	fromPoints.push_back(point1);
	fromPoints.push_back(point2);
	fromPoints.push_back(point3);
	
	warpAffine(image, image, getAffineTransform(fromPoints, landmarksPositions), Size(176,192), INTER_CUBIC);

	// Only export the aligned image to file if creating data
	if (isCreatingData) {
		imwrite(exportedDir+file, image);
	}
	
	return image;
}

// The method to find 68 facial landmarks from a list of detected faces
void findLandmarks(Mat image, vector<Rect> faces, String file) {
	if (isDetecting) {
		detectedLabels.clear();
		for (Rect rect: faces) {
			// Crop the face out of the original image and resize it to the desired size
			Mat faceRegion = originalImageGray(rect);
			resize(faceRegion, faceRegion, Size(176,192));
		
			// Convert the openCV image to dlib image
			dlib::cv_image<unsigned char> regionImage(faceRegion);
		
			// Detect the facial landmarks in the current bounding box
			dlib::full_object_detection shape = pose_model(regionImage, faceSize);
			auto timeBegin = chrono::system_clock::now();
			int index = predictIndividual(alignImageAndPredict(faceRegion, shape, file));
			auto timeEnd = chrono::system_clock::now();
			chrono::duration<double> timePredict = timeEnd - timeBegin;
			
			cout << "Time of prediction: " << timePredict.count() << endl;
			cout << "NUMBER OF DETECTED LANDMARKS: " << shape.num_parts() << endl;
			cout << "Size of face: " << rect.size() << endl;
			detectedLabels.push_back(index);
		}
	}
	
	// Convert the openCV image to dlib image
	dlib::cv_image<dlib::bgr_pixel> cvImage(image);
	
	// Display it all on the screen
	if (!isCreatingData) {
		win.clear_overlay();
		win.set_image(cvImage);
	}
    
    if (isUsingImage) {
    	unsigned long temp1;
    	bool temp2;
    	if (displayOutput) {
			win.get_next_keypress(temp1, temp2);
    	}
    }
}

// Method to draw a rectangle box around the detected object on the image
void drawPeople(Mat image, vector<Rect> foundLocations, vector<double> confidences, double max, String file=String()) {
	int i = 0;
	
	// Loop through all detected faces in the image
	for(Rect rect: foundLocations) {
		// If confidences is available, display detected regions based on their probability
		if (!confidences.empty()) {
			cout << confidences.at(i) << endl;
			// Using 0.05 to have some tolerance
			if (confidences.at(i) == max) {
				rectangle(image, Point(rect.x, rect.y), Point(rect.x+rect.width, rect.y+rect.height), Scalar(255,0,0));
			}
		} else {
			rectangle(image, Point(rect.x, rect.y), Point(rect.x+rect.width, rect.y+rect.height), Scalar(255,0,0));
			if (i<detectedLabels.size()) {
				putText(image, labels[detectedLabels[i]], Point(rect.x, rect.y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
				cout << labels[detectedLabels[i]] << endl;
			}
		}
		i++;
	}
	
	findLandmarks(image, foundLocations, file);
	
	// imshow("Image", image);
	// We are using camera stream, so the waitTime is set to 1. Change it to 0 for a static image.
	// waitKey(1);
}

int main(int argc, char** argv) {
	// Prepare the commandline parser keys
	const String keys = 
	{
		"{help h          |                         | show help message}"
		"{image           |false                    | run the program with static image instead of video stream}"
		"{imgPth imagePath|                         | path of the directory containing static images}"
		"{winW winWidth	  |176                      | winSize width}"
		"{winH winHeight  |192                      | winSize height}"
		"{resizeScale     |1.5                      | Resize scale to find smaller faces}"
		"{cuda            |false                    | using CUDA}"
		"{detector        |                         | filename of the detector for face detection}"
		"{createData      |false                    | create data from the detection process}"
		"{referenceImage  |./referenceImage.jpg     | reference image to create data}"
		"{displayOutput   |false                    | display the output}"
		"{neuralNetModel  |model.pb                 | the .pb model file to load the network weight}"
		"{neuralNetText   |model.pbtxt              | the .pbtxt file to load model structure}"
		"{faceSVM         |faceSVM.yml              | name of the file to save the trained SVM}"
		"{labelsFile      |labels.txt               | file path to the labels file}"
	};
	
	CommandLineParser parser(argc, argv, keys);
	// Display help message if needed
	if (parser.has("help")) {
		parser.printMessage();
		exit(0);
	}
	
	// The flag to trigger CUDA if needed
	isUsingCuda = parser.get<bool>("cuda");
	// The flag to trigger using static image instead video stream
	isUsingImage = parser.get<bool>("image");
	// The flag to trigger creating data
	isCreatingData = parser.get<bool>("createData");
	// The flag to trigger displaying output
	displayOutput = parser.get<bool>("displayOutput");


	
	String detectorPath = parser.get<String>("detector");
	String imagePath = parser.get<String>("imgPth");
	String referenceImagePath = parser.get<String>("referenceImage");
	int winWidth = parser.get<int>("winW");
	int winHeight = parser.get<int>("winH");
	double resizeScale = parser.get<double>("resizeScale");
	
	String neuralNetModel = parser.get<String>("neuralNetModel");
	String neuralNetText = parser.get<String>("neuralNetText");
	String faceSVM = parser.get<String>("faceSVM");
	String labelsFile = parser.get<String>("labelsFile");
	
	
	// Getting all the images in the directory of imagePath if using static images
	// List of files name
    vector< String > files;
    // the openCV helper function to get all the file names in a directory and put into files
    glob( imagePath, files );
	
	// Variable for creating the HOG detector and descriptors.
	double scaleFactor = 1.2;
	Size winSize(winWidth, winHeight);
	
	// Dlib facial landmark model
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

	
	// The list to keep track of all the recorded runtime to calculate average runtime if needed
	list<double> totalRunningTime;
	
	// Setting up the onboard webcam of Jetson TX2
	const char* gst =  "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,\
			format=(string)I420, framerate=(fraction)30/1 ! \
			nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! \
			videoconvert ! video/x-raw, format=(string)BGR ! \
			appsink";
	VideoCapture vidCap(1);
	
	// Set the count for current frame to be 0
	currentFrame = 0;
	
	// Prepare different versions of the HOG detector because CUDA and sequential code have different types
	Ptr<cuda::HOG> cudaDetector;
	HOGDescriptor detector;
	
	// Create the dectector and result vector
	if (!isUsingCuda) {
		detector = HOGDescriptor(winSize, // winSize
		 Size(16,16), // blockSize
		 Size(8,8), // blockStride
		 Size(8,8), // cellSize
		 9, // nbins
		 1 ); //deriveAperture
	} else {	
		// Initialize cuda
		// ATTENTION: This step is essential for CUDA to reach its maximum processing efficiency
		cuda::GpuMat test;
		test.create(1, 1, CV_8U);
		test.release();
		cout << "Using CUDA" << endl;
		cudaDetector = cuda::HOG::create(winSize);		
	}
	
	
	// Prepare the detector for the HOG feature descriptor
	// CUDA version of HOG descriptor does not have the loading function from yml file.
	// This work around includes loading the SVM yml file to the sequential detector, and then export it to the CUDA version using getter method.
    detector.load(detectorPath);
    
    // Get the svmDetector loaded from the yml file.
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
		
	}
		
	// Vector of Rect containing the positions of detected objects	
	vector<Rect> foundLocations;
	
	
	// CUDA needs to store the input image in GpuMat, not Mat
	cuda::GpuMat cudaImage;
	Mat image;
	Mat tempImage;
	// ATTENTION: The CUDA version only works with grayImage???
	Mat grayImage;
	
	Mat imageReference = imread(referenceImagePath);
	cvtColor(imageReference, grayImage, COLOR_BGR2GRAY);
	createReferenceFace(grayImage);
	
	if (isCreatingData) {
		exportedDir = "./alignedImages/";
	}
	
	cout << "CREATED REFERENCE IMAGE" << endl;
	
	
	// Prepare models and neural network for facial recognition step
	prepareFacialRecognition(neuralNetModel, neuralNetText, faceSVM, labelsFile);
	
	// Using a static image as input instead of a video stream
	if (isUsingImage) {
		try {
			for (int i=0; i<files.size(); i++) {
				cout << "Working on " << files[i] << endl;
				tempImage = imread(files[i]);
			
				if (tempImage.empty()) {
					cout << files[i] << " is not a valid image." << endl;
					continue;
				} else {	
					// Resize the image to larger scale will help us to find smaller face, but it will requires more time to run and reduce the fps
					resize(tempImage, image, Size(), resizeScale, resizeScale);
		
					cvtColor(image, grayImage, COLOR_BGR2GRAY);
					originalImageGray = grayImage.clone();
		
		
					if (isUsingCuda) {
						// Convert the image to gray scale
						cvtColor(image, grayImage, COLOR_BGR2GRAY);
						originalImageGray = grayImage.clone();
						cudaImage.upload(grayImage);
						cudaDetector->detectMultiScale(cudaImage, foundLocations);
					} else {
						detector.detectMultiScale(image, foundLocations, 0, Size(8,8), Size(32,32), scaleFactor, 2, false);
			
					}
					drawPeople(image, foundLocations, vector<double>(), 0.0, files[i]);
					cout << "Done with " << files[i] << endl;
				}
			}
		} catch (cv::Exception & e) {
			cerr << e.msg << endl;
		}
		exit(0);
	}
	
	// RUNNING the video stream version
	while(totalRunningTime.size() < 1000) {
		vidCap.read(tempImage);
		
		vector<double> confidences;
		
		// Begin timing
		auto time1 = chrono::system_clock::now();
		
		// Resize the image to larger scale will help us to find smaller face, but it will requires more time to run and reduce the fps
		resize(tempImage, image, Size(), resizeScale, resizeScale);
		auto time2 = chrono::system_clock::now();
		
		// ONly do the detection every 2 frames
		if (currentFrame % 3 == 0) {
			// Using isDetecting to make sure the detection only happen once every 5 frames
			isDetecting = true;
			
			// Detect the object and store the location of objects into vector<rect>
			if (isUsingCuda) {
				// Convert the image into grayImage for the CUDA detector version
				cvtColor(image, grayImage, COLOR_BGR2GRAY);
				originalImageGray = grayImage.clone();
				cudaImage.upload(grayImage);
				cudaDetector->detectMultiScale(cudaImage, foundLocations);
				// In order to find the confidences for all detections, set groupThreshold to 0
				// cudaDetector->detectMultiScale(cudaImage, foundLocations, &confidences);				
			} else {
				detector.detectMultiScale(image, foundLocations, 0, Size(8,8), Size(32,32), scaleFactor, 2, false);
				cvtColor(image, grayImage, COLOR_BGR2GRAY);
				originalImageGray = grayImage.clone();
			}
		
		}
		auto time3 = chrono::system_clock::now();
		
		if (!confidences.empty()) { 
			auto max = max_element(begin(confidences), end(confidences));
			cout << "MAX VALUE IS: " << *max << endl;
			drawPeople(image, foundLocations, confidences, *max);
		} else {
			drawPeople(image, foundLocations, confidences, 0.0);
		}
		
		// Turn off isDetecting after each frame
		isDetecting = false;
		currentFrame += 1;

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
