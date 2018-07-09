#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>

#include<iostream>
#include<stdio.h>
#include<chrono>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	String model = "./model.pb";
	String modeltxt = "./model.pbtxt";
	Mat image = imread(argv[1]);
	
	cvtColor(image, image, COLOR_BGR2GRAY);
	
	// bitwise_not(image, image);
	
	imshow("image", image);
	
	waitKey(0);
	
	//String modelTxt = "./fashionMNISTmodel.pbtxt";
	
	cv::dnn::Net net = cv::dnn::readNetFromTensorflow(model, modeltxt);
	
	
	if (!net.empty()) {
		cout << "Loaded network successfully" << endl;
	}
	
	auto startTime = chrono::system_clock::now();
	
	Mat inputBlob = dnn::blobFromImage(image);   //Convert Mat to dnn::Blob image batch
    net.setInput(inputBlob);        //set the network input
    
	auto loadedInput = chrono::system_clock::now();

	cout << "Loaded input" << endl;
	
    cout << net.forward() << endl;   
    
    auto endTime = chrono::system_clock::now();
    chrono::duration<double> time = endTime - startTime;
    chrono::duration<double> inputTime = loadedInput - startTime;
    
    cout << "Load input time: " << inputTime.count() << endl;
    cout << "Total runtime time: " << time.count() << endl;
	
	return 0;
}
