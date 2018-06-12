#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>

#include<iostream>
#include<stdio.h>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	String model = "../../model.pb";
	Mat image = imread("./example.jpg");
	
	cvtColor(image, image, COLOR_BGR2GRAY);
	
	bitwise_not(image, image);
	
	imshow("image", image);
	
	waitKey(0);
	
	//String modelTxt = "./fashionMNISTmodel.pbtxt";
	
	cv::dnn::Net net = cv::dnn::readNetFromTensorflow(model);
	
	if (!net.empty()) {
		cout << "Loaded network successfully" << endl;
	}
	
	Mat inputBlob = dnn::blobFromImage(image);   //Convert Mat to dnn::Blob image batch
    net.setInput(inputBlob);        //set the network input

	cout << "Loaded input" << endl;
	
    cout << net.forward() << endl;   
	
	return 0;
}
