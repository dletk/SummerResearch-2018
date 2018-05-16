#include<opencv2/core.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>
#include<stdio.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat frame;
	VideoCapture vidCap(0);
	
	while(true) {
		vidCap.read(frame);
		imshow("Live", frame);
		waitKey(5);
	}
}
