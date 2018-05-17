#include<opencv2/opencv.hpp>
#include<stdio.h>
#include<iostream>
#include<chrono>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat image = imread(argv[1]);
	Mat grayImage;
	cvtColor(image, grayImage, COLOR_BGR2GRAY);
	
	double min, max;
	
	Mat harris, harris_normalized, dst_norm_scaled;
	auto start = chrono::system_clock::now();
	cornerHarris(grayImage, harris, 2, 3, 0.04);
	auto end1 = chrono::system_clock::now();
	
	normalize(harris, harris_normalized, 0, 255, NORM_MINMAX);
	convertScaleAbs( harris_normalized, dst_norm_scaled );
	
	minMaxLoc(harris_normalized, &min, &max);
	auto end2 = chrono::system_clock::now();
	for(int i=0; i < harris_normalized.rows; i++) {
		for(int j=0; j < harris_normalized.rows; j++) {

			if(harris_normalized.at<float>(i,j) > 200) {

				circle(dst_norm_scaled, Point(i,j), 5, Scalar(255,0,0));
			}
		}
	}
	auto end3 = chrono::system_clock::now();
	
	chrono::duration<double> time1 = end1 - start;
	chrono::duration<double> time2 = end2 - end1;
	chrono::duration<double> time3 = end3 - end2;
	chrono::duration<double> total = end3 - start;
	
	cout << "Time to run cornerHarris: " << time1.count() << endl;
	cout << "Time to run normalize and scale: " << time2.count() << endl;
	cout << "Time to run find corner: " << time3.count() << endl;
	cout << "Total runtime: " << total.count() << endl;
	
	imshow("Harris", dst_norm_scaled);
	waitKey(0);
	
	return 0;
}
