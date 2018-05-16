#include<opencv2/opencv.hpp>
#include<opencv2/features2d.hpp>
#include<iostream>
#include<stdio.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	//Preparing the detector
	Ptr<ORB> orbDetector = ORB::create(5000);
	
	Mat image1 = imread(argv[1]);
	Mat image2 = imread(argv[2]);
	
	Mat grayImage1, grayImage2;
	// ORB only works with gray image
	cv::cvtColor(image1, grayImage1, COLOR_BGR2GRAY);
	cv::cvtColor(image2, grayImage2, COLOR_BGR2GRAY);
	
	// Stores keypoints and descriptors
	vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	
	// Using an empty Mat() for mask
	orbDetector->detectAndCompute(grayImage1, Mat(), keypoints1, descriptors1);
	orbDetector->detectAndCompute(grayImage2, Mat(), keypoints2, descriptors2);
	
	// Matches features together
	vector<DMatch> matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptors1, descriptors2, matches, Mat());
	
	// Draw out matches
	Mat imageMatches;
	drawMatches(grayImage1, keypoints1, grayImage2, keypoints2, matches, imageMatches);
	imshow("MATCH", imageMatches);
	
	waitKey(0);
	return 0;
}
