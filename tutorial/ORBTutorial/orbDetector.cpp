#include<opencv2/opencv.hpp>
#include<opencv2/features2d.hpp>
#include<opencv2/cudafeatures2d.hpp>
#include<iostream>
#include<stdio.h>
#include<chrono>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	//Preparing the CPU detector and GPU detector
	Ptr<ORB> orbDetector = ORB::create(5000);
	// ALWAYS REMEMEBER to use cuda version instead of normal orb version
	Ptr<cuda::ORB> cudaOrbDetector = cuda::ORB::create(5000);
	
	// Initialize cuda
	cuda::GpuMat test;
	test.create(1, 1, CV_8U);
	test.release();
	
	cout << "DONE INITIALIZE" << endl;
	
	Mat image1 = imread(argv[1]);
	Mat image2 = imread(argv[2]);
	//image2 = image2(Rect(40,40, 200,200));

	Mat grayImage1, grayImage2;
	// ORB only works with gray image
	cv::cvtColor(image1, grayImage1, COLOR_BGR2GRAY);
	cv::cvtColor(image2, grayImage2, COLOR_BGR2GRAY);
	
	// Upload the images to cuda
	cuda::GpuMat cudaImage1(grayImage1);
	cuda::GpuMat cudaImage2(grayImage2);
	
	// Stores keypoints and descriptors
	vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	
	// REMEMBER: Cuda can only works with GpuMat, so keypoints should be GpuMat as well instead of vector
	cuda::GpuMat cudaDescriptors1, cudaDescriptors2, cudaKeypoints1, cudaKeypoints2;
	
	// CPU version implementation
	auto start = chrono::system_clock::now();
	// Using an empty Mat() for mask
	orbDetector->detectAndCompute(grayImage1, Mat(), keypoints1, descriptors1);
	orbDetector->detectAndCompute(grayImage2, Mat(), keypoints2, descriptors2);
	
	// Matches features together
	vector<DMatch> matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptors1, descriptors2, matches, Mat());
	auto end = chrono::system_clock::now();
	
	cout << "DONE CPU VERSION" << endl;
	
	//GPU version implementation
	auto gpuStart = chrono::system_clock::now();
	// Detect the keypoints
	cudaOrbDetector->detectAndComputeAsync(cudaImage1, cuda::GpuMat(), cudaKeypoints1, cudaDescriptors1);
	cudaOrbDetector->detectAndComputeAsync(cudaImage2, cuda::GpuMat(), cudaKeypoints2, cudaDescriptors2);
	
	cout << "DONE DETECT AND COMPUTE" << endl;
	
	// Matches features togeter
	vector<DMatch> matchesCuda;
	Ptr<cuda::DescriptorMatcher> matcherCuda = cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	
	cout << "DONE CREATE MATCHER" << endl;
	
	matcherCuda->match(cudaDescriptors1, cudaDescriptors2, matchesCuda, cuda::GpuMat());
	auto gpuEnd = chrono::system_clock::now();
	
	chrono::duration<double> elapsed_time = end - start;
	chrono::duration<double> elapsed_time_gpu = gpuEnd - gpuStart;
	cout << "CPU Time run: " << elapsed_time.count() << endl;
	cout << "GPU Time run: " << elapsed_time_gpu.count() << endl;

	// Draw out matches
	//Mat imageMatches;
	//drawMatches(grayImage1, cudaKeypoints1, grayImage2, cudaKeypoints2, matchesCuda, imageMatches);
	//imshow("MATCH", imageMatches);
	//waitKey(0);

	return 0;
}
