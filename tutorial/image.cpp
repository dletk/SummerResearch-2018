#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/xfeatures2d.hpp>
#include<opencv2/cudaimgproc.hpp>

#include<iostream>
#include<stdio.h>
#include<chrono>


using namespace cv;
using namespace std;
using namespace cv::cuda;
using namespace std::chrono;

int main(int argc, char** argv) {
	Mat image = imread(argv[1]);
	//imshow("Image", image);
	//waitKey(0);
	
	
	Mat gray;

	cv::cvtColor(image, gray, COLOR_BGR2GRAY);

	Mat cpuDestination;

	auto startCPU = high_resolution_clock::now();
	Canny(gray, cpuDestination, 3.0, 100);
	auto endCPU = high_resolution_clock::now();
	duration<double> elapsed_CPU_second = endCPU - startCPU;
	//imshow("Canny out", cpuDestination);
	//waitKey(0);
	cout << "Finished detection on CPU within: " << elapsed_CPU_second.count() << "s\n";
	startCPU = high_resolution_clock::now();
	Canny(gray, cpuDestination, 3.0, 100);
	endCPU = high_resolution_clock::now();
	elapsed_CPU_second = endCPU - startCPU;
	cout << "Finished detection on CPU within: " << elapsed_CPU_second.count() << "s\n";


	// Upload the gray image to GPU. Can also called: source.upload(gray)
	GpuMat source(gray);
	GpuMat cannyOutput;
	
	Ptr<CannyEdgeDetector> detector = createCannyEdgeDetector(3.0, 100.0);
	// Only timing the detection
	auto start = high_resolution_clock::now();
	// Using -> to call method because detector is a Pointer
	detector->detect(source, cannyOutput);
	auto end = high_resolution_clock::now();

	duration<double> elapsed_second = end - start;
	cout << "Finished detection on GPU within: " << elapsed_second.count() << "s\n";

	// Only timing the detection
	start = high_resolution_clock::now();
	// Using -> to call method because detector is a Pointer
	detector->detect(source, cannyOutput);
	end = high_resolution_clock::now();

	elapsed_second = end - start;
	cout << "Finished detection on GPU within: " << elapsed_second.count() << "s\n";

	// Download the result to Mat from GPU
	Mat result(cannyOutput);

	//imshow("Canny out", result);
	//waitKey(0);
	return 0;
}
