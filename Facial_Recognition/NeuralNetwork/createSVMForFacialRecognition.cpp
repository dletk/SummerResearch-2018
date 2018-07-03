#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>

#include<iostream>
#include<stdio.h>

using namespace cv;
using namespace std;

void loadImages(vector<Mat> &imageLists, vector<String> filePaths);
void computeFacialMeasurment(vector<Mat> &imageLists, vector<Mat> &measurement_list);


// Method to create a list of images from the input file paths
void loadImages(vector<Mat> &imageLists, vector<String> filePaths) {
	// Sort the filePath
	sort(filePaths.begin(), filePaths.end());
	
	for(int i = 0; i < filePaths.size(); i++) {
		String filePath = filePaths[i];
		Mat currentImage = imread(filePath, IMREAD_GRAYSCALE);
		
		// Notify failure
		if (currentImage.empty()) {
			cout << "File " << filePath << " is not a valid image." << endl; 
		} else {
			imageLists.push_back(currentImage);
		}
	}
	cout << "Finished loading images" << endl;
}

int main(int argc, char** argv) {
	// Prepare the commandline parser keys
	const String keys = 
	{
		"{help h          |     | show help message}"
		"{imgPth imagePath|     | path of the directory containing static images}"
	};
	
	CommandLineParser parser(argc, argv, keys);
	// Display help message if needed
	if (parser.has("help")) {
		parser.printMessage();
		exit(0);
	}
	
	String directoryPath = parser.get<String>("imagePath");
	
	vector<Mat> imageLists;
	vector<String> filePaths;
	
	glob(directoryPath, filePaths);
	
	loadImages(imageLists, filePaths);
}
