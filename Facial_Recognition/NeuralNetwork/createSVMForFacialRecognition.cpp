#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include<opencv2/ml.hpp>

#include<iostream>
#include<stdio.h>

using namespace cv;
using namespace std;
using namespace cv::ml;

void loadImages(vector<Mat> &imagesList, vector<String> filePaths, int imageWidth, int imageHeight, vector<int> &labels);
void computeFacialMeasurement(String model, String modeltxt, vector<Mat> &imagesList, vector<Mat> &measurementsList, bool showProgress);
void convert_to_ml( const vector<Mat> &measurementsList, Mat &trainData );

// Little helping function to print out the progress, acquired from https://stackoverflow.com/questions/14539867/how-to-display-a-progress-indicator-in-pure-c-c-cout-printf
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

void printProgress (double percentage) {
    double val = (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r\t[%.2f%%] [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush(stdout);
}

// Method to create a list of images from the input file paths
void loadImages(vector<Mat> &imagesList, vector<String> filePaths, int imageWidth, int imageHeight, vector<int> &labels) {
	// Sort the filePath
	sort(filePaths.begin(), filePaths.end());
	
	// Variable to keep track current label
	int currentLabel = 0;
	String lastLabel = String();
	
	for(int i = 0; i < filePaths.size(); i++) {
		String filePath = filePaths[i];
		Mat currentImage = imread(filePath, IMREAD_GRAYSCALE);
		
		// Notify failure
		if (currentImage.empty()) {
			cout << "File " << filePath << " is not a valid image." << endl; 
		} else if (imageWidth != currentImage.cols || imageHeight != currentImage.rows) {
			cout << "File " << filePath << " does not have the right size." << endl;
		} else {
			imagesList.push_back(currentImage);
			// Find the label of the file
			int begin = filePath.find_last_of("/")+1;
			int end = filePath.find_last_of(" ", filePath.length()-1);
			String labelImage = filePath.substr(begin, end-begin);
			if (labelImage == lastLabel) {
				labels.push_back(currentLabel);
			} else {
				currentLabel += 1;
				labels.push_back(currentLabel);
				lastLabel = labelImage;
			}
		}
	}
	cout << "Finished loading images" << endl;
}

// Method to load and computre the facial measurement of all input images
void computeFacialMeasurement(String model, String modeltxt, vector<Mat> &imagesList, vector<Mat> &measurementsList, bool showProgress) {
	// Load the model weights and structure into openCV
	cv::dnn::Net net = cv::dnn::readNetFromTensorflow(model, modeltxt);
	cout << "Begin to make measurement of all input images..." << endl;
	// Loop over the image lists and compute the measurements
	for (int i=0; i<imagesList.size(); i++) {
		Mat image = imagesList[i];
	
		Mat inputBlob = dnn::blobFromImage(image);   //Convert Mat to dnn::Blob image batch
    	net.setInput(inputBlob);
    	
    	// Produce measurement
    	Mat output = net.forward();
    	//cout << output << endl;
    	// Normalize the measurement
    	//cuda::normalize(output, output, 0, 1, NORM_MINMAX,-1);
    	
    	// Save the measurement, NOTICE: THE OUTPUT MAT WILL BE REUSED EVEN IT IS CREATED EVERYTIME, SO DO A CLONE TO COPY THE DATA
    	measurementsList.push_back(output.clone());
    	//cout << "ele" << endl;
		//for (Mat ele: measurementsList) {
		//	cout << ele << endl;
		//}
    	if (showProgress) {
    		printProgress(double (i) / double (imagesList.size()));
    	}
	}
	cout << endl;
	cout << "Finish all measurements." << endl;
}

/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (numSamples x max(numCols, numRows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/
void convert_to_ml(const vector< Mat > &measurementsList, Mat& trainData ) {
    //--Convert data
    const int rows = (int)measurementsList.size();
    const int cols = (int)std::max( measurementsList[0].cols, measurementsList[0].rows );
    
    
    Mat tmp( 1, cols, CV_32FC1 ); //< used for transposition if needed
    // Prepare the Mat containing the train data
    trainData = Mat( rows, cols, CV_32FC1 );
    for( size_t i = 0 ; i < measurementsList.size(); ++i ) {
    	
    	// Check if the size of the current sample is appropriate to process
    	// A sample should have 1 row OR 1 col. If the sample has 1 row, that is a normal case, 
    	// If a sample has 1 cols, it needs to be transpose before being added to train data.
        CV_Assert( measurementsList[i].cols == 1 || measurementsList[i].rows == 1 );
        
        // Transpose the inapproriate sample
        if( measurementsList[i].cols == 1 ) {
            transpose( measurementsList[i], tmp );
            cout << "transposed" << endl;
            tmp.copyTo( trainData.row( (int)i ) );
        } else if( measurementsList[i].rows == 1 ) {
            measurementsList[i].copyTo( trainData.row( (int)i ) );
        }
    }
}

int main(int argc, char** argv) {
	// Prepare the commandline parser keys
	const String keys = 
	{
		"{help h          |           | show help message}"
		"{imgPth imagePath|           | path of the directory containing static images}"
		"{model           |model.pb   | the .pb model file to load the network weight}"
		"{modeltxt        |model.pbtxt| the .pbtxt file to load model structure}"
		"{showProgress    |false      | show the progress of creating the SVM}"
		"{width           |176        | image width}"
		"{height          |192        | image height}"
		"{svmFileName     |faceSVM.yml| name of the file to save the trained SVM}"
	};
	
	CommandLineParser parser(argc, argv, keys);
	// Display help message if needed
	if (parser.has("help")) {
		parser.printMessage();
		exit(0);
	}
	
	String directoryPath = parser.get<String>("imagePath");
	String model = parser.get<String>("model");
	String modeltxt = parser.get<String>("modeltxt");
	bool showProgress = parser.get<bool>("showProgress");
	int imageWidth = parser.get<int>("width");
	int imageHeight = parser.get<int>("height");
	String svmFileName = parser.get<String>("svmFileName");
	
	vector<Mat> imagesList;
	vector<String> filePaths;
	vector<Mat> measurementsList;
	vector<int> labels;
	Mat trainData;
	
	glob(directoryPath, filePaths);
	
	loadImages(imagesList, filePaths, imageWidth, imageHeight, labels);
	computeFacialMeasurement(model, modeltxt, imagesList, measurementsList, showProgress);
	//for (int i=0; i < measurementsList.size(); i++) {
	//	cout << measurementsList[i] << endl;
	//}
	
	convert_to_ml(measurementsList, trainData );
	
    clog << "Training SVM...";
    Ptr< SVM > svm = SVM::create();
    /* Default values to train SVM */
    
   	svm->setCoef0( 0.01 );
    svm->setDegree( 5 );
    //svm->setTermCriteria( TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100000, 1e-6 ) );
    svm->setGamma(1500);
    svm->setKernel( SVM::RBF );
    svm->setNu( 0.5 );
    svm->setP( 0.1 ); // for EPSILON_SVR, epsilon in loss function?
    svm->setC( 1000000 ); // From paper, soft classifier
    svm->setType( SVM::C_SVC ); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task

    svm->train( trainData, ROW_SAMPLE, labels );
    clog << "...[done]" << endl;
    
    cout << "Test the trained SVM" << endl;
    Mat results;
    cout << svm->predict(trainData.row(5), results) << endl;
    cout << results << endl;
    cout << svm->predict(trainData.row(248), results) << endl;
    cout << results << endl;
    cout << svm->predict(trainData.row(857), results) << endl;
    cout << results << endl;
    
    
    
    // Save the model after training
    svm->save(svmFileName);
}
