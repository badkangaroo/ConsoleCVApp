// ConsoleCVApp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\features2d.hpp>
using namespace std;
using namespace cv;

void something();
/// Global variables
Mat src, src_gray;
Mat myHarris_dst; Mat myHarris_copy; Mat Mc;
Mat myShiTomasi_dst; Mat myShiTomasi_copy;

int myShiTomasi_qualityLevel = 50;
int myHarris_qualityLevel = 50;
int max_qualityLevel = 100;

double myHarris_minVal; double myHarris_maxVal;
double myShiTomasi_minVal; double myShiTomasi_maxVal;

RNG rng(12345);

const char* myHarris_window = "My Harris corner detector";
const char* myShiTomasi_window = "My Shi Tomasi corner detector";

/**
* @function myShiTomasi_function
*/
void myShiTomasi_function(int, void*)
{
	myShiTomasi_copy = src.clone();

	if (myShiTomasi_qualityLevel < 1) { myShiTomasi_qualityLevel = 1; }

	for (int j = 0; j < src_gray.rows; j++)
	{
		for (int i = 0; i < src_gray.cols; i++)
		{
			if (myShiTomasi_dst.at<float>(j, i) > myShiTomasi_minVal + (myShiTomasi_maxVal - myShiTomasi_minVal)*myShiTomasi_qualityLevel / max_qualityLevel)
			{
				circle(myShiTomasi_copy, Point(i, j), 4, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
			}
		}
	}
	imshow(myShiTomasi_window, myShiTomasi_copy);
}

/**
* @function myHarris_function
*/
void myHarris_function(int, void*)
{
	myHarris_copy = src.clone();

	if (myHarris_qualityLevel < 1) { myHarris_qualityLevel = 1; }

	for (int j = 0; j < src_gray.rows; j++)
	{
		for (int i = 0; i < src_gray.cols; i++)
		{
			if (Mc.at<float>(j, i) > myHarris_minVal + (myHarris_maxVal - myHarris_minVal)*myHarris_qualityLevel / max_qualityLevel)
			{
				circle(myHarris_copy, Point(i, j), 4, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
			}
		}
	}
	imshow(myHarris_window, myHarris_copy);
}

int main()
{
	// start with color image
	Mat reddotsOrig = imread("reddots.jpg", CV_LOAD_IMAGE_ANYCOLOR);
	
	// make a Hue Saturation Value version
	Mat reddotsHSV;
	cvtColor(reddotsOrig, reddotsHSV, COLOR_BGR2HSV);
	Scalar lower_red = Scalar(0, 150, 50);
	Scalar upper_red = Scalar(255, 255, 255);
	Mat redMask;
	inRange(reddotsHSV, lower_red, upper_red, redMask);
	GaussianBlur(redMask, redMask, Size(7, 7), 4.0, 4.0, 4);
	SimpleBlobDetector::Params params;
	params.minDistBetweenBlobs = 5.0f;
	params.filterByInertia = false;
	params.filterByConvexity = false;
	params.filterByColor = false;
	params.filterByCircularity = false;
	params.filterByArea = true;
	params.minArea = 1.0f;
	params.maxArea = 150.0f;
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	vector<KeyPoint> keyPoints;
	cout << "starting" << endl;
	detector->detect(redMask, keyPoints);
	cout << "drawing" << endl;
	drawKeypoints(reddotsOrig, keyPoints, reddotsOrig, Scalar(255, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	for (int i = 0; i < keyPoints.size(); i++)
	{
		ostringstream temp;
		temp << i;
		putText(reddotsOrig, temp.str(), Point(keyPoints[i].pt), FONT_HERSHEY_TRIPLEX, 0.20, Scalar(0, 0, 255), 1, CV_AA);
		cout << i << endl;
	}
	imshow("counter", reddotsOrig);

	waitKey(0);
	return 0;


	/// Load source image and convert it to gray
	src = imread("building.jpg", CV_LOAD_IMAGE_ANYCOLOR);
	cvtColor(src, src_gray, COLOR_BGR2GRAY);

	/// Set some parameters
	int blockSize = 3; int apertureSize = 3;

	/// My Harris matrix -- Using cornerEigenValsAndVecs
	myHarris_dst = Mat::zeros(src_gray.size(), CV_32FC(6));
	Mc = Mat::zeros(src_gray.size(), CV_32FC1);

	cornerEigenValsAndVecs(src_gray, myHarris_dst, blockSize, apertureSize, BORDER_DEFAULT);

	/* calculate Mc */
	for (int j = 0; j < src_gray.rows; j++)
	{
		for (int i = 0; i < src_gray.cols; i++)
		{
			float lambda_1 = myHarris_dst.at<Vec6f>(j, i)[0];
			float lambda_2 = myHarris_dst.at<Vec6f>(j, i)[1];
			Mc.at<float>(j, i) = lambda_1*lambda_2 - 0.04f*pow((lambda_1 + lambda_2), 2);
		}
	}

	minMaxLoc(Mc, &myHarris_minVal, &myHarris_maxVal, 0, 0, Mat());

	/* Create Window and Trackbar */
	namedWindow(myHarris_window, WINDOW_AUTOSIZE);
	createTrackbar(" Quality Level:", myHarris_window, &myHarris_qualityLevel, max_qualityLevel, myHarris_function);
	myHarris_function(0, 0);

	/// My Shi-Tomasi -- Using cornerMinEigenVal
	myShiTomasi_dst = Mat::zeros(src_gray.size(), CV_32FC1);
	cornerMinEigenVal(src_gray, myShiTomasi_dst, blockSize, apertureSize, BORDER_DEFAULT);

	minMaxLoc(myShiTomasi_dst, &myShiTomasi_minVal, &myShiTomasi_maxVal, 0, 0, Mat());

	/* Create Window and Trackbar */
	namedWindow(myShiTomasi_window, WINDOW_AUTOSIZE);
	createTrackbar(" Quality Level:", myShiTomasi_window, &myShiTomasi_qualityLevel, max_qualityLevel, myShiTomasi_function);
	myShiTomasi_function(0, 0);




	// the Mat data type
	Mat img = imread("blob.jpg", CV_LOAD_IMAGE_ANYCOLOR);
	Mat gimg = img.clone();
	Size s;
	s.height = 7;
	s.width = 7;

	GaussianBlur(img, gimg, s, 4.0, 4.0, 4);
	Mat cHarris;
	cvtColor(img, cHarris, COLOR_BGR2GRAY);
	cornerHarris(cHarris, cHarris, 7, 5, 0.05, BorderTypes::BORDER_DEFAULT);
	imshow("Corners", cHarris);

	namedWindow("processed", cv::WINDOW_OPENGL);
	namedWindow("un processed", cv::WINDOW_OPENGL);

	Mat building = imread("building.jpg", CV_LOAD_IMAGE_ANYCOLOR);
	//copy source image to a single channel grey image
	Mat buildingG;
	cvtColor(building, buildingG, COLOR_BGR2GRAY);
	// get outlines of building in grey
	Canny(buildingG, buildingG, 50, 200, 3);
	// show the buildings lines
	imshow("Building Canny", buildingG);

	vector<Vec2f> lines;
	HoughLines(buildingG, lines, 1, CV_PI / 180, 80);
	cvtColor(buildingG, buildingG, COLOR_GRAY2BGR);
	Mat buildingWithLines = building.clone();
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0];
		float theta = lines[i][1];
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		Point pt1(cvRound(x0 + 100 * (-b)), cvRound(y0 + 100 * (a)));
		Point pt2(cvRound(x0 - 100 * (-b)), cvRound(y0 - 100 * (a)));
		line(buildingWithLines, pt1, pt2, Scalar(0, 0, 255), 1, LineTypes::LINE_AA, 0);
	}
	imshow("HoughLines on building", buildingWithLines);

	// trying harris 2
	Mat buildingH;
	vector<Vec4i> linesB;
	cvtColor(building, buildingH, COLOR_BGR2GRAY);
	Canny(buildingH, buildingH, 50, 200, 3);
	HoughLinesP(buildingH, linesB, 1, CV_PI / 180, 80, 30, 10);
	Mat orig = building.clone();
	for (size_t i = 0; i < linesB.size(); i++)
	{
		line(orig, Point(linesB[i][0], linesB[i][1]),
		Point(linesB[i][2], linesB[i][3]), Scalar(0, 0, 255), 1, 8); 
	}
	imshow("Hough Prob", orig);

	// start with an image with blobs
	Mat gray = imread("blob.jpg", IMREAD_GRAYSCALE);
	cvtColor(img, gray, COLOR_BGR2GRAY);
	medianBlur(gray, gray, 3);
	medianBlur(gray, gray, 3);
	medianBlur(gray, gray, 3);
	medianBlur(gray, gray, 3);
	medianBlur(gray, gray, 3);
	medianBlur(gray, gray, 3);
	medianBlur(gray, gray, 3);



	vector<Vec3f> circles;
	HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
		gray.rows / 32, // change this value to detect circles with different distances to each other
		100, 30, 1, 30 // change the last two parameters
					   // (min_radius & max_radius) to detect larger circles
	);

	for (size_t i = 0; i < circles.size(); i++)
	{
		Vec3i c = circles[i];
		circle(img, Point(c[0], c[1]), c[2], Scalar(0, 0, 255), 3, LINE_AA);
		circle(img, Point(c[0], c[1]), 2, Scalar(0, 255, 0), 3, LINE_AA);
	}

	Mat dst, detected_edges;

	int edgeThresh = 1;
	int lowThreshold;
	int const max_lowThreshold = 100;

	Mat greyEdges;
	cvtColor(img, greyEdges, CV_BGR2GRAY);
	/// Canny detector
	Canny( gray, gray, 1, 300, 3);

	imshow("Edges", gray);

	/// Reduce noise with a kernel 3x3
	blur(greyEdges, detected_edges, Size(3, 3));


	//Mat gimgWithKeyPoints;
	//drawKeypoints( blobImage, blobKeyPoints, gimgWithKeyPoints,  Scalar(0, 0, 255), DrawMatchesFlags::DRAW_OVER_OUTIMG);



	// doing something
	something();
	imshow("processed", gray);
	imshow("un processed", img);

	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	Mat edges;
	namedWindow("video edges", 1);
	for (;;)
	{
		Mat frame;
		cap >> frame; // get a new frame from camera
		Mat colorFrame = frame.clone();
		cvtColor(frame, edges, CV_BGR2GRAY);
		GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
		Canny(edges, edges, 0, 30, 3);
		vector<Vec4i> linesC;
		HoughLinesP(edges, linesC, 1, CV_PI / 180, 80, 30, 10);
		for (size_t i = 0; i < linesC.size(); i++)
		{
			line(colorFrame, Point(linesC[i][0], linesC[i][1]),
				Point(linesC[i][2], linesC[i][3]), Scalar(0, 255, 0), 1, 8);
		}
		imshow("video edges", colorFrame);
		if (waitKey(30) >= 0) break;
	}

	waitKey(0);
	return 0;
}

void something()
{
	// doing something here
	cout << "this is something" << endl;
}