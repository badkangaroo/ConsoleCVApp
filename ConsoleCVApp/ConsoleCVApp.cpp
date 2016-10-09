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

void CannyThreshold(int, void*)
{


}

int main()
{
	Mat img = imread("blob.jpg", CV_LOAD_IMAGE_ANYCOLOR);
	namedWindow("processed", cv::WINDOW_OPENGL);
	namedWindow("un processed", cv::WINDOW_OPENGL);
	cout << img.cols;
	// the Mat data type
	Mat gimg = img.clone();
	Size s;
	s.height = 7;
	s.width = 7;


	GaussianBlur(img, gimg, s, 4.0, 4.0, 4);
	Mat cHarris;
	cvtColor(img, cHarris, COLOR_BGR2GRAY);
	cornerHarris(cHarris, cHarris, 7, 5, 0.05, BorderTypes::BORDER_DEFAULT);
	imshow("Corners", cHarris);

	
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


	CannyThreshold(0, 0);

	//Mat gimgWithKeyPoints;
	//drawKeypoints( blobImage, blobKeyPoints, gimgWithKeyPoints,  Scalar(0, 0, 255), DrawMatchesFlags::DRAW_OVER_OUTIMG);


	// doing something
	something();
	imshow("processed", gray);
	imshow("un processed", img);
	waitKey(0);
	return 0;
}

void something()
{
	// doing something here
	cout << "this is something" << endl;
}