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



int main()
{
	Mat img = imread("blob.jpg", CV_LOAD_IMAGE_ANYCOLOR);
	namedWindow("processed", cv::WINDOW_OPENGL);
	namedWindow("un processed", cv::WINDOW_OPENGL);

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
	waitKey(0);
	return 0;
}

void something()
{
	// doing something here
	cout << "this is something" << endl;
}