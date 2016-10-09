// ConsoleCVApp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

using namespace std;
using namespace cv;

void something();

int main()
{
	Mat img = imread("image.jpg", CV_LOAD_IMAGE_ANYCOLOR);
	namedWindow("processed", cv::WINDOW_OPENGL);
	namedWindow("un processed", cv::WINDOW_OPENGL);
	cout << img.cols;
	// the Mat data type
	Mat gimg = img.clone();
	Size s;
	s.height = 7;
	s.width = 7;
	GaussianBlur(img, gimg, s, 4.0, 4.0, 4);

	// doing something
	something();
	imshow("processed", gimg);
	imshow("un processed", img);
	waitKey(0);
	return 0;
}

void something()
{
	// doing something here
	cout << "this is something" << endl;
}