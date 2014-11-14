//Background Substraction Using Codebook
//cv_yuv_codebook borrowed from code of "Learning OpenCV"
//Yidong Ma 12th, Nov 2014

#include "cv.h"
#include "highgui/highgui.hpp"
#include "objdetect/objdetect.hpp"
#include "imgproc/imgproc.hpp"
#include "cv_yuv_codebook.h"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


#define FRAMES_TO_LEARN 300//Amount of frames for learning. Learning Speed is approximatly 10frame/second.  
#define CHANNELS 3//CHANNELS of captured video
codeBook *cB;  //This will be our linear model of the image, a vector of lengh = height*width
int maxMod[CHANNELS]; //Add these (possibly negative) number onto max level when code_element determining if new pixel is foreground
int minMod[CHANNELS]; //Subract these (possible negative) number from min level code_element when determining if pixel is foreground
unsigned cbBounds[CHANNELS]; //Code Book bounds for learning
int nChannels = CHANNELS;
int imageLen;
bool ch[CHANNELS];
int learnCnt = 0;


void learnBackground(IplImage* fra);//learn background
void diffForeground(IplImage* fra);//substract foreground
int initCodeBook(CvCapture* cap);//initialize image model

int main(int argc, const char** argv)
{
	IplImage* frame;
	cvNamedWindow("backgroundDiff");
	CvCapture* capture;

	capture = cvCreateCameraCapture(0);
	assert (capture != NULL );
	
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_PLAIN|CV_FONT_BLACK, 1, 1, 0, 2, 8);

	//initiate codebook
	if (initCodeBook(capture) == -1)
		return -1;
	
	//learn
	while(learnCnt < FRAMES_TO_LEARN)
	{
		frame = cvQueryFrame(capture);
		if (frame == NULL) break;
		learnBackground(frame);
		if (learnCnt % 100 == 0){//clear stale data after every 100frames
			int cleanedCnt; //will hold number of codebook entries eliminated
			cleanedCnt = 0;
			for(int c=0; c<imageLen; c++)
			{
				cleanedCnt += cvclearStaleEntries(cB[c]);
			}
		}
		CvPoint ori = cvPoint(100, 100);
		CvScalar textColor = cvScalar(0, 0, 255);
		cvPutText(frame, "learning...", ori, &font, textColor);
		cvShowImage("backgroundDiff", frame);
		char c = cvWaitKey(30);
	}
	
	//substract background
	while(1){
		frame = cvQueryFrame(capture);
		if (frame == NULL) break;
		diffForeground(frame);
	
	}
	cvReleaseImage(&frame);
	cvReleaseCapture(&capture);

}

int initCodeBook(CvCapture* cap)
{
	IplImage* frame;
	for (int i = 0; i< 1000; i++){
		frame = cvQueryFrame(cap);
		if ( frame != NULL )
		{
			imageLen = frame->width*frame->height;
			cB = new codeBook [imageLen];
			for(int f = 0; f<imageLen; f++)
			{
				 cB[f].numEntries = 0;
			}
			for(int n=0; n<nChannels;n++)
			{
				cbBounds[n] = 10; //Learning bounds factor
			}
			maxMod[0] = 3;  //Set color thresholds to more likely values
			minMod[0] = 10;
			maxMod[1] = 1;
			minMod[1] = 1;
			maxMod[2] = 1;
			minMod[2] = 1;
			return 0;
		}
	}
	return -1;
}

void learnBackground(IplImage* fra){
	IplImage* yuvImage = cvCreateImage(cvGetSize(fra),8 , CHANNELS);
	cvCvtColor( fra, yuvImage, CV_BGR2YCrCb );//convert frame in RGB model to YUV model
	uchar *pColor; //YUV pointer
	pColor = (uchar *)((yuvImage)->imageData);//colors of frame in yuv model
	for(int c=0; c<imageLen; c++)
	{
		cvupdateCodeBook(pColor, cB[c], cbBounds, nChannels);//for each pixel, update codebook for the pixel.
		pColor += 3;
	}
	learnCnt += 1;//frames learned add one.
	cvReleaseImage(&yuvImage);
}

void diffForeground(IplImage* fra){
	IplImage* yuvImage = cvCreateImage(cvGetSize(fra),8 , CHANNELS);//convert frame to YUV model
	IplImage* maskImage = cvCreateImage(cvGetSize(fra),8 , 1);//convert frame to foreground mask.
	IplImage* maskCC = cvCreateImage(cvGetSize(fra),8 , 1);//add rectangle to mask
	cvCvtColor( fra, yuvImage, CV_BGR2YCrCb );
	uchar *pMask,*pColor;
//For connected components bounding box and center of mass if wanted, else can leave out by default
	int num = 7; //Just chose 7 arbitrarily, could be 1, 20, anything
	CvRect bbs[7];
	CvPoint centers[7];

	pColor = (uchar *)((yuvImage)->imageData); //3 channel yuv image
	pMask = (uchar *)((maskImage)->imageData); //1 channel image
	for(int c=0; c<imageLen; c++)
	{
		uchar maskQ = cvbackgroundDiff(pColor, cB[c], nChannels, minMod, maxMod);
		*pMask++ = maskQ;
		pColor += 3;
	}
	//Ths part just to visualize bounding boxes and centers if desired
	cvCopy(maskImage,maskCC);
	cvconnectedComponents(maskCC,1,4.0, &num, bbs, centers);
	for(int f=0; f<num; f++)
	{
		CvPoint pt1, pt2; //Draw the bounding box in white
		pt1.x = bbs[f].x;
		pt1.y = bbs[f].y;
		pt2.x = bbs[f].x+bbs[f].width;
		pt2.y = bbs[f].y+bbs[f].height;
		cvRectangle(maskCC,pt1,pt2, CV_RGB(255,255,255),2);
		pt1.x = centers[f].x - 3; //Draw the center of mass in black
		pt1.y = centers[f].y - 3;
		pt2.x = centers[f].x +3;
		pt2.y = centers[f].y + 3;
		cvRectangle(maskCC,pt1,pt2, CV_RGB(0,0,0),2);
		
	}
	cvShowImage("backgroundDiff", maskCC);
	char c = cvWaitKey(30);
	cvReleaseImage(&yuvImage);
	cvReleaseImage(&maskImage);
	cvReleaseImage(&maskCC);
	
}