#ifndef CAPTURELIBRARY_H
#define CAPTURELIBRARY_H

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "videoInput.h"
#include <iostream>

/* The Capture base class */
class Capture {
public:
	//The default constructor
	Capture() { undistort=false;}

	//The destructor cleans up if necessary
	virtual ~Capture() { }

	//When asked for an image we return 0 as this is an abstract base class
	virtual IplImage* getFrame() { return 0;}
	
	//Return the camera parameters
	CvMat* getParameters() { 
		return params.intrinsics;
	}

	//Return the camera Distortion
	CvMat* getDistortion() { 
		if (undistort) return 0; 
		else return params.distortion; 
	}

	//Get the cameras width and height
	int getWidth() { return params.width; }
	int getHeight() { return params.height; }
	
	//Toggle undistortion
	bool getUndistort() { return undistort; }
	void setUndistort(bool _undistort) { 
		//We can't undistort if we have no parameters
		if (params.intrinsics==0) {undistort = false; return;}
		undistort = _undistort; 
		//Set up the intrinsic parameters for undistortion
		params.intrinsics->data.db[2] = undistort?params.width/2.0:params.principalX;
		params.intrinsics->data.db[5] = undistort?params.height/2.0:params.principalY;
	}

protected:
	//A structure containing information about our Capture Object
	struct CaptureParams {
		CvMat* intrinsics, *distortion;
		CvMat* undistortX, *undistortY;
	
		int width, height;
		double principalX, principalY;

		CaptureParams() {intrinsics = distortion = undistortX = undistortY =0; width=height=0;}
		~CaptureParams() {
			if (intrinsics) cvReleaseMat(&intrinsics);
			if (distortion) cvReleaseMat(&distortion);
			if (undistortX) cvReleaseMat(&undistortX);
			if (undistortY) cvReleaseMat(&undistortY);
		}
	} params;

	bool undistort;
	
	//Load capture parameters from a file
	bool loadCaptureParams(char *filename) {
		CvFileStorage* fs = cvOpenFileStorage( filename, 0, CV_STORAGE_READ );
		if (fs==0) return false; 

		CvFileNode* fileparams;
		//Read the Image Width
		fileparams = cvGetFileNodeByName( fs, NULL, "image_width" );
		params.width = cvReadInt(fileparams,-1);
		//Read the Image Height
		fileparams = cvGetFileNodeByName( fs, NULL, "image_height" );
		params.height = cvReadInt(fileparams,-1);
		//Read the Camera Parameters
		fileparams = cvGetFileNodeByName( fs, NULL, "camera_matrix" );
		params.intrinsics = (CvMat*)cvRead( fs, fileparams );
		//Read the Camera Distortion 
		fileparams = cvGetFileNodeByName( fs, NULL, "distortion_coefficients" );
		params.distortion = (CvMat*)cvRead( fs, fileparams );
		cvReleaseFileStorage( &fs );

		//Store the principal points
		params.principalX = params.intrinsics->data.db[2]; params.principalY = params.intrinsics->data.db[5]; 

		initialiseParameters(params.width, params.height);
		return true;
	}

	//Initialise parameters and if need be, reshape matrices
	void initialiseParameters(int width, int height) {
		//If this isn't the first time calling this, do so
		if (params.undistortX) cvReleaseMat(&params.undistortX);
		if (params.undistortY) cvReleaseMat(&params.undistortY);

		//Make sure undistortion is off
		setUndistort(false);

		//If we need to modify the params do it
		if (width != params.width || height != params.height) {
			float oWidth = (float)params.width, oHeight = (float)params.height;
			float nWidth = (float)width, nHeight = (float)height;
			
			float scaleFactor = (oWidth/oHeight > nWidth/nHeight)?oHeight/nHeight:oWidth/nWidth;
			params.intrinsics->data.db[0]/= scaleFactor;	params.intrinsics->data.db[4]/= scaleFactor;
			params.intrinsics->data.db[2]/= scaleFactor;	params.intrinsics->data.db[5]/= scaleFactor;
			params.width = width; params.height = height; 
		}

		//Store the principal points
		params.principalX = params.intrinsics->data.db[2]; params.principalY = params.intrinsics->data.db[5]; 

		//Initialize Undistortion Maps
		params.undistortX = cvCreateMat(params.height, params.width, CV_32F);
		params.undistortY = cvCreateMat(params.height, params.width, CV_32F);
		cvInitUndistortMap(params.intrinsics, params.distortion, params.undistortX, params.undistortY);

		//Turn undistortion on
		setUndistort(true);
	}

};

class CvCamera: public Capture {
public:
	CvCamera(int camIndex) {init(camIndex, 0);}
	CvCamera(int camIndex, char* parametersFile) {init(camIndex, parametersFile);}
	~CvCamera() {cvReleaseCapture(&cap);}

	IplImage* getFrame()	{
		if (params.width ==0 || params.height ==0) return 0;
		IplImage* newFrame = cvCreateImage(cvSize(params.width, params.height), IPL_DEPTH_8U, 3);
		cvCopy(cvQueryFrame(cap), newFrame);
		
		if (undistort) {
			IplImage *unDistortedFrame = cvCreateImage(cvGetSize(newFrame), newFrame->depth, newFrame->nChannels);
			cvRemap( newFrame, unDistortedFrame, params.undistortX, params.undistortY);
			cvReleaseImage(&newFrame); newFrame = unDistortedFrame;
		}

		return newFrame;
	}

	void init(int cameraIndex, char* parametersFile) {
		//Set the camera index
		camIndex = cameraIndex; 
		
		//Initialize the videoinput library
		cap = cvCreateCameraCapture(camIndex);

		bool loadedParams = false;
		//If we're using the parameter files load them
		if (parametersFile && strlen(parametersFile)>0)
			loadedParams = loadCaptureParams(parametersFile);
	
		//If we've specified a size, use that
		if (loadedParams) initialiseParameters(320,240);
		params.width = 320; params.height = 240; 

		if (params.width == 0 || params.height == 0) std::cerr << "Invalid Camera Resolution" << std::endl;

	}
private:
	CvCapture *cap;
	int camIndex;
};

class Camera: public Capture {
public:
	//Camera Constructors
	Camera() {init(0, cvSize(320,240), 0);}
	Camera(char* parametersFile) {init(0, cvSize(-1,-1), parametersFile);}
	Camera(int cameraIndex, char* parametersFile) {init(cameraIndex, cvSize(-1,-1), parametersFile);};
	Camera(int cameraIndex) {init(cameraIndex, cvSize(320,240), 0);};
	Camera(CvSize imgSize) {init(0, imgSize, 0);};
	Camera(CvSize imgSize, char* parametersFile) {init(0, imgSize, parametersFile);};
	Camera(int cameraIndex, CvSize imgSize) {init(cameraIndex, imgSize, 0);};
	Camera(int cameraIndex, CvSize imgSize, char* parametersFile) {init(cameraIndex, imgSize, parametersFile);};
	~Camera() { delete vi; }

	IplImage* getFrame()	{
		if (vi) {
			if (camIndex >= vi->devicesFound || params.width ==0 || params.height ==0) return 0;
			IplImage* newFrame = cvCreateImage(cvSize(params.width, params.height), IPL_DEPTH_8U, 3);
			bool gotFrame = vi->getPixels(camIndex, (unsigned char*)newFrame->imageData, false, true);

			if (!gotFrame) { cvReleaseImage(&newFrame); return 0; }

			if (undistort) {
				IplImage *unDistortedFrame = cvCreateImage(cvGetSize(newFrame), newFrame->depth, newFrame->nChannels);
				cvRemap( newFrame, unDistortedFrame, params.undistortX, params.undistortY);
				cvReleaseImage(&newFrame); newFrame = unDistortedFrame;
			}
			return newFrame;
		}
		return 0;
	}

	/*Find out if auto white balancing is on*/
	bool getAutoWhiteBalance() {
		long pMin, pMax, pStep, pVal, pFlags, pDef;
		//Query the camera
		vi->getVideoSettingFilter(camIndex, vi->propWhiteBalance, pMin, pMax, pStep, pVal, pFlags, pDef); 
		//if pFlags is 1, auto white balancing is on
		return (pFlags==1);
	}

	/*Turn white balancing on or off*/
	void setAutoWhiteBalance(bool whiteBal, int value=-1) {
		//Get the current value
		long pMin, pMax, pStep, pVal, pFlags, pDef;
		vi->getVideoSettingFilter(camIndex, vi->propWhiteBalance, pMin, pMax, pStep, pVal, pFlags, pDef); 

		if (value>=pMin && value<=pMax) pVal = value;
		//The value is 1 if white balancing is on, 2 if it isn't
		if (whiteBal)
			vi->setVideoSettingFilter(camIndex, vi->propWhiteBalance, pVal, 1); 
		else
			vi->setVideoSettingFilter(camIndex, vi->propWhiteBalance, pVal, 2); 
		
	}

protected:
	void init(int cameraIndex, CvSize imgSize, char* parametersFile) {
		//Set the camera index
		camIndex = cameraIndex;
		
		//Initialize the videoinput library
		vi = new videoInput(); vi->setVerbose(false); vi->setUseCallback(true);

		bool loadedParams = false;
		//If we're using the parameter files load them
		if (parametersFile && strlen(parametersFile)>0)
			loadedParams = loadCaptureParams(parametersFile);
	
		//If we've specified a size, use that
		if (imgSize.width !=-1 && imgSize.height !=-1) {
			vi->setupDevice(cameraIndex, imgSize.width, imgSize.height);
			if (loadedParams) initialiseParameters(imgSize.width, imgSize.height);
			params.width = imgSize.width; params.height = imgSize.height; 
		} else {
			//Otherwise use the camera parameter file
			vi->setupDevice(cameraIndex, params.width, params.height);
		}

		if (vi->devicesFound==0) std::cerr << "No Cameras Found" << std::endl;
		else if (cameraIndex >= vi->devicesFound) std::cerr << "Invalid Camera Index" << std::endl;
		else if (params.width == 0 || params.height == 0) std::cerr << "Invalid Camera Resolution" << std::endl;

	}
	videoInput *vi;
	int camIndex;
};

class Video: public Capture {
public:
	Video(char *videoFile) {
		captureObj = cvCreateFileCapture(videoFile);
		if (!captureObj) {std::cerr << "Unable to load video file: " << videoFile << std::endl; exit(0); }

		params.width = (int)cvGetCaptureProperty(captureObj, CV_CAP_PROP_FRAME_WIDTH); 
		params.height = (int)cvGetCaptureProperty(captureObj, CV_CAP_PROP_FRAME_HEIGHT);	
	}

	Video(char *videoFile, char* parametersFile) {
		captureObj = cvCreateFileCapture(videoFile);
		if (!captureObj) { std::cerr << "Unable to load video file: " << videoFile << std::endl; exit(0); }

		if (!loadCaptureParams(parametersFile)) { std::cerr << "Unable to camera parameter file: " << parametersFile << std::endl; cvReleaseCapture(&captureObj); exit(0); }

		initialiseParameters((int)cvGetCaptureProperty(captureObj, CV_CAP_PROP_FRAME_WIDTH), (int)cvGetCaptureProperty(captureObj, CV_CAP_PROP_FRAME_HEIGHT));
	}

	~Video() { if (captureObj) cvReleaseCapture( &captureObj ); }
	
	IplImage* getFrame() { 	
		IplImage *newFrame;

		//Make sure we have a capture object
		if (captureObj) {
			if( !cvGrabFrame( captureObj )) return 0;
		}
		//Make a copy of the image
		newFrame = cvCloneImage(cvRetrieveFrame( captureObj ));

		if (undistort) {
			//If we are undistorting, create a new Frame, undistort into it, then release the original
			//Frame and set it's pointer to the undistortedFrame
			IplImage *unDistortedFrame = cvCreateImage(cvGetSize(newFrame), newFrame->depth, newFrame->nChannels);
			cvRemap( newFrame, unDistortedFrame, params.undistortX, params.undistortY);
			//cvUndistort2( newFrame, unDistortedFrame, captureParams, captureDistortion );
			cvReleaseImage(&newFrame); newFrame = unDistortedFrame;
		}

		if( newFrame->origin == IPL_ORIGIN_BL ) {
			//If the frame is upside down, flip it the right way up
			IplImage *flippedFrame = cvCreateImage( cvGetSize(newFrame), newFrame->depth, newFrame->nChannels);
			cvFlip( newFrame, flippedFrame, 0 );
			cvReleaseImage(&newFrame); newFrame = flippedFrame;
		}

		//Return the new frame
		return newFrame;
	}

private:
	CvCapture *captureObj;
};

#endif
