#include "OPIRALibrary.h"
#include "opencv/highgui.h"
#include <iostream>

using namespace OPIRALibrary;
using namespace std; 

/* Constructor */
Registration::Registration() {
	displayImage = false;
}

/* Destructor */
Registration::~Registration() {
	//Clean up any markers we still have hanging around
	for (unsigned int i=0; i<markers.size(); i++) { cvReleaseImage(&markers.at(i).image); };
	markers.clear();
}

/* Add a new marker to our list */
bool Registration::addMarker(string markerName) {
	//Create a new marker object and set it up
	Marker marker; marker.image = cvLoadImage(markerName.c_str());
	if (marker.image==0) { std::cerr << "Cannot Load marker: " << markerName << endl; return false; }
	marker.size = cvGetSize(marker.image);
	marker.name = markerName;

	//Attempt to register it, if it succeeds, add it to the list of trained markers
	if (regAlgorithm->addMarker(marker)) {
		markers.push_back(marker);
	} else {
		std::cerr << "Unable to find any interest points in marker: " << markerName << endl;  return false;
	}

	return true;
}

bool Registration::addResizedMarker(string markerName, int maxLengthSize) {
	//Create a new marker object and set it up
	Marker marker; 
	
	IplImage *origImage	= cvLoadImage(markerName.c_str());
	if (origImage==0) { std::cerr << "Cannot Load marker: " << markerName << endl; return false; }

	//If the marker isn't already resized
	if (max(origImage->width, origImage->height) != maxLengthSize) {
		//Calculate the new size
		if (origImage->width > origImage->height) {
			marker.size = cvSize(maxLengthSize, int(maxLengthSize*float(origImage->height)/float(origImage->width)));
		} else {
			marker.size = cvSize(int(maxLengthSize*float(origImage->width)/float(origImage->height)), maxLengthSize);
		}
		//Resize the image
		marker.image = cvCreateImage(marker.size, origImage->depth, origImage->nChannels);
		cvResize(origImage, marker.image);
		cvReleaseImage(&origImage);
	} else {
		//If it is scaled, just copy the reference
		marker.image = origImage;
	}

	marker.size = cvGetSize(marker.image);
	marker.name = markerName;

	//Attempt to register it, if it succeeds, add it to the list of trained markers
	if (regAlgorithm->addMarker(marker)) {
		markers.push_back(marker);
	} else {
		std::cerr << "Unable to find any interest points in marker: " << markerName << endl;  return false;
	}

	return true;
}

bool Registration::addScaledMarker(string markerName, int maxLengthScale) {
	//Create a new marker object and set it up
	Marker marker; marker.image = cvLoadImage(markerName.c_str());
	if (marker.image==0) { std::cerr << "Cannot Load marker: " << markerName << endl; return false; }
	
	//If the marker isn't already scaled
	if (max(marker.image->width, marker.image->height) != maxLengthScale) {
		if (marker.image->width > marker.image->height) {
			marker.size = cvSize(maxLengthScale, int(maxLengthScale*float(marker.image->height)/float(marker.image->width)));
		} else {
			marker.size = cvSize(int(maxLengthScale*float(marker.image->width)/float(marker.image->height)), maxLengthScale);
		}
	} else {
		marker.size = cvGetSize(marker.image);
	}

	marker.name = markerName;

	//Attempt to register it, if it succeeds, add it to the list of trained markers
	if (regAlgorithm->addMarker(marker)) {
		markers.push_back(marker);
	} else {
		std::cerr << "Unable to find any interest points in marker: " << markerName << endl;  return false;
	}

	return true;
}

bool Registration::addResizedScaledMarker(string markerName, int maxLengthSize, int maxLengthScale) {
	//Create a new marker object and set it up
	Marker marker; 
	
	IplImage *origImage	= cvLoadImage(markerName.c_str());
	if (origImage==0) { std::cerr << "Cannot Load marker: " << markerName << endl; return false; }

	//If the marker isn't already resized
	if (max(origImage->width, origImage->height) != maxLengthSize) {
		//Calculate the new size
		if (origImage->width > origImage->height) {
			marker.size = cvSize(maxLengthSize, int(maxLengthSize*float(origImage->height)/float(origImage->width)));
		} else {
			marker.size = cvSize(int(maxLengthSize*float(origImage->width)/float(origImage->height)), maxLengthSize);
		}
		//Resize the image
		marker.image = cvCreateImage(marker.size, origImage->depth, origImage->nChannels);
		cvResize(origImage, marker.image);
		cvReleaseImage(&origImage);
	} else {
		//If it is scaled, just copy the reference
		marker.image = origImage;
	}

	//If the marker isn't already scaled
	if (max(marker.image->width, marker.image->height) != maxLengthScale) {
		if (marker.image->width > marker.image->height) {
			marker.size = cvSize(maxLengthScale, int(maxLengthScale*float(marker.image->height)/float(marker.image->width)));
		} else {
			marker.size = cvSize(int(maxLengthScale*float(marker.image->width)/float(marker.image->height)), maxLengthScale);
		}
	} else {
		marker.size = cvGetSize(marker.image);
	}

	marker.name = markerName;

	//Attempt to register it, if it succeeds, add it to the list of trained markers
	if (regAlgorithm->addMarker(marker)) {
		markers.push_back(marker);
	} else {
		std::cerr << "Unable to find any interest points in marker: " << markerName << endl;  return false;
	}

	return true;
}

/* Remove a marker from our list */
bool Registration::removeMarker(string markerName) {
	//Loop through until we find the marker in question
	for (unsigned int i=0; i<markers.size(); i++) {
		if (markers.at(i).name == markerName) {
			//Remove it from the registration algorithm and clean up
			regAlgorithm->removeMarker(markers.at(i));
			cvReleaseImage(&markers.at(i).image);
			markers.erase(markers.begin()+i);
			return true;
		}
	}
	return false;
}

/* Find a marker in our list */
Marker Registration::getMarker(string filename) {
	//Attempt to find marker and return it
	for (unsigned int i=0; i<markers.size(); i++) {
		if (markers.at(i).name == filename) return markers.at(i);
	}
	//If not return an empty Marker
	return Marker();
}

/* Default perform registration function returns nothing */
vector<MarkerTransform> Registration::performRegistration(IplImage* frame_input, CvMat* captureParams, CvMat* captureDistortion) 
{	return vector<MarkerTransform>(); };


/*Convert from a homography to an OpenCV Transformation Matrix */
CvMat* Registration::homographyToCvTransMat(CvMat* H, CvMat* camParams) {
	//Accessor for the homography
	float *h = (float*) H->data.ptr;

	//Calculate lambda
	CvMat *camParamsInv = cvCreateMat(3,3,CV_64F);
	cvInvert(camParams, camParamsInv);
	double *c = (double*)camParamsInv->data.ptr;
	double mh1 = c[0]*h[0] + c[1]*h[3] + c[2]*h[6];
	double mh2 = c[3]*h[0] + c[4]*h[3] + c[5]*h[6];
	double mh3 = c[6]*h[0] + c[7]*h[3] + c[8]*h[6];
	double lambda = 1/ sqrt(mh1*mh1 + mh2*mh2 + mh3*mh3);

	//Calculate R1, R2, R3 and T
	CvMat *camExtr = cvCreateMat(4,4,CV_64FC1);
	double *M = (double*)camExtr->data.ptr;

	// R1
	M[0] = lambda*mh1;
	M[4] = lambda*mh2;
	M[8] = lambda*mh3;
	// R2
	M[1] = lambda*(c[0]*h[1] + c[1]*h[4] + c[2]*h[7]);
	M[5] = lambda*(c[3]*h[1] + c[4]*h[4] + c[5]*h[7]);
	M[9] = lambda*(c[6]*h[1] + c[7]*h[4] + c[8]*h[7]);
	// R3
	M[2] = M[4]*M[9] - M[8]*M[5];
	M[6] = M[8]*M[1] - M[0]*M[9];
	M[10] = M[0]*M[5] - M[4]*M[1];
	// t
	M[3]  = lambda*(c[0]*h[2] + c[1]*h[5] + c[2]*h[8]);
	M[7]  = lambda*(c[3]*h[2] + c[4]*h[5] + c[5]*h[8]);
	M[11] = lambda*(c[6]*h[2] + c[7]*h[5] + c[8]*h[8]);

	M[12]=M[13]=M[14]=0; M[15]=1;

	cvReleaseMat(&camParamsInv);
	return camExtr;
}


/*Process Matches to find the OpenCV transformation matrix*/
CvMat* Registration::matchesToCvTransMat(PointMatches m, CvMat* captureParams, CvMat* captureDistortion) {
	//Set up the OpenCV transformation matrix as an identity matrix
	CvMat *cvTransMat = cvCreateMat(4,4, CV_32FC1); cvSetIdentity(cvTransMat);

	//Make the object points homogeneous
	CvMat objectPoints2D = cvMat( m.count, 1, CV_32FC2, m.featMarker );
	CvMat *objectPoints3D = cvCreateMat( m.count, 1, CV_32FC3);
	cvConvertPointsHomogeneous(&objectPoints2D, objectPoints3D);

	//Found points in the scene
	CvMat imagePoints = cvMat(m.count,1, CV_32FC2, m.featImage);

	//Set the translation vector and rotation matrix to point inside the cvTransMat struct
	CvMat *transVector = cvCreateMatHeader(1, 3, CV_32F); cvGetSubRect(cvTransMat, transVector, cvRect(3,0,1,3));
	CvMat *rotMat = cvCreateMatHeader(3,3, CV_32FC1); cvGetSubRect(cvTransMat, rotMat, cvRect(0,0,3,3));
	
	//Find the Extrinsic Camera Parameters, and convert the rotation vector to the rotation matrix
	CvMat *rotVector= cvCreateMat(1, 3, CV_32F); 
	cvFindExtrinsicCameraParams2(objectPoints3D, &imagePoints, captureParams, captureDistortion, rotVector, transVector);
	cvRodrigues2(rotVector, rotMat);
	
	//Clean up
	cvReleaseMat(&objectPoints3D); cvReleaseMat(&rotVector);
	cvReleaseMatHeader(&transVector); cvReleaseMatHeader(&rotMat);

	return cvTransMat;
}

/* Find the good homography from a set of points */
CvMat* Registration::getGoodHomography(PointMatches bestMatch, CvMat* captureParams, CvMat* captureDistortion, CvSize markerSize) {
	//Get the OpenCV Transformation matrix
	CvMat *cvTransMat = matchesToCvTransMat(bestMatch, captureParams, captureDistortion);
	//Call the overloaded function which uses the OpenCV Transformation matrix
	CvMat *homography = getGoodHomography(cvTransMat, captureParams, captureDistortion, markerSize);
	//Clean up
	cvReleaseMat(&cvTransMat);
	return homography;
}

/* Find the good homography using the OpenCV Transformation Matrix */
CvMat* Registration::getGoodHomography(CvMat *cvTransMat, CvMat* captureParams, CvMat* captureDistortion, CvSize markerSize) {
	//Set up the homography
	CvMat *homography = cvCreateMat(3,3,CV_32FC1);

	//Set up the corners on the marker and the location of the corners in the frame
	CvMat *frameCorners = cvCreateMat(4, 2, CV_32FC1);
	CvMat *markerCorners = cvCreateMat(4,2, CV_32FC1); cvZero(markerCorners);
	markerCorners->data.fl[2] = markerCorners->data.fl[4] = (float)markerSize.width;
	markerCorners->data.fl[5] = markerCorners->data.fl[7] = (float)markerSize.height;

	//Set the translation vector and rotation matrix to point inside the cvTransMat struct
	CvMat *transVector = cvCreateMatHeader(1, 3, CV_32F); cvGetSubRect(cvTransMat, transVector, cvRect(3,0,1,3));
	CvMat *rotMat = cvCreateMatHeader(3,3, CV_32FC1); cvGetSubRect(cvTransMat, rotMat, cvRect(0,0,3,3));
	
	//Convert the Rotation Matrix to a vector
	CvMat *rotVector = cvCreateMat(1,3,CV_32FC1); cvRodrigues2(rotMat, rotVector);

#if 1
	/*OpenCV 2.0 Memory Leak Workaround*/
	CvMat* frameCorners64 = cvCreateMat(frameCorners->rows, frameCorners->cols, CV_64FC1);
	CvMat* markerCorners64 = cvCreateMat(markerCorners->rows, markerCorners->cols+1, CV_64FC1);
	cvConvertPointsHomogeneous(markerCorners, markerCorners64);
	//Find the location of the marker corners
	cvProjectPoints2(markerCorners64, rotVector, transVector, captureParams, captureDistortion, frameCorners64);
	cvConvert(frameCorners64, frameCorners); 
	cvReleaseMat(&markerCorners64); cvReleaseMat(&frameCorners64);
#else 
	//Find the location of the marker corners
	cvProjectPoints2(markerCorners, rotVector, transVector, captureParams, captureDistortion, frameCorners);
#endif

	//Get the perspective Transform from the matches
	cvGetPerspectiveTransform((CvPoint2D32f*)&frameCorners->data.ptr[0], (CvPoint2D32f*)&markerCorners->data.ptr[0], homography);
	cvInvert(homography, homography);

	//Clean up the matrices
	cvReleaseMat(&markerCorners); cvReleaseMat(&frameCorners); 
	cvReleaseMat(&rotVector); cvReleaseMatHeader(&transVector); cvReleaseMatHeader(&rotMat);

	return homography;
}

/*Create a MarkerTransform object from a set of points*/
MarkerTransform Registration::computeMarkerTransform(PointMatches pMatch, int index, CvSize frameSize, CvMat *captureParams, CvMat *captureDistortion) {
	MarkerTransform mt;
	
	//Get the OpenCV Transformation Matrix
	CvMat *cvTransMat;
	if (markers.at(index).size.width != markers.at(index).image->width || markers.at(index).size.height != markers.at(index).image->height) {
		PointMatches _pMatch; _pMatch = ScaleMatches(float(markers.at(index).size.width)/float(markers.at(index).image->width), pMatch);
		cvTransMat = matchesToCvTransMat(_pMatch, captureParams, captureDistortion);
		_pMatch.clear();
	} else {
		//Get the OpenCV Transformation Matrix
		cvTransMat = matchesToCvTransMat(pMatch, captureParams, captureDistortion);
	}

	//Set up the marker, score and homography
	mt.marker = markers.at(index);
	mt.score = pMatch.count;
	mt.homography = getGoodHomography(cvTransMat, captureParams, captureDistortion, markers.at(index).size);
	
	//Get the OpenGL Matrices
	mt.projMat = calcProjection(captureParams, captureDistortion, frameSize);
	mt.viewPort = calcViewpoint(captureParams, captureDistortion, frameSize);
	mt.transMat = calcTransform(cvTransMat);

	//Clean up
	cvReleaseMat(&cvTransMat);
	return mt;
}

/*Create a MarkerTransform object from a homography*/
MarkerTransform Registration::computeMarkerTransform(CvMat* homography, int matchCount, int index, CvSize frameSize, CvMat *captureParams, CvMat *captureDistortion) {
	MarkerTransform mt;

	//Set up the marker, score and homography
	mt.marker = markers.at(index);
	mt.score = matchCount;
	mt.homography = cvCloneMat(homography);

	//Get the OpenCV Transformation Matrix
	CvMat *cvTransMat = homographyToCvTransMat(homography, captureParams);

	//Get the OpenGL Matrices
	mt.projMat = calcProjection(captureParams, captureDistortion, frameSize);
	mt.viewPort = calcViewpoint(captureParams, captureDistortion, frameSize);
	mt.transMat = calcTransform(cvTransMat);
	cvReleaseMat(&cvTransMat);

	return mt;
}

PointMatches Registration::ScaleMatches(float scale, PointMatches p) {
	PointMatches _p; _p.clone(p);
	for (int i=0; i<_p.count; i++) {
		_p.featMarker[i].x*=scale; 
		_p.featMarker[i].y*=scale;
	}
	return _p;
}