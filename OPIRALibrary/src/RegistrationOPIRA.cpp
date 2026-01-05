#include "OPIRALibrary.h"
#include "opencv/highgui.h"

using namespace OPIRALibrary;
using namespace std; 

RegistrationOPIRA::RegistrationOPIRA(RegistrationAlgorithm *registrationAlgorithm): RegistrationOpticalFlow(registrationAlgorithm) {}

RegistrationOPIRA::~RegistrationOPIRA() {}

vector<MarkerTransform> RegistrationOPIRA::performRegistration(IplImage* frame_input, CvMat* captureParams, CvMat* captureDistortion) 
	{
	vector<MarkerTransform> retVal;
	string windowText;
	if (previousImage==0) previousImage = cvCreateImage(cvGetSize(frame_input), frame_input->depth, frame_input->nChannels);

	vector<PointMatches> matches = regAlgorithm->findAllMatches(frame_input);

	currentImage = cvCreateImage(cvGetSize(frame_input), IPL_DEPTH_8U, 1); cvConvertImage(frame_input, currentImage);
	currentPyramid = cvCreateImage(cvGetSize(currentImage), IPL_DEPTH_32F, 1);

	for (unsigned int i=0; i<matches.size(); i++) {
		PointMatches *bestMatch=0;
		PointMatches pMatch = Ransac(matches.at(i));

		if (pMatch.count >=minRegistrationCount) { bestMatch = &pMatch; windowText = regAlgorithm->getName(); }
		
		//If we are able to match at least 2 points through optical flow
		PointMatches optFlow = opticalFlow(previousMatches.at(i));
		if (optFlow.count >=minOptFlowCount && optFlow.count>=pMatch.count) { bestMatch = &optFlow; windowText = "Optical Flow"; }
		previousMatches.at(i).clear();

		if (bestMatch !=0) {
			//Perform OPIRA Registration
			CvMat* homography = getGoodHomography(*bestMatch, captureParams, captureDistortion, markers.at(i).size);
			PointMatches unSift = undistortRegister(frame_input, i, homography, cvGetSize(markers.at(i).image));
			if (unSift.count>=minRegistrationCount && unSift.count > bestMatch->count) { bestMatch = &unSift; windowText = "OPIRA";}  else { unSift.clear(); }
			cvReleaseMat(&homography);

			//Copy the points into the previous array
			previousMatches.at(i).clone(*bestMatch);

			//Display the matches if we feel so inclined
			if (displayImage) displayMatches(markers.at(i).image, frame_input, *bestMatch, regAlgorithm->getName() + " OPIRA", windowText, 1.0);

			//Calculate the OGL viewpoint and projection and transformation matrices
			retVal.push_back(computeMarkerTransform(*bestMatch, i, cvGetSize(frame_input), captureParams, captureDistortion));
			unSift.clear();
		}

		//Cleanup
		matches.at(i).clear();
		pMatch.clear();
		optFlow.clear();
	}

	cvReleaseImage(&previousImage); previousImage = currentImage; currentImage = 0;
	cvReleaseImage(&previousPyramid); previousPyramid = currentPyramid; currentPyramid = 0;
	return retVal;
}

PointMatches RegistrationOPIRA::undistortRegister(IplImage* frame_input, int index, CvMat* homography, CvSize s) {
	//Create an image for undistortion
	if (s.width <=0 || s.height <=0) s=cvGetSize(frame_input);
	IplImage *unDistort = cvCreateImage(s, frame_input->depth, frame_input->nChannels);
	//Perform the undistortion
	cvWarpPerspective(frame_input, unDistort, homography, 1+8+16);
	//Find new Matches
	PointMatches matches =  regAlgorithm->findMatches(unDistort, index);
	if (displayImage) displayMatches(markers.at(index).image , unDistort, matches, "OPIRA", regAlgorithm->getName(),1.0, 1.0);

	//Release the undistorted image
	cvReleaseImage(&unDistort);

	//If we find matches
	if (matches.count > 0) {
		CvMat mMatches = cvMat(matches.count, 1, CV_32FC2, matches.featImage);
		//Transform the matches from undistorted to distorted image
		cvPerspectiveTransform(&mMatches, &mMatches, homography);

		PointMatches rMatches = Ransac(matches);
		matches.clear();

		//Run ransac over the features
		return rMatches;
	}

	matches.clear();
	return PointMatches();
}