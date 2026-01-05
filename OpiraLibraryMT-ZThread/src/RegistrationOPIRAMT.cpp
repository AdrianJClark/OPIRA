#include "OPIRALibraryMT-ZThread.h"
#include "highgui.h"
#include <iostream>

using namespace OPIRALibrary;
RegistrationOPIRAMT::RegistrationOPIRAMT(RegistrationAlgorithm *registrationAlgorithm): RegistrationOPIRA(registrationAlgorithm)
{
	regThread = new RegistrationThread(registrationAlgorithm->clone()); thread = new Thread(regThread);
}

bool RegistrationOPIRAMT::removeMarker(string markerName) {
	for (unsigned int i=0; i<markers.size(); i++) {
		if (markers.at(i).name == markerName) {
			//Remove the thread searching for that marker
			regOPIRAThreads.at(i)->stop(); opiraThreads.at(i)->wait(); 
			while (!regOPIRAThreads.at(i)->isFinished()) Sleep(1);
			delete opiraThreads.at(i);
			opiraThreads.erase(opiraThreads.begin()+i);
			regOPIRAThreads.erase(regOPIRAThreads.begin()+i);

			//Remove it from the registration algorithm and clean up
			regThread->removeMarker(markers.at(i));
			cvReleaseImage(&markers.at(i).image);
			markers.erase(markers.begin()+i);

			//Remove previous matches
			previousMatches.at(i).clear();
			previousMatches.erase(previousMatches.begin()+i); 

			return true;
		}
	}
	return false;
}

bool RegistrationOPIRAMT::addMarker(string markerName) {
	//Create a new marker object and set it up
	Marker marker; marker.image = cvLoadImage(markerName.c_str());
	if (marker.image==0) { std::cerr << "Cannot Load marker: " << markerName << endl; return false; }
	marker.size = cvGetSize(marker.image);
	marker.name = markerName;

	//Attempt to register it, if it succeeds, add it to the list of trained markers
	if (regAlgorithm->addMarker(marker)) {
		regThread->addMarker(marker);
		previousMatches.push_back(PointMatches());
		RegistrationThreadOPIRA *r = new RegistrationThreadOPIRA(regAlgorithm->clone(), markers.size(), marker.size);
		regOPIRAThreads.push_back(r); opiraThreads.push_back(new Thread(r));
		markers.push_back(marker);
	} else {
		std::cerr << "Unable to find any interest points in marker: " << markerName << endl;  return false;
	}

	return true;
}

bool RegistrationOPIRAMT::addResizedMarker(string markerName, int maxLengthSize) {
	//Create a new marker object and set it up
	Marker marker; IplImage *origImage	= cvLoadImage(markerName.c_str());
	if (origImage==0) { std::cerr << "Cannot Load marker: " << markerName << endl; return false; }

	//If the marker isn't already scaled
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
		regThread->addMarker(marker);
		previousMatches.push_back(PointMatches());
		RegistrationThreadOPIRA *r = new RegistrationThreadOPIRA(regAlgorithm->clone(), markers.size(), marker.size);
		regOPIRAThreads.push_back(r); opiraThreads.push_back(new Thread(r));
		markers.push_back(marker);
	} else {
		std::cerr << "Unable to find any interest points in marker: " << markerName << endl;  return false;
	}

	return true;
}

bool RegistrationOPIRAMT::addScaledMarker(string markerName, int maxLengthScale) {
	//Create a new marker object and set it up
	Marker marker; marker.image = cvLoadImage(markerName.c_str());
	if (marker.image==0) { std::cerr << "Cannot Load marker: " << markerName << endl; return false; }
	marker.name = markerName;

	//If the marker isn't already scaled
	if (max(marker.image->width, marker.image->height) != maxLengthScale) {
		//Calculate the new size
		if (marker.image->width > marker.image->height) {
			marker.size = cvSize(maxLengthScale, int(maxLengthScale*float(marker.image->height)/float(marker.image->width)));
		} else {
			marker.size = cvSize(int(maxLengthScale*float(marker.image->width)/float(marker.image->height)), maxLengthScale);
		}
	} else {
		//If it is scaled, just copy the reference
		marker.size = cvGetSize(marker.image);
	}

	//Attempt to register it, if it succeeds, add it to the list of trained markers
	if (regAlgorithm->addMarker(marker)) {
		regThread->addMarker(marker);
		previousMatches.push_back(PointMatches());
		RegistrationThreadOPIRA *r = new RegistrationThreadOPIRA(regAlgorithm->clone(), markers.size(), marker.size);
		regOPIRAThreads.push_back(r); opiraThreads.push_back(new Thread(r));
		markers.push_back(marker);
	} else {
		std::cerr << "Unable to find any interest points in marker: " << markerName << endl;  return false;
	}

	return true;
}

bool RegistrationOPIRAMT::addResizedScaledMarker(string markerName, int maxLengthSize, int maxLengthScale) {
	//Create a new marker object and set it up
	Marker marker; IplImage *origImage	= cvLoadImage(markerName.c_str());
	if (origImage==0) { std::cerr << "Cannot Load marker: " << markerName << endl; return false; }

	//If the marker isn't already scaled
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
		//Calculate the new size
		if (marker.image->width > marker.image->height) {
			marker.size = cvSize(maxLengthScale, int(maxLengthScale*float(marker.image->height)/float(marker.image->width)));
		} else {
			marker.size = cvSize(int(maxLengthScale*float(marker.image->width)/float(marker.image->height)), maxLengthScale);
		}
	} else {
		//If it is scaled, just copy the reference
		marker.size = cvGetSize(marker.image);
	}

	marker.name = markerName;

	//Attempt to register it, if it succeeds, add it to the list of trained markers
	if (regAlgorithm->addMarker(marker)) {
		regThread->addMarker(marker);
		previousMatches.push_back(PointMatches());
		RegistrationThreadOPIRA *r = new RegistrationThreadOPIRA(regAlgorithm->clone(), markers.size(), marker.size);
		regOPIRAThreads.push_back(r); opiraThreads.push_back(new Thread(r));
		markers.push_back(marker);
	} else {
		std::cerr << "Unable to find any interest points in marker: " << markerName << endl;  return false;
	}

	return true;
}
RegistrationOPIRAMT::~RegistrationOPIRAMT() 
{
	for (unsigned int i=0; i<regOPIRAThreads.size(); i++) {
		regOPIRAThreads.at(i)->stop(); 
		while (!regOPIRAThreads.at(i)->isFinished()) opiraThreads.at(i)->yield();
		delete opiraThreads.at(i);//delete regOPIRAThreads.at(i);
	}
	opiraThreads.clear(); regOPIRAThreads.clear();

	regThread->stop();
	while (!regThread->isFinished()) thread->yield();
	delete thread;
}

vector<MarkerTransform> RegistrationOPIRAMT::performRegistration(IplImage* frame_input, CvMat* captureParams, CvMat* captureDistortion) 
	{
	vector<MarkerTransform> retVal;
	string windowText;
	if (previousImage==0) previousImage = cvCreateImage(cvGetSize(frame_input), frame_input->depth, frame_input->nChannels);

	vector<PointMatches> matches = regThread->findAllMatches(frame_input);

	currentImage = cvCreateImage(cvGetSize(frame_input), IPL_DEPTH_8U, 1); cvConvertImage(frame_input, currentImage);
	currentPyramid = cvCreateImage(cvGetSize(currentImage), IPL_DEPTH_32F, 1);

	for (unsigned int i=0; i<matches.size(); i++) {
		PointMatches *bestMatch=0;
		PointMatches pMatch = Ransac(matches.at(i));

		if (pMatch.count >=minRegistrationCount) { bestMatch = &pMatch; windowText = regAlgorithm->getName(); }
		
		//If we are able to match at least 2 points through optical flow
		PointMatches optFlow;
		if (i<previousMatches.size()) {
			optFlow.clear(); optFlow = opticalFlow(previousMatches.at(i));
			if (optFlow.count >=minOptFlowCount && optFlow.count>=pMatch.count) { bestMatch = &optFlow; windowText = "Optical Flow"; }
			previousMatches.at(i).clear();
		}

		if (bestMatch !=0 && i < markers.size()) {
			//Perform OPIRA Registration
			CvMat* homography = getGoodHomography(*bestMatch, captureParams, captureDistortion, markers.at(i).size);
			PointMatches unSift = regOPIRAThreads.at(i)->undistortRegister(frame_input, homography, cvGetSize(markers.at(i).image));
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