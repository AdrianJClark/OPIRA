#include "RegistrationAlgorithms\SURF.h"

#include "surflib.h"
#include "fasthessian.h"
#include "image.h"
#include <opencv2\highgui\highgui.hpp>

using namespace OPIRALibrary;
using namespace surf;


	SURF::SURF(bool rotationInvariant, bool descriptor128bit, bool doubleImSize, int nOctaves, int nSamplingSteps, int initalLobeSize, double blobThres, int windowSize) {
		upright = !rotationInvariant;
		extended = descriptor128bit;
		doubleImageSize = doubleImSize;
		
		octaves = nOctaves;
		samplingStep = nSamplingSteps;
		initLobe = initalLobeSize;
		thres = blobThres;
		indexSize = windowSize;
	}

	SURF::~SURF() { 
	}

	RegistrationAlgorithm* SURF::clone() {
	SURF* r = new SURF(*this);
	r->trainedMarkers = RegistrationAlgorithm::cloneTrainedMarker(trainedMarkers);
	r->allMarkers = RegistrationAlgorithm::cloneTrainedMarker(allMarkers);
	return r;
	}

	string SURF::getName() {
		return "SURF";
	}

RegistrationAlgorithm::FeatureVector SURF::performRegistration(IplImage* image) {
	FeatureVector features; 

	IplImage *doubleIm = cvCreateImage(cvGetSize(image), IPL_DEPTH_64F, 1);
	//If it's a colour image
	if (image->nChannels > 1) {
		//Convert it to black and white before scaling to a float
		IplImage *imageBW = cvCreateImage(cvGetSize(image), image->depth, 1);
		cvConvertImage(image, imageBW);
		cvConvertScale(imageBW, doubleIm, 1.0/255.0);
		cvReleaseImage(&imageBW);
	} else {
		//Else, just scale it
		cvConvertScale(image, doubleIm, 1.0/255.0);
	}
	

	Image *im = new Image(doubleIm->width, doubleIm->height);

	double **pixels = im->getPixels();
	for (int i=0; i<doubleIm->height; i++) {
		memcpy(pixels[i], doubleIm->imageData+(sizeof(double)*i*doubleIm->width), sizeof(double)*doubleIm->width);
	}

	// Create the integral image
	Image *iim = new Image(im, false);

	
	//Create the image points vector
	vector<Ipoint> *ipts = new vector<Ipoint>;
	ipts->reserve(1000);

	// Extract interest points with Fast-Hessian
	FastHessian *fh = new FastHessian(iim, // pointer to integral image
                 *ipts,
                 thres, // blob response threshold 
                 doubleImageSize, // double image size flag 
                 initLobe * 3, // 3 times lobe size equals the mask size 
                 samplingStep, // subsample the blob response map 
                 octaves // number of octaves to be analysed 
				 );
	fh->getInterestPoints();

  // Initialise the SURF descriptor
  Surf *des = new Surf(iim, // pointer to integral image   
           doubleImageSize, // double image size flag  
           upright, // rotation invariance or upright 
           extended, // use the extended descriptor 
           indexSize // square size of the descriptor window (default 4x4)
		   );

	features.descriptorLength = des->getVectLength();

	for (unsigned int i=0; i<ipts->size(); i++) {
		Feature f(features.descriptorLength); f.position = cvPoint2D32f(ipts->at(i).x, ipts->at(i).y);
		
		// set the curreinterest point
		des->setIpoint(&ipts->at(i));
		// assign reproducible orientation
		des->assignOrientation();
		// make the SURF descriptor
		des->makeDescriptor();
		
		for (int j=0; j< features.descriptorLength; j++)  f.descriptor[j] = ipts->at(i).ivec[j];
		features.push_back(f);
	}

	delete des;
	delete fh;
	delete ipts;
	
	
	delete im;
	delete iim;
	
	
	cvReleaseImage(&doubleIm);

	return features;
	}
