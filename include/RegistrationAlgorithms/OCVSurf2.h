#ifndef POINTBASEDOCVSURFALGORITHMORIG_H
#define POINTBASEDOCVSURFALGORITHMORIG_H

#include "OPIRALibrary.h"
using namespace OPIRALibrary;

class OCVSurfOrig: public RegistrationAlgorithm {
public:
	OCVSurfOrig(bool descriptor128bit=false, int nOctaves=4, int nSamplingSteps = 2, double blobThres=400);
	~OCVSurfOrig();
	RegistrationAlgorithm* clone();

	std::string getName();
	FeatureVector performRegistration(IplImage* image);

	

private:
	cv::SURF surf;
	int extended; // 0 means basic descriptors (64 elements each),
                  // 1 means extended descriptors (128 elements each)
    double hessianThreshold; // only features with keypoint.hessian larger than that are extracted.
                  // good default value is ~300-500 (can depend on the average
                  // local contrast and sharpness of the image).
                  // user can further filter out some features based on their hessian values
                  // and other characteristics
    int octaves; // the number of octaves to be used for extraction.
                  // With each next octave the feature size is doubled (3 by default)
    int octaveLayers; // The number of layers within each octave (4 by default)
};

#endif