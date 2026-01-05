#include "RegistrationAlgorithms/OCVGPUSurf.h"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/core/core.hpp"

using namespace OPIRALibrary;
using namespace cv;
using namespace cv::gpu;

OCVGPUSurf::OCVGPUSurf(bool descriptor128bit, int nOctaves, int nOctaveLayers, double blobThresh, int nSamplingSteps) {
	samplingSteps = nSamplingSteps;
	extended = descriptor128bit?1:0;
    hessianThreshold = blobThresh;
    octaves = nOctaves;
    octaveLayers = nOctaveLayers;

	surf = cv::gpu::SURF_GPU(hessianThreshold, octaves, octaveLayers, extended);
}

RegistrationAlgorithmGPU* OCVGPUSurf::clone() {
	OCVGPUSurf* r = new OCVGPUSurf(*this);
	r->trainedMarkers = RegistrationAlgorithmGPU::cloneTrainedMarker(trainedMarkers);
	//r->allMarkers = RegistrationAlgorithmGPU::cloneTrainedMarker(allMarkers);
	return r;
}

string OCVGPUSurf::getName() {
	return "OpenCV SURF";
}

RegistrationAlgorithmGPU::RegisteredImageGPU OCVGPUSurf::performRegistrationGPU(IplImage* _image) {
	RegistrationAlgorithmGPU::RegisteredImageGPU features;
	cv::Mat image(_image);

	GpuMat gpu_image;
	if (_image->nChannels > 1) {
		cv::Mat imageBW; cv::cvtColor(image, imageBW, CV_RGB2GRAY);
		gpu_image = GpuMat(imageBW);
	} else {
		gpu_image = GpuMat(image);
	}

	surf(gpu_image, GpuMat(), features.gpu_keypoints, features.gpu_descriptors);
	surf.downloadKeypoints(features.gpu_keypoints, features.keypoints);
	surf.downloadDescriptors(features.gpu_descriptors, features.descriptors);

	return features;
}

OCVGPUSurf::~OCVGPUSurf() {

}
