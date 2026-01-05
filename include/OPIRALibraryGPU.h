#ifndef OPIRALIBRARYGPU_H
#define OPIRALIBRARYGPU_H

#include "OPIRALibrary.h"
#include "opencv2/gpu/gpu.hpp"

namespace OPIRALibrary {

	class RegistrationAlgorithmGPU: public RegistrationAlgorithm {
	public:
		RegistrationAlgorithmGPU();
		~RegistrationAlgorithmGPU();
		virtual std::string getName();
		virtual bool addMarker(Marker marker);
		virtual bool removeMarker(Marker marker);
		virtual RegistrationAlgorithmGPU *clone();
		virtual std::vector<PointMatches> findAllMatches(IplImage* image);
		virtual PointMatches findMatches(IplImage* image, int index);
	protected:
		struct RegisteredImageGPU {
			std::vector<cv::KeyPoint> keypoints; std::vector<float> descriptors;
			cv::gpu::GpuMat gpu_keypoints, gpu_descriptors;
			std::string markerFilename;
			cv::gpu::BruteForceMatcher_GPU<cv::L2<float>> matcher;
		};

		std::vector <RegisteredImageGPU> trainedMarkers;

		virtual RegisteredImageGPU performRegistrationGPU(IplImage* image);
	public:
		std::vector<RegisteredImageGPU> cloneTrainedMarker(std::vector<RegisteredImageGPU> trainedMarkers);
		RegisteredImageGPU cloneTrainedMarker(RegisteredImageGPU trainedMarker);
	private: 
		int findMarkerIndex(int featureIndex);
	};
};

#endif