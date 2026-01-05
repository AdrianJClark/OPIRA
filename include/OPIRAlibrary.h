#ifndef OPIRALIBRARY_H
#define OPIRALIBRARY_H

#include <vector>
#include "opencv/cv.h"

namespace OPIRALibrary {
	struct Marker {
		CvSize size;
		std::string name;
		IplImage* image;
	};

	struct MarkerTransform {
		Marker marker;
		CvMat *homography;
		int *viewPort;
		double *transMat, *projMat;
		int score;

		MarkerTransform(): viewPort(0), projMat(0), transMat(0), score(0), homography(0) {}
		
		void clear() {
			if (homography!=0) cvReleaseMat(&homography);
			if (viewPort!=0) free(viewPort);
			if (projMat!=0) free(projMat);
			if (transMat!=0) free(transMat);
		}

	};

	class PointMatches {
		public:
			PointMatches();
			~PointMatches();

			void resize(int size);
			void clear();
			void clone(PointMatches src);

			CvPoint2D32f *featMarker, *featImage; CvMat* homography;
			int count; float *score;
	};

	class RegistrationAlgorithm {
	public:
		RegistrationAlgorithm();
		~RegistrationAlgorithm();
		virtual std::string getName();
		virtual bool addMarker(Marker marker);
		virtual bool removeMarker(Marker marker);
		virtual RegistrationAlgorithm *clone();
		virtual std::vector<PointMatches> findAllMatches(IplImage* image);
		virtual PointMatches findMatches(IplImage* image, int index);
	protected:
		struct TrainedMarker {
			cv::flann::Index* tree; 
			cv::Mat *descriptors; 
			CvPoint2D32f* features; 
			int numPoints; 
			std::string markerFilename;
			int descriptorLength;
		};

		struct Feature {
			CvPoint2D32f position; 
			float* descriptor;
			Feature(int descriptorLength) {descriptor = (float*)malloc(descriptorLength*sizeof(float));}
			Feature(int descriptorLength, float* descript) {descriptor = (float*)malloc(descriptorLength*sizeof(float)); memcpy(descriptor, descript, descriptorLength*sizeof(float));}
		};

		class FeatureVector: public std::vector<Feature> {
		public:
			int descriptorLength;
			void clear() { for (unsigned int i=0; i<size(); i++) free(at(i).descriptor); vector::clear();}
		};

		TrainedMarker trainMarker(FeatureVector pa);
		PointMatches matchANN(FeatureVector pa, float thresh, TrainedMarker mt);
		std::vector<PointMatches> matchAllANN(FeatureVector features, float thresh);
		std::vector <TrainedMarker> trainedMarkers;
		TrainedMarker allMarkers;
		std::vector<int> allMarkersPos;
		virtual FeatureVector performRegistration(IplImage* image);
	public:
		std::vector<TrainedMarker> RegistrationAlgorithm::cloneTrainedMarker(std::vector<TrainedMarker> trainedMarkers);
		TrainedMarker RegistrationAlgorithm::cloneTrainedMarker(TrainedMarker trainedMarker);
	private: 
		int RegistrationAlgorithm::findMarkerIndex(int featureIndex);
	};


	class Registration {
	public:
		Registration();
		virtual ~Registration();
		virtual std::vector<MarkerTransform> performRegistration(IplImage* frame_input, CvMat* captureParams, CvMat* captureDistortion);
		Marker getMarker(std::string filename);
		
		//Add Marker Functions
		virtual bool addMarker(std::string markerName);
		virtual bool addResizedMarker(std::string markerName, int maxLengthSize);
		virtual bool addScaledMarker(std::string markerName, int maxLengthScale);
		virtual bool addResizedScaledMarker(std::string markerName, int maxLengthSize, int maxLengthScale);

		virtual bool removeMarker(std::string markerName);
		bool displayImage;
	protected:
		std::vector<Marker> markers;
		
		CvMat* homographyToCvTransMat(CvMat* H, CvMat* camParams);
		CvMat* matchesToCvTransMat(PointMatches m, CvMat* captureParams, CvMat* captureDistortion);
		
		CvMat* getGoodHomography(CvMat *cvTransMat, CvMat* captureParams, CvMat* captureDistortion, CvSize markerSize);
		CvMat* getGoodHomography(PointMatches bestMatch, CvMat* captureParams, CvMat* captureDistortion, CvSize markerSize);
		
		MarkerTransform computeMarkerTransform(PointMatches pMatch, int index, CvSize frameSize, CvMat *captureParams, CvMat *captureDistortion);
		MarkerTransform computeMarkerTransform(CvMat* homography, int matchCount, int index, CvSize frameSize, CvMat *captureParams, CvMat *captureDistortion);

		PointMatches ScaleMatches(float scale, PointMatches p);
		RegistrationAlgorithm *regAlgorithm;
	};

	class RegistrationStandard:public Registration {
	public:
		RegistrationStandard(RegistrationAlgorithm *RegistrationAlgorithm);
		
		~RegistrationStandard();
		std::vector<MarkerTransform> performRegistration(IplImage* frame_input, CvMat* captureParams, CvMat* captureDistortion);
	protected:
		static const int minRegistrationCount=4;
	};

	class RegistrationOpticalFlow:public RegistrationStandard {
	public:
		RegistrationOpticalFlow(RegistrationAlgorithm *registrationAlgorithm);

		//Add Marker Functions
		virtual bool addMarker(std::string markerName);
		virtual bool addResizedMarker(std::string markerName, int maxLengthSize);
		virtual bool addScaledMarker(std::string markerName, int maxLengthScale);
		virtual bool addResizedScaledMarker(std::string markerName, int maxLengthSize, int maxLengthScale);

		virtual bool removeMarker(std::string markerFilename);
		~RegistrationOpticalFlow();
		std::vector<MarkerTransform> performRegistration(IplImage* frame_input, CvMat* captureParams, CvMat* captureDistortion);
	protected:
		PointMatches opticalFlow(PointMatches prevMatches);
		IplImage* previousImage, *currentImage; IplImage* previousPyramid, *currentPyramid;
		std::vector<PointMatches> previousMatches;
		static const int minOptFlowCount=4;
	};

	class RegistrationOPIRA:public RegistrationOpticalFlow {
	public:
		RegistrationOPIRA(RegistrationAlgorithm *registrationAlgorithm);

		~RegistrationOPIRA();
		std::vector<MarkerTransform> performRegistration(IplImage* frame_input, CvMat* captureParams, CvMat* captureDistortion);
	protected:
		PointMatches undistortRegister(IplImage* frame_input, int index, CvMat* homography, CvSize MarkerSize=cvSize(-1,-1));
		int OPIRAmaxDimension;
	};

	PointMatches Ransac(PointMatches corspMap);
	PointMatches Ransac2(PointMatches corspMap);
	
	int* calcViewpoint(CvMat* captureParams, CvMat* captureDistortion, CvSize imgSize);
	double* calcProjection(CvMat* captureParams, CvMat* captureDistortion, CvSize imgSize, double dNear=10.0, double dFar=10000.0);
	double* calcTransform(CvMat *cvTransMat);

	void displayMatches(IplImage *marker, IplImage *scene, PointMatches matches, std::string windowName, std::string windowCaption = "",  float markerScale=1.0, float frameScale=1.0);
};


#endif
