#ifndef OPIRALIBRARYMT_H
#define OPIRALIBRARYMT_H

#include "OPIRALibrary.h"

namespace OPIRALibrary {

	class Mutex {
	public:
		Mutex();
		~Mutex();
		void acquire();
		void release();
	private:
		bool mutex;
	};

	class RegistrationThread {
	public:
		RegistrationThread(RegistrationAlgorithm *registrationAlgorithm);
		~RegistrationThread();
		//void run();
		void stop();
		std::vector<PointMatches> findAllMatches(IplImage *frame_input);
		void deleteReg();
		bool removeMarker(Marker marker);
		bool addMarker(Marker marker);
		bool isFinished();

		struct ThreadData {
			bool running, updated, finished;
			RegistrationAlgorithm *regAlgorithm;
			IplImage* curFrame, *newFrame;
			std::vector<PointMatches> *curMatches; 
			Mutex curMutex, newMutex;
		};

	protected:
		std::vector<PointMatches> allOpticalFlow(IplImage *previousImage, std::vector<PointMatches> prevMatches, IplImage *currentImage);
		ThreadData *tData;
	};

	class RegistrationThreadOPIRA {
	public:
		RegistrationThreadOPIRA(RegistrationAlgorithm *registrationAlgorithm, CvSize markerSize);
		~RegistrationThreadOPIRA();
		//void run();
		void stop();
		PointMatches undistortRegister(IplImage *frame_input, CvMat* homography, CvSize frameSize);
		void deleteReg();
		bool removeMarker(Marker marker);
		bool addMarker(Marker marker);
		bool isFinished();

		struct ThreadOPIRAData {
			bool running, updated, finished;
			RegistrationAlgorithm *regAlgorithm;
			IplImage* curFrame, *newFrame;
			PointMatches curMatches;
			Mutex curMutex, newMutex;

			CvMat* newHomography;
			CvSize mSize;
		};

	protected:
		PointMatches opticalFlow(IplImage *previousImage, PointMatches prevMatches, IplImage *currentImage);
		ThreadOPIRAData *tData;
	};

	class RegistrationStandardMT:public RegistrationStandard {
	public:
		RegistrationStandardMT(RegistrationAlgorithm *registrationAlgorithm);
		~RegistrationStandardMT();
		
		virtual bool addMarker(std::string markerName);
		virtual bool addResizedMarker(std::string markerName, int maxLengthSize);
		virtual bool addScaledMarker(std::string markerName, int maxLengthScale);
		virtual bool addResizedScaledMarker(std::string markerName, int maxLengthSize, int maxLengthScale);

		virtual bool removeMarker(std::string markerName);

		std::vector<MarkerTransform> performRegistration(IplImage* frame_input, CvMat* captureParams, CvMat* captureDistortion);

	protected:
		RegistrationThread *regThread;
	};

	class RegistrationOpticalFlowMT:public RegistrationOpticalFlow {
	public:
		RegistrationOpticalFlowMT(RegistrationAlgorithm *registrationAlgorithm);
		~RegistrationOpticalFlowMT();

		virtual bool addMarker(std::string markerName);
		virtual bool addResizedMarker(std::string markerName, int maxLengthSize);
		virtual bool addScaledMarker(std::string markerName, int maxLengthScale);
		virtual bool addResizedScaledMarker(std::string markerName, int maxLengthSize, int maxLengthScale);
		virtual bool removeMarker(std::string markerName);

		std::vector<MarkerTransform> performRegistration(IplImage* frame_input, CvMat* captureParams, CvMat* captureDistortion);
	protected:
		RegistrationThread *regThread;
	};

	class RegistrationOPIRAMT:public RegistrationOPIRA {
	public:
		RegistrationOPIRAMT(RegistrationAlgorithm *registrationAlgorithm);
		~RegistrationOPIRAMT();
		std::vector<MarkerTransform> performRegistration(IplImage* frame_input, CvMat* captureParams, CvMat* captureDistortion);

		virtual bool addMarker(std::string markerName);
		virtual bool addResizedMarker(std::string markerName, int maxLengthSize);
		virtual bool addScaledMarker(std::string markerName, int maxLengthScale);
		virtual bool addResizedScaledMarker(std::string markerName, int maxLengthSize, int maxLengthScale);

		virtual bool removeMarker(std::string markerName);

	protected:
		RegistrationThread *regThread;
		std::vector<RegistrationThreadOPIRA*> regOPIRAThreads;
	};

};

#endif