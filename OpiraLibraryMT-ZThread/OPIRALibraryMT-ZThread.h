#ifndef OPIRALIBRARYMTZTHREAD_H
#define OPIRALIBRARYMTZTHREAD_H

#include "OPIRALibrary.h"
#include "zthread/Thread.h"
#include "zthread/Mutex.h"
#include "zthread/FastMutex.h"

using namespace ZThread;
namespace OPIRALibrary {

	class RegistrationThread : public Runnable {
	public:
		RegistrationThread(RegistrationAlgorithm *registrationAlgorithm);
		~RegistrationThread();
		void run();
		void stop();
		vector<PointMatches> findAllMatches(IplImage *frame_input);
		void deleteReg();
		bool removeMarker(Marker marker);
		bool addMarker(Marker marker);
		bool isFinished();
	protected:
		RegistrationAlgorithm *regAlgorithm;
		bool running, updated, finished;
		PointMatches opticalFlow(IplImage *previousImage, PointMatches prevMatches, IplImage *currentImage);
		vector<PointMatches> allOpticalFlow(IplImage *previousImage, vector<PointMatches> prevMatches, IplImage *currentImage);
		
		FastMutex curMutex; FastMutex newMutex;

		//The last known good registration
		IplImage* curFrame; vector<PointMatches> curMatches; 
		//Latest frame for processing
		IplImage *newFrame;
	};

	class RegistrationThreadOPIRA : public RegistrationThread {
	public:
		RegistrationThreadOPIRA(RegistrationAlgorithm *registrationAlgorithm, int markerIndex, CvSize markerSize);
		~RegistrationThreadOPIRA();
		void run();
		PointMatches undistortRegister(IplImage *frame_input, CvMat* homography, CvSize frameSize);
	protected:
		int mIndex; CvSize mSize;
		PointMatches curMatches;
		CvMat* newHomography;
	};

	class RegistrationStandardMT:public RegistrationStandard {
	public:
		RegistrationStandardMT(RegistrationAlgorithm *registrationAlgorithm);
		~RegistrationStandardMT();
		
		virtual bool addMarker(string markerName);
		virtual bool addResizedMarker(string markerName, int maxLengthSize);
		virtual bool addScaledMarker(string markerName, int maxLengthScale);
		virtual bool addResizedScaledMarker(string markerName, int maxLengthSize, int maxLengthScale);

		virtual bool removeMarker(string markerName);

		vector<MarkerTransform> performRegistration(IplImage* frame_input, CvMat* captureParams, CvMat* captureDistortion);

	protected:
		RegistrationThread *regThread; Thread *thread;
	};

	class RegistrationOpticalFlowMT:public RegistrationOpticalFlow {
	public:
		RegistrationOpticalFlowMT(RegistrationAlgorithm *registrationAlgorithm);
		~RegistrationOpticalFlowMT();

		virtual bool addMarker(string markerName);
		virtual bool addResizedMarker(string markerName, int maxLengthSize);
		virtual bool addScaledMarker(string markerName, int maxLengthScale);
		virtual bool addResizedScaledMarker(string markerName, int maxLengthSize, int maxLengthScale);
		virtual bool removeMarker(string markerName);

		vector<MarkerTransform> performRegistration(IplImage* frame_input, CvMat* captureParams, CvMat* captureDistortion);
	protected:
		RegistrationThread *regThread; Thread *thread;
	};

	class RegistrationOPIRAMT:public RegistrationOPIRA {
	public:
		RegistrationOPIRAMT(RegistrationAlgorithm *registrationAlgorithm);
		~RegistrationOPIRAMT();
		vector<MarkerTransform> performRegistration(IplImage* frame_input, CvMat* captureParams, CvMat* captureDistortion);

		virtual bool addMarker(string markerName);
		virtual bool addResizedMarker(string markerName, int maxLengthSize);
		virtual bool addScaledMarker(string markerName, int maxLengthScale);
		virtual bool addResizedScaledMarker(string markerName, int maxLengthSize, int maxLengthScale);

		virtual bool removeMarker(string markerName);

	protected:
		RegistrationThread *regThread; Thread *thread;
		std::vector<RegistrationThreadOPIRA*> regOPIRAThreads; std::vector<Thread*> opiraThreads;
	};

};

#endif