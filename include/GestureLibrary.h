#include "OPIRALibrary.h"


namespace OPIRALibrary {

	class GestureHandWave {
	public:
		GestureHandWave();
		~GestureHandWave();

	int getHandWaveAmount(IplImage* currentImage, IplImage* differenceImage, MarkerTransform mt);
	private:
		CvPoint handPosition[10];
	};

	IplImage* getDifferenceImage(IplImage* currentImage, MarkerTransform mt);
}