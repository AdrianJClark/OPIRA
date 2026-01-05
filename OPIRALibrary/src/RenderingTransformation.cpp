#include "OPIRALibrary.h"

/* Calculate the OpenGL Viewpoint*/
int* OPIRALibrary::calcViewpoint(CvMat* captureParams, CvMat* captureDistortion, CvSize imgSize){
	int *viewport=(int*)malloc(sizeof(int)*4);

	//The principal point is stored in indexes 2 and 5, we also scale it here
	//by Windows Size/Camera Size
	double principal_x_window = captureParams->data.db[2];
	double principal_y_window = captureParams->data.db[5];
	//The dimensions of the window
	double x_window_d = principal_x_window - (imgSize.width/2.0);
	double y_window_d = -(principal_y_window - (imgSize.height)/2.0);//-(principal_y_window - (imgHeight/2.0));
	int transX = (x_window_d < 0.0) ? ((int)(x_window_d - 0.5)) :((int)(x_window_d + 0.5));
	int transY = (y_window_d < 0.0) ? ((int)(y_window_d - 0.5)) :((int)(y_window_d + 0.5));

	//Setup the viewpoint
	viewport[0] = transX;
	viewport[1] = transY;
	viewport[2] = imgSize.width;
	viewport[3] = imgSize.height;

	return viewport;
};

/* Calculate the OpenGL Projection Matrix */
double* OPIRALibrary::calcProjection(CvMat* captureParams, CvMat* captureDistortion, CvSize imgSize, double dNear, double dFar) {

	double *projMat=(double*)calloc(16, sizeof(double));

	//Get camera parameteres
	double fovy = 2 * atan(imgSize.height/(2*captureParams->data.db[4])) * 57.295779513082320876798154814105;
	double aspect = ((double)imgSize.width/(double)imgSize.height) * ((double)captureParams->data.db[0] / (double)captureParams->data.db[4]) ;

	//Set Frustum Co-ordinates
	double top = dNear*tan(fovy * 0.0087266462599716478846184538424431);
	double bottom = -top;
	double left = bottom*aspect;
	double right = top*aspect;

	//Fill in Matrix
	projMat[0] = (2*dNear)/(right-left);
	projMat[2] = (right+left)/(right-left);
	projMat[5] = (2*dNear)/(top-bottom);
	projMat[9] = (top+bottom)/(top-bottom);
	projMat[10] = -(dFar+dNear)/(dFar-dNear);
	projMat[11] = -1;
	projMat[14] = -(2*dFar*dNear)/(dFar-dNear);

	return projMat;
}; 

/* Calculate the OpenGL transformation matrix from an OpenCV transformation Matrix */
double* OPIRALibrary::calcTransform(CvMat *cvTransMat) {
	//Initialize some space for the transformation matrix
	double *transMat = (double*)malloc(16*sizeof(double));

	//Initialize the Reflection matrix so that models are oriented correctly in OpenGL
	CvMat *cvReflect = cvCreateMat(4,4,CV_32FC1); cvZero(cvReflect);
	cvReflect->data.fl[0] = cvReflect->data.fl[15] = 1; cvReflect->data.fl[5] = cvReflect->data.fl[10] = -1;

	//Transpose the homography so it's in the correct order, and reflect it
	CvMat *cvOutput = cvCreateMat(4,4,CV_32FC1);
	cvTranspose(cvTransMat, cvOutput);
	cvMatMul(cvOutput, cvReflect, cvOutput);

	//Copy the data into the transformation matrix
	for (int i=0; i<16; i++) transMat[i] = cvOutput->data.fl[i];

	//Clean up
	cvReleaseMat(&cvReflect);
	cvReleaseMat(&cvOutput);

	return transMat;
}