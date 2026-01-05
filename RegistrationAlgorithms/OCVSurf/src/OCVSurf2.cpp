#include "RegistrationAlgorithms\OCVSurf2.h"
#include "opencv/highgui.h"

using namespace OPIRALibrary;
using namespace cv;

static CvSeq* icvFastHessianDetector( const CvMat* sum, const CvMat* mask_sum, CvMemStorage* storage, const CvSURFParams* params );

OCVSurfOrig::OCVSurfOrig(bool descriptor128bit, int nOctaves, int nSamplingSteps, double blobThresh) {
	surf = SURF(blobThresh, nOctaves, nSamplingSteps, descriptor128bit?true:false);
/*	extended = descriptor128bit?1:0;
    hessianThreshold = blobThres;
    octaves = nOctaves;
    octaveLayers = nSamplingSteps;
	*/
}

RegistrationAlgorithm* OCVSurfOrig::clone() {
	OCVSurfOrig* r = new OCVSurfOrig(*this);
	r->trainedMarkers = RegistrationAlgorithm::cloneTrainedMarker(trainedMarkers);
	r->allMarkers = RegistrationAlgorithm::cloneTrainedMarker(allMarkers);
	return r;
}

string OCVSurfOrig::getName() {
	return "OpenCV SURF";
}

RegistrationAlgorithm::FeatureVector OCVSurfOrig::performRegistration(IplImage* image) {
	FeatureVector features; 
	std::vector<KeyPoint> keyPoints;
	std::vector<float> descriptors;

	if (image->nChannels > 1) {
		IplImage *imageBW = cvCreateImage(cvGetSize(image), image->depth, 1);
		cvConvertImage(image, imageBW);
	
		surf(imageBW, cv::Mat(), keyPoints, descriptors);

		cvReleaseImage(&imageBW);

	} else {
		surf(image, cv::Mat(), keyPoints, descriptors);
	}

	features.descriptorLength = surf.extended?128:64;

	for (unsigned int i=0; i<keyPoints.size(); i++) {
		Feature f(features.descriptorLength, &descriptors[i*features.descriptorLength]); f.position= cvPoint2D32f(keyPoints.at(i).pt.x, keyPoints.at(i).pt.y);
		features.push_back(f);
	}

	descriptors.clear();
	keyPoints.clear();

	return features;
}

OCVSurfOrig::~OCVSurfOrig() {

}


struct CvSurfHF
{
    int p0, p1, p2, p3;
    float w;
};

/* Wavelet size at first layer of first octave. */ 
const int HAAR_SIZE0 = 9;    

/* Wavelet size increment between layers. This should be an even number, 
 such that the wavelet sizes in an octave are either all even or all odd.
 This ensures that when looking for the neighbours of a sample, the layers
 above and below are aligned correctly. */
const int HAAR_SIZE_INC = 6;

static void
icvResizeHaarPattern( const int src[][5], CvSurfHF* dst, int n, int oldSize, int newSize, int widthStep )
{
    float ratio = (float)newSize/oldSize;
    for( int k = 0; k < n; k++ )
    {
        int dx1 = cvRound( ratio*src[k][0] );
        int dy1 = cvRound( ratio*src[k][1] );
        int dx2 = cvRound( ratio*src[k][2] );
        int dy2 = cvRound( ratio*src[k][3] );
        dst[k].p0 = dy1*widthStep + dx1;
        dst[k].p1 = dy2*widthStep + dx1;
        dst[k].p2 = dy1*widthStep + dx2;
        dst[k].p3 = dy2*widthStep + dx2;
        dst[k].w = src[k][4]/((float)(dx2-dx1)*(dy2-dy1));
    }
}

CV_INLINE float
icvCalcHaarPattern( const int* origin, const CvSurfHF* f, int n )
{
    double d = 0;
    for( int k = 0; k < n; k++ )
        d += (origin[f[k].p0] + origin[f[k].p3] - origin[f[k].p1] - origin[f[k].p2])*f[k].w;
    return (float)d;
}

CV_INLINE int 
icvInterpolateKeypoint( float N9[3][9], int dx, int dy, int ds, CvSURFPoint *point )
{
    int solve_ok;
    float A[9], x[3], b[3];
    CvMat _A = cvMat(3, 3, CV_32F, A);
    CvMat _x = cvMat(3, 1, CV_32F, x);                
    CvMat _b = cvMat(3, 1, CV_32F, b);

    b[0] = -(N9[1][5]-N9[1][3])/2;  /* Negative 1st deriv with respect to x */
    b[1] = -(N9[1][7]-N9[1][1])/2;  /* Negative 1st deriv with respect to y */
    b[2] = -(N9[2][4]-N9[0][4])/2;  /* Negative 1st deriv with respect to s */

    A[0] = N9[1][3]-2*N9[1][4]+N9[1][5];            /* 2nd deriv x, x */
    A[1] = (N9[1][8]-N9[1][6]-N9[1][2]+N9[1][0])/4; /* 2nd deriv x, y */
    A[2] = (N9[2][5]-N9[2][3]-N9[0][5]+N9[0][3])/4; /* 2nd deriv x, s */
    A[3] = A[1];                                    /* 2nd deriv y, x */
    A[4] = N9[1][1]-2*N9[1][4]+N9[1][7];            /* 2nd deriv y, y */
    A[5] = (N9[2][7]-N9[2][1]-N9[0][7]+N9[0][1])/4; /* 2nd deriv y, s */
    A[6] = A[2];                                    /* 2nd deriv s, x */
    A[7] = A[5];                                    /* 2nd deriv s, y */
    A[8] = N9[0][4]-2*N9[1][4]+N9[2][4];            /* 2nd deriv s, s */

    solve_ok = cvSolve( &_A, &_b, &_x );
    if( solve_ok )
    {
        point->pt.x += x[0]*dx;
        point->pt.y += x[1]*dy;
        point->size = cvRound( point->size + x[2]*ds ); 
    }
    return solve_ok;
}

static CvSeq* icvFastHessianDetector( const CvMat* sum, const CvMat* mask_sum,
    CvMemStorage* storage, const CvSURFParams* params )
{
    CvSeq* points = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvSURFPoint), storage );

    /* Sampling step along image x and y axes at first octave. This is doubled
       for each additional octave. WARNING: Increasing this improves speed, 
       however keypoint extraction becomes unreliable. */
    const int SAMPLE_STEP0 = 1; 


    /* Wavelet Data */
    const int NX=3, NY=3, NXY=4, NM=1;
    const int dx_s[NX][5] = { {0, 2, 3, 7, 1}, {3, 2, 6, 7, -2}, {6, 2, 9, 7, 1} };
    const int dy_s[NY][5] = { {2, 0, 7, 3, 1}, {2, 3, 7, 6, -2}, {2, 6, 7, 9, 1} };
    const int dxy_s[NXY][5] = { {1, 1, 4, 4, 1}, {5, 1, 8, 4, -1}, {1, 5, 4, 8, -1}, {5, 5, 8, 8, 1} };
    const int dm[NM][5] = { {0, 0, 9, 9, 1} };
    CvSurfHF Dx[NX], Dy[NY], Dxy[NXY], Dm;

    CvMat** dets = (CvMat**)cvStackAlloc((params->nOctaveLayers+2)*sizeof(dets[0]));
    CvMat** traces = (CvMat**)cvStackAlloc((params->nOctaveLayers+2)*sizeof(traces[0]));
    int *sizes = (int*)cvStackAlloc((params->nOctaveLayers+2)*sizeof(sizes[0]));

    double dx = 0, dy = 0, dxy = 0;
    int octave, layer, sampleStep, size, margin;
    int rows, cols;
    int i, j, sum_i, sum_j;
    const int* s_ptr;
    float *det_ptr, *trace_ptr;

    /* Allocate enough space for hessian determinant and trace matrices at the 
       first octave. Clearing these initially or between octaves is not
       required, since all values that are accessed are first calculated */
    for( layer = 0; layer <= params->nOctaveLayers+1; layer++ )
    {
        dets[layer]   = cvCreateMat( (sum->rows-1)/SAMPLE_STEP0, (sum->cols-1)/SAMPLE_STEP0, CV_32FC1 );
        traces[layer] = cvCreateMat( (sum->rows-1)/SAMPLE_STEP0, (sum->cols-1)/SAMPLE_STEP0, CV_32FC1 );
    }

    for( octave = 0, sampleStep=SAMPLE_STEP0; octave < params->nOctaves; octave++, sampleStep*=2 )
    {
        /* Hessian determinant and trace sample array size in this octave */
        rows = (sum->rows-1)/sampleStep;
        cols = (sum->cols-1)/sampleStep;

        /* Calculate the determinant and trace of the hessian */
        for( layer = 0; layer <= params->nOctaveLayers+1; layer++ )
        {
            sizes[layer] = size = (HAAR_SIZE0+HAAR_SIZE_INC*layer)<<octave;
            icvResizeHaarPattern( dx_s, Dx, NX, 9, size, sum->cols );
            icvResizeHaarPattern( dy_s, Dy, NY, 9, size, sum->cols );
            icvResizeHaarPattern( dxy_s, Dxy, NXY, 9, size, sum->cols );
            /*printf( "octave=%d layer=%d size=%d rows=%d cols=%d\n", octave, layer, size, rows, cols );*/
            
            margin = (size/2)/sampleStep;
            for( sum_i=0, i=margin; sum_i<=(sum->rows-1)-size; sum_i+=sampleStep, i++ )
            {
                s_ptr = sum->data.i + sum_i*sum->cols;
                det_ptr = dets[layer]->data.fl + i*dets[layer]->cols + margin;
                trace_ptr = traces[layer]->data.fl + i*traces[layer]->cols + margin;
                for( sum_j=0, j=margin; sum_j<=(sum->cols-1)-size; sum_j+=sampleStep, j++ )
                {
                    dx  = icvCalcHaarPattern( s_ptr, Dx, 3 );
                    dy  = icvCalcHaarPattern( s_ptr, Dy, 3 );
                    dxy = icvCalcHaarPattern( s_ptr, Dxy, 4 );
                    s_ptr+=sampleStep;
                    *det_ptr++ = (float)(dx*dy - 0.81*dxy*dxy);
                    *trace_ptr++ = (float)(dx + dy);
                }
            }
        }

        /* Find maxima in the determinant of the hessian */
        for( layer = 1; layer <= params->nOctaveLayers; layer++ )
        {
            size = sizes[layer];
            icvResizeHaarPattern( dm, &Dm, NM, 9, size, mask_sum ? mask_sum->cols : sum->cols );
            
            /* Ignore pixels without a 3x3 neighbourhood in the layer above */
            margin = (sizes[layer+1]/2)/sampleStep+1; 
            for( i = margin; i < rows-margin; i++ )
            {
                det_ptr = dets[layer]->data.fl + i*dets[layer]->cols;
                trace_ptr = traces[layer]->data.fl + i*traces[layer]->cols;
                for( j = margin; j < cols-margin; j++ )
                {
                    float val0 = det_ptr[j];
                    if( val0 > params->hessianThreshold )
                    {
                        /* Coordinates for the start of the wavelet in the sum image. There   
                           is some integer division involved, so don't try to simplify this
                           (cancel out sampleStep) without checking the result is the same */
                        int sum_i = sampleStep*(i-(size/2)/sampleStep);
                        int sum_j = sampleStep*(j-(size/2)/sampleStep);

                        /* The 3x3x3 neighbouring samples around the maxima. 
                           The maxima is included at N9[1][4] */
                        int c = dets[layer]->cols;
                        const float *det1 = dets[layer-1]->data.fl + i*c + j;
                        const float *det2 = dets[layer]->data.fl   + i*c + j;
                        const float *det3 = dets[layer+1]->data.fl + i*c + j;
                        float N9[3][9] = { { det1[-c-1], det1[-c], det1[-c+1],          
                                             det1[-1]  , det1[0] , det1[1],
                                             det1[c-1] , det1[c] , det1[c+1]  },
                                           { det2[-c-1], det2[-c], det2[-c+1],       
                                             det2[-1]  , det2[0] , det2[1],
                                             det2[c-1] , det2[c] , det2[c+1 ] },
                                           { det3[-c-1], det3[-c], det3[-c+1],       
                                             det3[-1  ], det3[0] , det3[1],
                                             det3[c-1] , det3[c] , det3[c+1 ] } };

                        /* Check the mask - why not just check the mask at the center of the wavelet? */
                        if( mask_sum )
                        {
                            const int* mask_ptr = mask_sum->data.i +  mask_sum->cols*sum_i + sum_j;
                            float mval = icvCalcHaarPattern( mask_ptr, &Dm, 1 );
                            if( mval < 0.5 )
                                continue;
                        }

                        /* Non-maxima suppression. val0 is at N9[1][4]*/
                        if( val0 > N9[0][0] && val0 > N9[0][1] && val0 > N9[0][2] &&
                            val0 > N9[0][3] && val0 > N9[0][4] && val0 > N9[0][5] &&
                            val0 > N9[0][6] && val0 > N9[0][7] && val0 > N9[0][8] &&
                            val0 > N9[1][0] && val0 > N9[1][1] && val0 > N9[1][2] &&
                            val0 > N9[1][3]                    && val0 > N9[1][5] &&
                            val0 > N9[1][6] && val0 > N9[1][7] && val0 > N9[1][8] &&
                            val0 > N9[2][0] && val0 > N9[2][1] && val0 > N9[2][2] &&
                            val0 > N9[2][3] && val0 > N9[2][4] && val0 > N9[2][5] &&
                            val0 > N9[2][6] && val0 > N9[2][7] && val0 > N9[2][8] )
                        {
                            /* Calculate the wavelet center coordinates for the maxima */
                            double center_i = sum_i + (double)(size-1)/2;
                            double center_j = sum_j + (double)(size-1)/2;

                            CvSURFPoint point = cvSURFPoint( cvPoint2D32f(center_j,center_i), 
                                                             CV_SIGN(trace_ptr[j]), sizes[layer], 0, val0 );
                           
                            /* Interpolate maxima location within the 3x3x3 neighbourhood  */
                            int ds = sizes[layer]-sizes[layer-1];
                            int interp_ok = icvInterpolateKeypoint( N9, sampleStep, sampleStep, ds, &point );

                            /* Sometimes the interpolation step gives a negative size etc. */
                            if( interp_ok && point.size >= 1 &&
                                point.pt.x >= 0 && point.pt.x <= (sum->cols-1) &&
                                point.pt.y >= 0 && point.pt.y <= (sum->rows-1) )
                            {    
                                /*printf( "KeyPoint %f %f %d\n", point.pt.x, point.pt.y, point.size );*/
                                cvSeqPush( points, &point );
                            }    
                        }
                    }
                }
            }
        }
    }

    /* Clean-up */
    for( layer = 0; layer <= params->nOctaveLayers+1; layer++ )
    {
        cvReleaseMat( &dets[layer] );
        cvReleaseMat( &traces[layer] );
    }

    return points;
}

