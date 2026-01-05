#include <stdlib.h>

#include "GL\glut.h"

#include "OPIRALibrary.h"
#include "OPIRALibraryMT.h"
#include "CaptureLibrary.h"
#include "RegistrationAlgorithms\OCVSurf.h"

#include "OPIRALibraryGPU.h"
#include "RegistrationAlgorithms\OCVGPUSurf.h"

//#include "OPIRALibraryMT.h"

using namespace OPIRALibrary;
using namespace std;

void initGLTextures();
void draw(IplImage* frame_input, std::vector<MarkerTransform> mt, std::string windowName);

#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480

GLuint GLTextureID;

bool running = true;

#include <windows.h>

LARGE_INTEGER freq, start, end; int fps;
int frameCount=0;

void main(int argc, char **argv) {

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(640, 480);
	glutCreateWindow("SimpleTest");
	
	//Create a new Registration Algorithm using OPIRA and the SURF algorithm
	Registration *r = new RegistrationOPIRA(new OCVGPUSurf());
	r->addMarker("celicasmall.bmp");

	//Initialise our Camera
	Capture* camera = new Camera(cvSize(640,480), "camera.yml");
	
	initGLTextures();

	QueryPerformanceFrequency(&freq); QueryPerformanceCounter(&start); fps=0;

	while (running) {
		//Grab a frame 
		IplImage *new_frame = camera->getFrame();

		if (new_frame==0) break;

		//Perform registration
		vector<MarkerTransform> mt = r->performRegistration(new_frame, camera->getParameters(), camera->getDistortion());

		draw(new_frame, mt, "Main");

		frameCount++; QueryPerformanceCounter(&end);
		if (((end.QuadPart-start.QuadPart)*1000.0/freq.QuadPart)>1000) { fps = frameCount; frameCount=0; start = end; }

		//Check for the escape key and give the computer some processing time
		switch (cvWaitKey(1)) {
			case 27:
				running = false; break;
		}
	
		//Clean up
		cvReleaseImage(&new_frame);
		for (unsigned int i=0; i<mt.size(); i++) { mt.at(i).clear(); } mt.clear();
	};

	delete camera;
	delete r;

}


void draw(IplImage* frame_input, std::vector<MarkerTransform> mt, std::string windowName)
{
	//Clear the depth buffer 
	glClearDepth( 1.0 ); glClear(GL_DEPTH_BUFFER_BIT); glDepthFunc(GL_LEQUAL);

	//Set the viewport to the window size
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    //Set the Projection Matrix to an ortho slightly larger than the window
	glMatrixMode(GL_PROJECTION); glLoadIdentity();
	glOrtho(-0.5, WINDOW_WIDTH-0.5, WINDOW_HEIGHT-0.5, -0.5, 1.0, -1.0);
    //Set the modelview to the identity
	glMatrixMode(GL_MODELVIEW); glLoadIdentity();

	//Turn off Light and enable a texture
	glDisable(GL_LIGHTING);	glEnable(GL_TEXTURE_2D); glDisable(GL_DEPTH_TEST);

	glBindTexture(GL_TEXTURE_2D, GLTextureID);
	glTexImage2D(GL_TEXTURE_2D, 0, 3, frame_input->width, frame_input->height, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, frame_input->imageData);
	
	//Draw the background
	glPushMatrix();
        glColor3f(255, 255, 255);
        glBegin(GL_TRIANGLE_STRIP);
            glTexCoord2f(0.0, 0.0);	glVertex2f(0.0, 0.0);
            glTexCoord2f(1.0, 0.0);	glVertex2f(WINDOW_WIDTH, 0.0);
            glTexCoord2f(0.0, 1.0);	glVertex2f(0.0, WINDOW_HEIGHT);
            glTexCoord2f(1.0, 1.0);	glVertex2f(WINDOW_WIDTH, WINDOW_HEIGHT);
        glEnd();
	glPopMatrix();

	//Turn off Texturing
	glBindTexture(GL_TEXTURE_2D, 0);
	glEnable(GL_LIGHTING);	glDisable(GL_TEXTURE_2D); glEnable(GL_DEPTH_TEST);

	//Loop through all the markers found
	for (unsigned int i =0; i<mt.size(); i++) {
		double* projectionMat = mt.at(i).projMat;
		double* translationMat = mt.at(i).transMat;
		CvSize markerSize = mt.at(i).marker.size;

		//Set the Viewport Matrix
		glViewport(0,0,WINDOW_WIDTH,WINDOW_HEIGHT);

		//Load the Projection Matrix
		glMatrixMode(GL_PROJECTION);
		glLoadMatrixd( projectionMat );

		//Load the camera modelview matrix 
		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixd( translationMat );

		//Draw the boundry rectangle
		glColor3f(0,0,0);
		glBegin(GL_LINE_LOOP);
			glVertex3d(0,					0,0);
			glVertex3d(markerSize.width,	0,0);
			glVertex3d(markerSize.width,	markerSize.height,0);
			glVertex3d(0,					markerSize.height,0);
		glEnd();

		//Draw the infamous Teapot
		glEnable(GL_LIGHTING);
			glTranslatef(markerSize.width/2.0, markerSize.height/2.0,-25);
			glRotatef(-90, 1.0,0,0.0);
			if (mt.at(i).marker.name =="CelicaSmall.bmp") glColor3f(1,0,0);
			if (mt.at(i).marker.name =="MagicLandSmall.bmp") glColor3f(0,1,0);
			if (mt.at(i).marker.name =="NZi3MapSmall.jpg") glColor3f(0,0,1);
			glutSolidTeapot(50.0);
		glDisable(GL_LIGHTING);


	}
	
	//Copy the OpenGL Graphics context into an IPLImage
	IplImage* outImage = cvCreateImage(cvSize(WINDOW_WIDTH,WINDOW_HEIGHT), IPL_DEPTH_8U, 3);
	glReadPixels(0,0,WINDOW_WIDTH,WINDOW_HEIGHT,GL_RGB, GL_UNSIGNED_BYTE, outImage->imageData);
	cvCvtColor( outImage, outImage, CV_BGR2RGB );
	cvFlip(outImage, outImage);

	char sfps[50]; sprintf(sfps, "%dfps", fps);
	CvFont f; cvInitFont(&f, CV_FONT_HERSHEY_PLAIN, 1.2, 1.2,0,2);
	cvPutText(outImage, sfps, cvPoint(5, 20), &f, cvScalar(0,255,255,0));

	cvNamedWindow(windowName.c_str()), cvShowImage(windowName.c_str(), outImage);
	cvReleaseImage(&outImage);
}


void initGLTextures() {
	//Set up Materials 
	GLfloat mat_specular[] = { 0.4, 0.4, 0.4, 1.0 };
	GLfloat mat_diffuse[] = { .8,.8,.8, 1.0 };
	GLfloat mat_ambient[] = { .4,.4,.4, 1.0 };

	glShadeModel(GL_SMOOTH);//smooth shading
	glMatrixMode(GL_MODELVIEW);
	glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialf(GL_FRONT, GL_SHININESS, 100.0);//define the material
	glColorMaterial(GL_FRONT, GL_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);//enable the material
	glEnable(GL_NORMALIZE);

	//Set up Lights
	GLfloat light0_ambient[] = {0.1, 0.1, 0.1, 0.0};
	float light0_diffuse[] = { 0.8f, 0.8f, 0.8, 1.0f };
	glLightfv (GL_LIGHT0, GL_AMBIENT, light0_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
    glEnable(GL_LIGHT0);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_NORMALIZE);
	glHint (GL_LINE_SMOOTH_HINT, GL_NICEST);	
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glEnable (GL_LINE_SMOOTH);	

	//Initialise the OpenCV Image for GLRendering
	glGenTextures(1, &GLTextureID); 	// Create a Texture object
    glBindTexture(GL_TEXTURE_2D, GLTextureID);  //Use this texture
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);	
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);	
	glBindTexture(GL_TEXTURE_2D, 0);

}
