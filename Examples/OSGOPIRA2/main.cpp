#include <windows.h>

//OPIRA
#include <CaptureLibrary.h>
#include <OPIRALibrary.h>
#include <OPIRALibraryGPU.h>
#include <RegistrationAlgorithms/OCVGpuSurf.h>

//OpenSceneGraph
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osg/Texture2D>
#include <osg/Geode>
#include <osg/Geometry>
#include <osgDB/ReadFile>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/PositionAttitudeTransform>
#include <osg/io_utils>
#include <GL/glut.h>

class OSGModel: public osg::Group {
public:
	osg::ref_ptr<osg::Node> model;
	osg::ref_ptr<osg::MatrixTransform> transform;
	osg::ref_ptr<osg::MatrixTransform> osgTransform;
	osg::ref_ptr<osg::Switch> visible;
	
	OSGModel(std::string filename) {
		model = osgDB::readNodeFile(filename);
		transform = new osg::MatrixTransform(osg::Matrix::identity());
		osgTransform = new osg::MatrixTransform(osg::Matrix::rotate(osg::DegreesToRadians(180.0f), osg::X_AXIS));
		visible = new osg::Switch();
		visible->addChild(transform); transform->addChild(osgTransform); osgTransform->addChild(model);
		this->addChild(visible);
	};
};

osgViewer::Viewer viewer;
osg::ref_ptr<osg::Image> mVideoImage;
osg::ref_ptr<osg::Camera> fgCamera;
osg::ref_ptr<OSGModel> car;

void initOGL(int argc, char **argv);
void render(IplImage* frame_input, std::vector<MarkerTransform> mt);

int _width = 640, _height = 480;

void main(int argc, char **argv) {   

	initOGL(argc, argv);

	Capture *camera = new Camera(0, cvSize(320,240), "camera.yml");
	fgCamera->setProjectionMatrix(osg::Matrix(calcProjection(camera->getParameters(), camera->getDistortion(), cvSize(camera->getWidth(), camera->getHeight()))));

	Registration *registration = new RegistrationOPIRA(new OCVGPUSurf());
	registration->addMarker("CelicaSmall.bmp");

	car = new OSGModel("car.ive");
	fgCamera->addChild(car);

	bool running = true;
	while (running) {
		IplImage *frame = camera->getFrame();
		std::vector<MarkerTransform> mt = registration->performRegistration(frame, camera->getParameters(), camera->getDistortion());
		render(frame, mt);

		switch(cvWaitKey(1)) {
			case 27:
				running = false; break;
		}

		cvReleaseImage(&frame);
		for (int i=0; i<mt.size(); i++) mt.at(i).clear(); mt.clear();
	}

	delete registration;
	delete camera;
}

void initOGL(int argc, char **argv) {

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
	glutInitWindowSize(_width, _height);
	glutCreateWindow("SimpleTest");

    viewer.setUpViewerAsEmbeddedInWindow(0,0,_width,_height);
	viewer.setThreadingModel(osgViewer::Viewer::SingleThreaded);
	viewer.setKeyEventSetsDone(0);

	osg::ref_ptr<osg::Group> root = new osg::Group();
	viewer.setSceneData(root.get());
    viewer.realize();

	mVideoImage = new osg::Image();

	// ----------------------------------------------------------------
	// Video background
	// ----------------------------------------------------------------
	osg::ref_ptr<osg::Camera> bgCamera = new osg::Camera();
	bgCamera->getOrCreateStateSet()->setRenderBinDetails(0, "RenderBin");
	bgCamera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
	bgCamera->setClearMask(GL_DEPTH_BUFFER_BIT);
	bgCamera->getOrCreateStateSet()->setMode(GL_LIGHTING, GL_FALSE);
	bgCamera->getOrCreateStateSet()->setMode(GL_DEPTH_TEST, GL_FALSE);
	bgCamera->setProjectionMatrixAsOrtho2D(0, _width, 0, _height);

	osg::ref_ptr<osg::Geometry> geomBG = osg::createTexturedQuadGeometry(osg::Vec3(0, 0, 0), osg::X_AXIS * _width, osg::Y_AXIS * _height, 0, 1, 1, 0);
	geomBG->getOrCreateStateSet()->setTextureAttributeAndModes(0, new osg::Texture2D(mVideoImage));

	osg::ref_ptr<osg::Geode> geodeBG = new osg::Geode();
	geodeBG->addDrawable(geomBG.get());
	bgCamera->addChild(geodeBG.get());
	root->addChild(bgCamera.get());

	// ----------------------------------------------------------------
	// Foreground 3D content
	// ----------------------------------------------------------------
	fgCamera = new osg::Camera();
	fgCamera->getOrCreateStateSet()->setRenderBinDetails(1, "RenderBin");
	fgCamera->setReferenceFrame(osg::Transform::ABSOLUTE_RF_INHERIT_VIEWPOINT);
	fgCamera->setClearMask(GL_DEPTH_BUFFER_BIT);
	
	root->addChild(fgCamera.get());
}


void render(IplImage* frame_input, std::vector<MarkerTransform> mt) { 
	//Copy the frame into the background image
	IplImage *scaleImage = cvCreateImage(cvSize(512,512), IPL_DEPTH_8U, 3);
	cvResize(frame_input, scaleImage); cvCvtColor(scaleImage, scaleImage, CV_RGB2BGR);
 	mVideoImage->setImage(scaleImage->width, scaleImage->height, 0, 3, GL_RGB, GL_UNSIGNED_BYTE, (unsigned char*)scaleImage->imageData, osg::Image::NO_DELETE);
	
	car->visible->setAllChildrenOff();
	for (int i=0; i<mt.size(); i++) {
		if (mt.at(i).marker.name == "CelicaSmall.bmp") {
			car->visible->setAllChildrenOn();
			car->transform->setMatrix(osg::Matrix(mt.at(i).transMat));
		}
	}

	viewer.frame();

	IplImage* outImage = cvCreateImage(cvSize(_width,_height), IPL_DEPTH_8U, 3);
	glReadPixels(0,0,_width,_height,GL_RGB, GL_UNSIGNED_BYTE, outImage->imageData);
	cvCvtColor( outImage, outImage, CV_BGR2RGB );
	cvFlip(outImage, outImage);

	cvShowImage("Rendered Image", outImage);
	cvReleaseImage(&outImage);

	cvReleaseImage(&scaleImage);
}