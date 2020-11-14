#ifndef SYSTEM_H
#define SYSTEM_H

#include <unistd.h>
#include <string>
#include <thread>
#include <opencv2/core/core.hpp>
#include "../EstimateMotion/EstMotion.h" 
#include "../EstimateDepths/include/stereomatch.h"
#include "../RefineMotion/RefineMotion.h"

using namespace std;

namespace SFRSS
{
class EstMotion;
class StereoMatch;
class RefineMotion;
enum WROK_MODE{
    RAW_MODE,
    MIDDLE_MODE,
    FINAL_MODE, 
    STAND_MODE, // same with global shutter
    CALC_STATE_MODE, // just estimate states
};
class System
{
public:
    // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
    System(const string &strSettingsFile, string savePath, int workMode);
    ~System();
    int Run2(const cv::Mat & leftImage, const cv::Mat & rightImage, double tframe, string imageName);
    int Run3(const cv::Mat & leftImage, const cv::Mat & rightImage, double tframe, string imageName);

    void Reset();
    void Shutdown();

private:
    // Reset flag
    bool mbReset;

    EstMotion * mpEstMotion;
    StereoMatch * mpStereoMatch;
    RefineMotion * mpRefineMotion;

    int mdisparitySize;
    int m_width;
    int m_height;
    int mInvalidDisp;

    void * mptrDisparity0;
    void * mptrDisparity1;
    bool mIsSubPixel;
    int mSubPixelLvl;

    float * mptrDisparityBuffer;
    uint8_t * mptrUndistImg;

    FramePym * ptrLeft;
    FramePym * ptrRight;

    string mSaveDirPath;
    string mBasePath;

    int mworkMode;
    vector<double> lastState;

    double * mptrDepthMap;
    double * mptrDepthMap1;
    double * mptrErrorMap;
};

}

#endif // SYSTEM_H
