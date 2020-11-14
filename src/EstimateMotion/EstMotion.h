
#ifndef ESTMOTION_H
#define ESTMOTION_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "../System/System.h"
#include "ORBextractor.h"
#include "Frame.h"
#include "Initializer.h"

#include <mutex>

namespace SFRSS
{
class System;
// class Initializer;
class EstMotion
{
    typedef pair<int,int> Match;
public:
    EstMotion(System* pSys, const string &strSettingPath, string savePath);

    vector<vector<double>> Est(const cv::Mat &imLeft,const cv::Mat &imRight, const double &timestamp, string imageName, bool JustStatistic);

    // void ChangeCalibration(const string &strSettingPath);

    vector<vector<double>> GetProjRes();
    vector<vector<double>> GetCloseFormRes();

public:

    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    int mSensor;

    Frame mLeftFrame;
    Frame mRightFrame;
    cv::Mat mImGray;

    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    list<cv::Mat> mlRelativeFramePoses;
    list<double> mlFrameTimes;
    list<bool> mlbLost;

    bool mbOnlyTracking;

    void Reset();

protected:

    void Refine(vector<vector<int>> & CandidateGroup, vector<vector<double>> & MotionStates);
    bool mbVO;

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    //Current matches in frame
    int mnMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    Frame mLastFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;

    //Motion Model
    cv::Mat mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;


    //Calibration matrix
    cv::Mat mK0;
    cv::Mat mK1;
    cv::Mat mDistCoef0;
    cv::Mat mDistCoef1;
    float mbf;

    int mRows, mCols;

    cv::Mat P0, P1, R0, R1, R, T;
    cv::Mat M1l,M2l,M1r,M2r;
    cv::Mat unM1l,unM2l,unM1r,unM2r,unM1r_x2,unM2r_x2;

    int TargetWidth, TargetHeight;
    double TargetFx, TargetFy, TargetCx, TargetCy;
    cv::Mat TargetK, TargetK2;

    Initializer * mpInitializer;
    System * mpSystem;
    vector<Match> mvMatches12;

    string mBasePath;
};

} 

#endif // ESTMOTION_H
