
#ifndef INITIALIZER_H
#define INITIALIZER_H

#include<opencv2/opencv.hpp>
#include "Frame.h"


namespace SFRSS
{

// THIS IS THE INITIALIZER FOR MONOCULAR SLAM. NOT USED IN THE STEREO OR RGBD CASE.
class Initializer
{
    typedef pair<int,int> Match;

public:

    // Fix the reference frame
    Initializer(Frame &ReferenceFrame, float sigma = 1.0, int iterations = 200, std::string path = std::string());

    // Computes in parallel a fundamental matrix and a homography
    // Selects a model and tries to recover the motion and the structure from motion
    bool Initialize(Frame &CurrentFrame, const vector<int> &vMatches12,
                    cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, cv::Mat &R, cv::Mat &T, cv::Mat imLeftUn, cv::Mat imRightUn);

    void CheckEssential(std::string resPath, std::string matchPath, cv::Mat R, cv::Mat T, cv::Mat K0, cv::Mat K1);
    int  FindMotionMThread(const vector<vector<size_t>> & Sets);
    int TestR6M(std::string path, std::string matchPath, cv::Mat R, cv::Mat T, cv::Mat K0, cv::Mat K1);
    int Refine(vector<vector<int>> & CandidateGroup, vector<vector<double>> & MotionStates, int type);
    int Refine2(vector<vector<int>> & CandidateGroup, vector<vector<double>> & MotionStates, int type);
    int Refine3(vector<vector<int>> & CandidateGroup, vector<vector<double>> & MotionStates, int type);
    int SetUndistImagePair(cv::Mat & leftImg, cv::Mat & rightImg);

    vector<vector<double>> GetFinalRes(){
        return mRefinedStates;
    }

    void UpdateSolutionsFromCloseForm(vector<vector<int>> finalInliers, vector<vector<double>> finalStateGroup);
    void UpdateSolutionsFromProjError(vector<vector<int>> finalInliers, vector<vector<double>> finalStateGroup);

    int Refine_Proj(vector<vector<int>> &CandidateGroup, vector<vector<double>> &MotionStates, int type);

    vector<vector<double>> GetProjRes();
    vector<vector<double>> GetCloseFormRes();

private:

    void FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21);
    void FindFundamental(vector<bool> &vbInliers, float &score, cv::Mat &F21);
    cv::Mat ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);
    cv::Mat ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);
    float CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma);
    float CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma);
    bool ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);
    bool ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);
    void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);
    void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);
    int CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax);

    void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);

    vector<cv::KeyPoint> mvKeys1Norm;
    vector<cv::KeyPoint> mvKeys2Norm;

    vector<cv::KeyPoint> mvKeys1Un;
    vector<cv::KeyPoint> mvKeys2Un;

    vector<cv::KeyPoint> mvKeys1;
    vector<cv::KeyPoint> mvKeys2;

    // Current Matches from Reference to Current
    vector<Match> mvMatches12;
    vector<bool> mvbMatched1;

    // Calibration
    cv::Mat mK;
    cv::Mat mK2;
    cv::Mat targetK;

    // Standard Deviation and Variance
    float mSigma, mSigma2;

    // Ransac max iterations
    int mMaxIterations;

    // Ransac sets
    vector<vector<size_t> > mvSets;

    cv::Mat mR;
    cv::Mat mT;

    double mLastV0;
    std::string mPath; // Save Debug info 

    vector<double> mLeftIndexs;
    vector<double> mRightIndexs;
    vector<cv::Point2f> mLeftPts;
    vector<cv::Point2f> mRightPts;

    vector<vector<double>> mRefinedStates;
    vector<vector<double>> mCovarianceVec;

    cv::Mat mLeftImgRgb;
    cv::Mat mRightImgRgb;

    vector<int> mCloseFormInliers;
    vector<vector<double>> mCloseFormStates;

    vector<int> mProjInliers;
    vector<vector<double>> mProjStates;
};

} 

#endif // INITIALIZER_H
