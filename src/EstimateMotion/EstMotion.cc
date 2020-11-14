
#include "EstMotion.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "ORBmatcher.h"
#include "MinimalSolver.h"
#include <opencv2/core/matx.hpp>

#include<iostream>
#include<mutex>

using namespace std;

namespace SFRSS
{

EstMotion::EstMotion(System *pSys, const string &strSettingPath, string savePath): mpSystem(pSys) {
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    cout << "setting file:" << strSettingPath << endl;
    // cout << (int)fSettings["Camera.width"] << endl;

    cv::Mat K0, K1;
    fSettings["M1"] >> K0;
    fSettings["M2"] >> K1;

    K0.copyTo(mK0);
    K1.copyTo(mK1);

    cv::Mat DistCoef0, DistCoef1;
    fSettings["D1"] >> DistCoef0;
    fSettings["D2"] >> DistCoef1;
    DistCoef0 = DistCoef0.t();  // (1,N) -> (N,1)
    DistCoef1 = DistCoef1.t();
    if (DistCoef0.at<double>(0,4) == 0.0){
        DistCoef0.resize(4);
    }
    else{
        DistCoef0.resize(5);
    }

    if (DistCoef1.at<double>(0,4) == 0.0){
        DistCoef1.resize(4);
    }
    else{
        DistCoef1.resize(5);
    }

    DistCoef0.copyTo(mDistCoef0);
    DistCoef1.copyTo(mDistCoef1);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if (fps == 0)
        fps = 4;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx0: " << K0.at<double>(0,0) << endl;
    cout << "- fy0: " << K0.at<double>(1,1) << endl;
    cout << "- cx0: " << K0.at<double>(0,2) << endl;
    cout << "- cy0: " << K0.at<double>(1,2) << endl;
    cout << "- k1: " << DistCoef0.at<float>(0) << endl;
    cout << "- k2: " << DistCoef0.at<float>(1) << endl;
    if(DistCoef0.rows==5)
        cout << "- k3: " << DistCoef0.at<float>(4) << endl;
    cout << "- p1: " << DistCoef0.at<float>(2) << endl;
    cout << "- p2: " << DistCoef0.at<float>(3) << endl;
    cout << endl;

    cout << "- fx1: " << K1.at<double>(0,0) << endl;
    cout << "- fy1: " << K1.at<double>(1,1) << endl;
    cout << "- cx1: " << K1.at<double>(0,2) << endl;
    cout << "- cy1: " << K1.at<double>(1,2) << endl;
    cout << "- k1: " << DistCoef1.at<float>(0) << endl;
    cout << "- k2: " << DistCoef1.at<float>(1) << endl;
    if(DistCoef1.rows==5)
        cout << "- k3: " << DistCoef1.at<float>(4) << endl;
    cout << "- p1: " << DistCoef1.at<float>(2) << endl;
    cout << "- p2: " << DistCoef1.at<float>(3) << endl;
    cout << endl;

    fSettings["P1"] >> P0;
    fSettings["P2"] >> P1;

    fSettings["R1"] >> R0;
    fSettings["R2"] >> R1;

    fSettings["R"] >> R;
    fSettings["T"] >> T;

    mCols = (int)fSettings["Camera.width"];
    mRows = (int)fSettings["Camera.height"];

    if(K0.empty() || K1.empty() || P0.empty() || P1.empty() || R0.empty() || R1.empty() || DistCoef0.empty() || DistCoef1.empty())
    {
        cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
        return;
    }

    TargetWidth = (int)fSettings["Target.width"];
    TargetHeight = (int)fSettings["Target.height"];
    TargetFx = (double)fSettings["Target.fx"];
    TargetFy = (double)fSettings["Target.fy"];
    TargetCx = (double)fSettings["Target.cx"];
    TargetCy = (double)fSettings["Target.cy"];

    cout << "- TargetWidth" << ": " << TargetWidth << endl;
    cout << "- TargetHeight" << ": " << TargetHeight << endl;
    cout << "- TargetFx" << ": " << TargetFx << endl;
    cout << "- TargetFy" << ": " << TargetFy << endl;
    cout << "- TargetCx" << ": " << TargetCx << endl;
    cout << "- TargetCy" << ": " << TargetCy << endl;

    TargetK = cv::Mat(3, 3, CV_64F, 0.0);
    TargetK.at<double>(0,0) = TargetFx*TargetWidth;
    TargetK.at<double>(0,2) = TargetCx*TargetWidth;
    TargetK.at<double>(1,1) = TargetFy*TargetHeight;
    TargetK.at<double>(1,2) = TargetCy*TargetHeight;
    TargetK.at<double>(2,2) = 1.0;

    TargetK2 = cv::Mat(3, 3, CV_64F, 0.0);
    TargetK2.at<double>(0,0) = TargetFx*TargetWidth*2;
    TargetK2.at<double>(0,2) = TargetCx*TargetWidth*2;
    TargetK2.at<double>(1,1) = TargetFy*TargetHeight*2;
    TargetK2.at<double>(1,2) = TargetCy*TargetHeight*2;
    TargetK2.at<double>(2,2) = 1.0;

    int IsRotate = 0;
    fSettings["IsRotate"] >> IsRotate;

    if (IsRotate){
        cv::Mat r90(3,3,CV_64F, cv::Scalar(0));
        r90.at<double>(0,1) = -1.0;
        r90.at<double>(1,0) = 1.0;
        r90.at<double>(2,2) = 1.0;
        R0 = r90 * R0;
        R1 = r90 * R1;
    }

    cv::initUndistortRectifyMap(K0, DistCoef0, R0, TargetK, cv::Size(TargetWidth,TargetHeight), CV_32F, M1l, M2l);
    cv::initUndistortRectifyMap(K1, DistCoef1, R1, TargetK, cv::Size(TargetWidth,TargetHeight), CV_32F, M1r, M2r);

    cv::Mat reye = cv::Mat::eye(3,3,CV_64F);
    cout << reye << std::endl;
    cv::initUndistortRectifyMap(K0, DistCoef0, reye, TargetK, cv::Size(TargetWidth,TargetHeight), CV_32F, unM1l, unM2l);
    cv::initUndistortRectifyMap(K1, DistCoef1, reye, TargetK, cv::Size(TargetWidth,TargetHeight), CV_32F, unM1r, unM2r);

    // For better cost volume
    cv::initUndistortRectifyMap(K1, DistCoef1, reye, TargetK2, cv::Size(TargetWidth*2,TargetHeight*2), CV_32F, unM1r_x2, unM2r_x2);

    cout << "- fps: " << fps << endl;

    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if (mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    mThDepth = mbf*(float)fSettings["ThDepth"]/K0.at<double>(0,0);
    cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;

    mpInitializer = NULL;

    mBasePath = savePath;
}


vector<string> split(const string &s, const string &seperator){
    vector<string> result;
    typedef string::size_type string_size;
    string_size i = 0;

    while(i != s.size()){
        int flag = 0;
        while(i != s.size() && flag == 0){
        flag = 1;
        for(string_size x = 0; x < seperator.size(); ++x)
            if(s[i] == seperator[x]){
            ++i;
            flag = 0;
            break;
            }
        }
    
        flag = 0;
        string_size j = i;
        while(j != s.size() && flag == 0){
            for(string_size x = 0; x < seperator.size(); ++x)
                if(s[j] == seperator[x]){
                    flag = 1;
                    break;
                }
            if(flag == 0) 
                ++j;
        }
        if(i != j){
            result.push_back(s.substr(i, j-i));
            i = j;
        }
    }
    return result;
}

int LoadMatchs(std::string matchPath, vector<vector<cv::Point2f>> & LeftPts, vector<vector<cv::Point2f>> & RightPts){
    FILE * fp = fopen(matchPath.c_str(), "r");
    if (fp == NULL){
        printf("Open %s error!\n", matchPath.c_str());
        return -1;
    }

    LeftPts.clear();
    RightPts.clear();

    vector<cv::Point2f > tLeftPts, tRightPts;

    cv::Point2f pt0,pt1;
    char buffer[1024] = {0};
    while(fgets(buffer,1024,fp)){
        buffer[strlen(buffer) - 1] = '\0';
        std::string tStr = std::string(buffer);
        std::vector<std::string> strVector = split(tStr, " ");

        tLeftPts.clear();
        tRightPts.clear();

        if (strVector.size() != 60){
            printf("split error!!\n");
            break;
        }

        for(int ii = 12; ii < 60; ii+=8){
            pt0.x = atof(strVector[ii + 2].c_str());
            pt0.y = atof(strVector[ii + 3].c_str());

            pt1.x = atof(strVector[ii + 5].c_str());
            pt1.y = atof(strVector[ii + 6].c_str());

            tLeftPts.push_back(pt0);
            tRightPts.push_back(pt1);
        }
        LeftPts.push_back(tLeftPts);
        RightPts.push_back(tRightPts);
    }
    fclose(fp);
    return 0;
}


vector<vector<double>> EstMotion::Est(const cv::Mat &left, const cv::Mat &right, const double &timestamp, string imageName, bool JustStatistic)
{
    mLeftFrame = Frame(left,timestamp,mpORBextractorLeft,mK0,mDistCoef0,mbf,mThDepth,R0,TargetK,cv::Size(TargetWidth,TargetHeight));
    mRightFrame = Frame(right,timestamp,mpORBextractorRight,mK1,mDistCoef1,mbf,mThDepth,R1,TargetK,cv::Size(TargetWidth,TargetHeight));


    // Try to initialize
    if ((int)mLeftFrame.mvKeys.size() <= 100 || (int)mRightFrame.mvKeys.size() <= 100)
    {
        std::cout << "The number of extracted features is too small!" << (int)mLeftFrame.mvKeys.size() 
                                                             << (int)mRightFrame.mvKeys.size() << std::endl;
        return vector<vector<double>>();
    }

    // assign -1 to all matches slots
    fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
    mvbPrevMatched.resize(mLeftFrame.mvKeysRec.size()); // std::vector<cv::Point2f>
    for(size_t i = 0; i < mLeftFrame.mvKeysRec.size(); i++)
        mvbPrevMatched[i] = mLeftFrame.mvKeysRec[i].pt;

    // Find correspondences
    ORBmatcher matcher(0.9,true);
    cv::Mat imLeftRect, imRightRect;
    int nmatches = matcher.SearchForInitialization(mLeftFrame,mRightFrame,mvbPrevMatched,mvIniMatches,200, imLeftRect, imRightRect);

    // Statistic epipolar distance
    if (JustStatistic){
        vector<double> epiDist;
        for (unsigned int ii = 0; ii < mvIniMatches.size(); ++ii){
            if (mvIniMatches[ii] >= 0){
                double leftV = mLeftFrame.mvKeysRec[ii].pt.y;
                double rightV = mRightFrame.mvKeysRec[mvIniMatches[ii]].pt.y;
                epiDist.push_back(rightV - leftV);
            }
        }

        // record epiDist
        FILE * fp = fopen((mBasePath + "/RSResult/epiDist/" + imageName + ".txt").c_str(), "w");        
        for (unsigned int ii = 0; ii < epiDist.size(); ++ii){
            fprintf(fp, "%f\n", epiDist[ii]);
        }
        fclose(fp);

        vector<vector<double>> tm;
        return tm;
    }


    // Show Matches
    bool IsShowRectfiedMatches = false;
    if (IsShowRectfiedMatches){
        cv::remap(left,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
        cv::remap(right,imRightRect,M1r,M2r,cv::INTER_LINEAR);

        std::vector< cv::DMatch > matches;
        // std::vector< char > mask;
        // mask.resize(mvIniMatches.size());
        std::vector< cv::KeyPoint > keyvec0;
        std::vector< cv::KeyPoint > keyvec1;
        int count = 0;
        for (unsigned int ii = 0; ii < mvIniMatches.size(); ++ii){
            if (mvIniMatches[ii] >= 0){
                matches.push_back(cv::DMatch(count,count,0.1));
                keyvec0.push_back(mLeftFrame.mvKeysRec[ii]);
                keyvec1.push_back(mRightFrame.mvKeysRec[mvIniMatches[ii]]);
                count += 1;
            }
        }

        // Show one by one
        printf("Matched number:%d\n", count);
        for (int iid = 0; iid < count; ++iid){
            std::vector< cv::DMatch > tmatches;
            tmatches.push_back(cv::DMatch(iid, iid, 0.1));
            cv::Mat out;
            cv::drawMatches(imLeftRect, keyvec0, imRightRect, keyvec1, tmatches, out);
            cv::imshow("match", out);
            cv::waitKey(0);
        }
    }

    // Set Reference Frame
    if (mpInitializer)
        delete mpInitializer;
    mpInitializer = new Initializer(mLeftFrame, 1.0, 200, "/home/wangke/project/SFRSS/debug/" + imageName);

    cv::Mat Rcw; // Current Camera Rotation 
    cv::Mat tcw; // Current Camera Translation
    vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

    cv::Mat imLeftUn, imRightUn;

    if (mpInitializer->Initialize(mRightFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated, R, T, imLeftUn, imRightUn))
    {
        cout << "Final matches: " << nmatches << endl;
    }

    // Estimate Depth
    return mpInitializer->GetFinalRes();
}

vector<vector<double>> EstMotion::GetProjRes(){
    return mpInitializer->GetProjRes();    
}

vector<vector<double>> EstMotion::GetCloseFormRes(){
    return mpInitializer->GetCloseFormRes();    
}

void EstMotion::Reset()
{

}

// void EstMotion::ChangeCalibration(const string &strSettingPath)
// {

//     cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

//     cv::Mat K0, K1;
//     fSettings["M1"] >> K0;
//     fSettings["M2"] >> K1;

//     K0.copyTo(mK0);
//     K1.copyTo(mK1);

//     cv::Mat DistCoef0, DistCoef1;
//     fSettings["D1"] >> DistCoef0;
//     fSettings["D2"] >> DistCoef1;
//     DistCoef0 = DistCoef0.t();  // (1,N) -> (N,1)
//     DistCoef1 = DistCoef1.t();
//     if (DistCoef0.at<double>(0,4) == 0.0){
//         DistCoef0.resize(4);
//     }
//     else{
//         DistCoef0.resize(5);
//     }

//     if (DistCoef1.at<double>(0,4) == 0.0){
//         DistCoef1.resize(4);
//     }
//     else{
//         DistCoef1.resize(5);
//     }

//     DistCoef0.copyTo(mDistCoef0);
//     DistCoef1.copyTo(mDistCoef1);

//     mbf = fSettings["Camera.bf"];

//     float fps = fSettings["Camera.fps"];
//     if (fps == 0)
//         fps = 4;

//     // Max/Min Frames to insert keyframes and to check relocalisation
//     mMinFrames = 0;
//     mMaxFrames = fps;

//     cout << endl << "Camera Parameters: " << endl;
//     cout << "- fx0: " << K0.at<double>(0,0) << endl;
//     cout << "- fy0: " << K0.at<double>(1,1) << endl;
//     cout << "- cx0: " << K0.at<double>(0,2) << endl;
//     cout << "- cy0: " << K0.at<double>(1,2) << endl;
//     cout << "- k1: " << DistCoef0.at<float>(0) << endl;
//     cout << "- k2: " << DistCoef0.at<float>(1) << endl;
//     if(DistCoef0.rows==5)
//         cout << "- k3: " << DistCoef0.at<float>(4) << endl;
//     cout << "- p1: " << DistCoef0.at<float>(2) << endl;
//     cout << "- p2: " << DistCoef0.at<float>(3) << endl;

//     cout << "- fx1: " << K1.at<double>(0,0) << endl;
//     cout << "- fy1: " << K1.at<double>(1,1) << endl;
//     cout << "- cx1: " << K1.at<double>(0,2) << endl;
//     cout << "- cy1: " << K1.at<double>(1,2) << endl;
//     cout << "- k1: " << DistCoef1.at<float>(0) << endl;
//     cout << "- k2: " << DistCoef1.at<float>(1) << endl;
//     if(DistCoef1.rows==5)
//         cout << "- k3: " << DistCoef1.at<float>(4) << endl;
//     cout << "- p1: " << DistCoef1.at<float>(2) << endl;
//     cout << "- p2: " << DistCoef1.at<float>(3) << endl;

//     Frame::mbInitialComputations = true;
// }



}
