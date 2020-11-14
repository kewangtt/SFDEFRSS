#include "System.h"
#include <thread>
#include <iomanip>
#include <string>
#include <iostream>
#include "../RefineMotion/RefineMotion.h"
#include "../util/globalCalib.h"
#include "../util/globalFuncs.h"
#include <Eigen/Geometry>

using namespace std;
namespace SFRSS
{
System::System(const string &strSettingsFile, string basePath, int workMode){

    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened()){
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }
    fsSettings.release();

    InitialGlobalInfo(strSettingsFile);
    mptrUndistImg = NULL;


    mSaveDirPath = basePath + "/Result";
    mBasePath = basePath;

    // Initialize the Tracking thread
    // (it will live in the main thread of execution, the one that called this constructor)
    mpEstMotion = new EstMotion(this, strSettingsFile, basePath);
  
    SetBaselineScaleG(2);
    mpRefineMotion = new RefineMotion(this, RowTimeG);
    mpStereoMatch = new StereoMatch(this, strSettingsFile);

    // Build Mapping object
    int input_depth = 8; // corrected by gamma curve 8bits ->16bits
    int output_depth = 16;
    mdisparitySize = 256;
    mIsSubPixel = true;
    if (mdisparitySize >= 256 || mIsSubPixel){
        output_depth = 16;
    }
    if (mIsSubPixel){
        mSubPixelLvl = 16;
    }
    else{
        mSubPixelLvl = 1;
    }
    mInvalidDisp = mpStereoMatch->InitStereoMatch(mdisparitySize, input_depth, output_depth, mIsSubPixel);
    SetDispG(mdisparitySize, mInvalidDisp, mSubPixelLvl);
    mpStereoMatch->InitRSProjection(10, 0); 
    mpStereoMatch->SetShowMaxDisparity(mdisparitySize); // MaxShowDisparity

    mptrDisparity0 = (void *)malloc(wG[0]*hG[0]*output_depth/8);
    mptrDisparity1 = (void *)malloc(wG[0]*hG[0]*output_depth/8);
    mptrUndistImg = (uint8_t *)malloc(wG[0]*hG[0]);
    mptrDisparityBuffer = NULL;

    ptrLeft = new FramePym();
    ptrRight = new FramePym();

    mworkMode = workMode;

    mptrDepthMap = (double *)malloc(sizeof(double)*wG[0]*hG[0]);
    mptrDepthMap1 = (double *)malloc(sizeof(double)*wG[0]*hG[0]);
    mptrErrorMap = (double *)malloc(sizeof(double)*wG[0]*hG[0]);
}

System::~System(){
    if (mptrDisparityBuffer) free(mptrDisparityBuffer);
    if (mptrDisparity0) free(mptrDisparity0);
    if (mptrDisparity1) free(mptrDisparity1);
    if (mptrErrorMap) free(mptrErrorMap);
    if (mptrUndistImg) free(mptrUndistImg);
    if (mptrDepthMap) free(mptrDepthMap);
    if (mptrDepthMap1) free(mptrDepthMap1);

    delete ptrLeft;
    delete ptrRight;
}

uint8_t vecMed(std::vector<uint8_t> vec){
    if(vec.empty()) return 0;
    else {
        std::sort(vec.begin(), vec.end());
        if(vec.size() % 2 == 0)
            return (vec[vec.size()/2 - 1] + vec[vec.size()/2]) / 2;
        else
            return vec[vec.size()/2];
    }
}


int SaveRawDisparity(uint16_t * mptrDisparity, string dirpath, string name){
    // SaveDisbarity
    dirpath = dirpath + "/" + name + ".bin";
    FILE * fptr = fopen(dirpath.c_str(), "wb");
    unsigned int unit_size = 2;
    fwrite(&(hG[0]), sizeof(int), 1, fptr);
    fwrite(&(wG[0]), sizeof(int), 1, fptr);
    fwrite(&(unit_size), sizeof(unit_size), 1, fptr);
    fwrite(&(subPixelLevlG), sizeof(subPixelLevlG), 1, fptr);
    fwrite(mptrDisparity, sizeof(uint16_t), hG[0]*wG[0], fptr);
    fclose(fptr);

    return 0;
}



// invalided value: 0
int SaveRawDepthMap(double * mptrDepthMap, string dirpath, string name){
    // SaveDisbarity
    dirpath = dirpath + "/" + name + ".bin";
    FILE * fptr = fopen(dirpath.c_str(), "wb");
    fwrite(mptrDepthMap, sizeof(double), hG[0]*wG[0], fptr);
    fclose(fptr);

    return 0;
}


int SaveRGBDisparity(uint16_t * mptrDisparity, string dirpath, string name, bool IsFlip = false){
    dirpath = dirpath + "/" + name;

    int lvl = 0;
	cv::Mat_<uint16_t> disparity(hG[lvl], wG[lvl]); 
	cv::Mat disparity_8u,disparity_color;

	memcpy(disparity.data, mptrDisparity, wG[lvl]*hG[lvl]*sizeof(uint16_t)); // uint16_t

	// Transfer disparity to color space (uint8 space)
	disparity.convertTo(disparity_8u, CV_8U, 255. / maxDisparityG[lvl] / subPixelLevlG);
	cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);
	disparity_color.setTo(cv::Scalar(0, 0, 0), disparity >= 256*subPixelLevlG);
	disparity_color.setTo(cv::Scalar(0, 0, 0), disparity == invalidDispG);
	disparity_color.setTo(cv::Scalar(0, 0, 0), disparity == 0);

    if (IsFlip){
        cv::flip(disparity_color, disparity_color, 0);
        cv::flip(disparity_color, disparity_color, 1);
    }

	cv::imwrite(dirpath, disparity_color);

    return 0;
}




// Undist RS effect and calc 3D point cloud
int UndistRSEffectToRect(uint8_t * img, uint16_t * disparity, int lvl, 
                   vector<double> motionState, 
                   uint8_t * destImg, 
                   double * destDepthMap,
                   vector<Vec3> * pointtCloud){
    pointtCloud->clear();

	Vec3 Velocity(motionState[3] * RowTimeG, motionState[4] * RowTimeG, motionState[5] * RowTimeG);
	Vec3 Angular(motionState[0] * RowTimeG, motionState[1] * RowTimeG, motionState[2] * RowTimeG);

	double AngSpeed = Angular.norm();
	Vec3 AngAxis;
	if (AngSpeed < 0.0000000001){
	    AngAxis[0] = 1.0;
	    AngAxis[1] = 1.0;
	    AngAxis[2] = 1.0;
		AngAxis = AngAxis / AngAxis.norm();
	}
	else{
		AngAxis = Angular / AngSpeed;
	}

    int scale;
    scale = 1 << lvl;
    Vec3 curLeftPoint;

    memset(destDepthMap, 0, sizeof(double)*wG[lvl]*hG[lvl]);
    memset(destImg, 0, sizeof(uint8_t)*wG[lvl]*hG[lvl]);
    for (int vv = 0; vv < hG[lvl]; ++vv){
        for (int uu = 0; uu < wG[lvl]; ++uu){
            int iid = vv*wG[lvl] + uu;
            uint16_t disp = disparity[iid];
            if (disp == invalidDispG || disp >= subPixelLevlG*maxDisparityG[lvl]){
                continue;
            }
            
            curLeftPoint << uu, vv, 1.0;
            curLeftPoint = KiG[lvl] * curLeftPoint;

            float rawU = (uu + 0.5)*scale - 0.5;
            float rawV = (vv + 0.5)*scale - 0.5;
            double leftRow = getInterpolatedElement(ptrLeftRemapYG, rawU, rawV, wG[0]);

            // construct ray
            Mat33 Rl = Eigen::AngleAxisd(AngSpeed*leftRow, AngAxis).matrix();
            Mat33 lrinv = Rl.inverse();
            Vec3 start = -leftRow*lrinv*Velocity;
            Vec3 inc = lrinv * curLeftPoint;

            // cout << "Velocity" << Velocity<<std::endl;
            // printf("%d\n",subPixelLevlG);

            double dispf = disparity[iid];
            // printf("dispf:%f\n", dispf);
            dispf = (dispf/subPixelLevlG) + 0.000001; // +1 
            // printf("baselineScaleG: %d\n", baselineScaleG);
            // printf("baselineG: %f\n", baselineG);
            double depthWorld = baselineScaleG * baselineG * fxG[lvl] / dispf; // baselineScaleG * 
            // printf("dispf2:%f\n", depthWorld);

            double lamba = (depthWorld - start[2]) / inc[2];
            Vec3 curWorldPoint = start + lamba * inc;
            // Notice this rotate
            curWorldPoint = RectLeftRG * curWorldPoint;

            pointtCloud->push_back(curWorldPoint);
            double curDepth = curWorldPoint[2];

            // pointtCloud->push_back(curWorldPoint);
            curWorldPoint = KG[lvl] * curWorldPoint;
            double Ku = curWorldPoint[0] / curWorldPoint[2];
            double Kv = curWorldPoint[1] / curWorldPoint[2];
            // printf("%f %f\n", Ku, Kv);
            int Kui = int(Ku + 0.5);
            int Kvi = int(Kv + 0.5);
            if (Kui < 0 || Kvi < 0 || Kui > wG[lvl] - 1 || Kvi > hG[lvl] - 1){
                continue;
            }
            destImg[Kvi*wG[lvl]+Kui] = img[iid];
            destDepthMap[Kvi*wG[lvl]+Kui] = curDepth;
            // destImg[iid] = img[iid];
            // destDisparity[iid] = disparity[iid];
        }
    }
    return 0;
}


void InterpolateDisp(uint8_t * ptrLeftImage, uint16_t * ptr_result_u16, int lvl)
{
    int localw = wG[lvl];
    int localh = hG[lvl];

    uint16_t *ptr_result_new = (uint16_t *)malloc(2*localw*localh);
    memcpy(ptr_result_new, ptr_result_u16, 2*localw*localh);

    // Bilateral interpolate for invalid value
    int windowSize = 7;
    int Radius = windowSize / 2;
    double sigma22 = 2*1*1;
    double ConfidenceThreshold = 0.1;
    double IntensityTheta = 0.5;
    double aveSumWeights = 0;
    double aveNum = 0;
    double coe = 14.794422;


    for (int vv = Radius; vv < localh - Radius; ++vv){
        for (int uu = Radius; uu < localw - Radius; ++uu){            
            int iid = vv*localw + uu;
            if (ptr_result_u16[iid] == invalidDispG){
                // Try to interpolate
                double SumWeights = 0;
                double TargetValue = 0;
                double CurWeight = 0;
                double dist = 0;
                float curIntensity = ptrLeftImage[iid];
                for (int lv = -Radius; lv <= Radius; ++lv){
                    int lrow = lv * localw;
                    for (int lu = -Radius; lu <= Radius; ++lu){
                        int liid = iid + lrow + lu;
                        if (ptr_result_u16[liid] == invalidDispG){
                            continue;
                        }
                        dist = sqrtf(lv*lv + lu*lu) + IntensityTheta * fabs(curIntensity - ptrLeftImage[liid]);
                        CurWeight = exp(-dist/sigma22)/coe;
                        SumWeights += CurWeight;
                        TargetValue += CurWeight*ptr_result_u16[liid];
                    }
                }
                if (SumWeights > 0){
                    aveSumWeights += SumWeights;
                    aveNum += 1;
                    // printf("%f\n", SumWeights);
                }
                
                if (SumWeights > ConfidenceThreshold){
                    TargetValue = TargetValue / SumWeights;
                    ptr_result_new[iid] = (uint16_t)TargetValue;
                }
            }
        }
    }
    memcpy(ptr_result_u16, ptr_result_new, 2*localw*localh);
    free(ptr_result_new);

}

int SaveStates(std::string path, vector<vector<double>> States){
    FILE * fp = fopen(path.c_str(), "w");
    for (unsigned int iid = 0; iid < States.size(); ++iid){
        vector<double> initialState = States[iid];
        fprintf(fp, "%.10f %.10f %.10f %.10f %.10f %.10f\n", initialState[0], initialState[1], initialState[2], initialState[3], initialState[4], initialState[5]);
    }
    fclose(fp);

    return 0;
}

int System::Run2(const cv::Mat & leftImage, const cv::Mat & rightImage, double tframe, string imageName){

    bool IsSave = true;
    bool IsShow = false;
    int endLvl = 1;

    vector<double> initialState;
    vector<double> finalState;

    vector<vector<double>> states;
    cv::Mat LeftUnDistTarget;
    cv::Mat RightUnDistTarget;

    cv::remap(leftImage, LeftUnDistTarget, unM1lG, unM2lG, cv::INTER_LINEAR);
    cv::remap(rightImage, RightUnDistTarget, unM1rG, unM2rG, cv::INTER_LINEAR);

    int pyramidLvl = PyrLevelsUsedG; // PyrLevelsUsedG;
    char Strbuffer[128] = {0};
    void * ptrDisparity = mptrDisparity0;

    ptrLeft->makeImages(LeftUnDistTarget.data);
    ptrRight->makeImages(RightUnDistTarget.data);
    mpRefineMotion->InitRefineMotion(ptrLeft, ptrRight);
    mpStereoMatch->SetImagePair(ptrLeft, ptrRight);
    vector<Vec3> pointCloud;

    if (lastState.size() > 0){
        finalState = lastState;         
    }
    else{
        finalState.push_back(0.0);
        finalState.push_back(0.0);
        finalState.push_back(0.0);
        finalState.push_back(0.0);
        finalState.push_back(0.0);
        finalState.push_back(0.0);
    }

    int iters[PYR_LEVELS] = {1,2,3,5,3,3};
    for (int lvl = pyramidLvl - 1; lvl >= endLvl; --lvl){
        for (int kk = 0; kk < iters[lvl]; ++kk){
            mpStereoMatch->RSProjection3(finalState, lvl);
            mpStereoMatch->StereoMatchFinalMode4(lvl, ptrDisparity);
            finalState = mpRefineMotion->Refine2((uint16_t *)ptrDisparity, finalState, lvl);
        }

        if (lvl == endLvl){
            mpStereoMatch->RSProjection3(finalState, 0);
            mpStereoMatch->StereoMatchFinalMode4(0, ptrDisparity, true);

            // IsSave
            if (IsSave){
                // SaveRawDisparity((uint16_t *)ptrDisparity, mSaveDirPath + "/02/bin", imageName);
                char tempbuffer[256] = {0};
                memcpy(tempbuffer, imageName.c_str(), imageName.length() - 4);
                
                UndistRSEffectToRect(LeftUnDistTarget.data,(uint16_t *)ptrDisparity,0,finalState,mptrUndistImg,mptrDepthMap, &pointCloud);
                SaveRawDepthMap(mptrDepthMap, mSaveDirPath + "/undist_depth", tempbuffer);
                // cv::Mat undistImg_Rect(hG[0], wG[0], CV_8U, mptrUndistImg);
                // cv::imwrite(mSaveDirPath + "/undist_rect/rgb/"+ imageName, undistImg_Rect);

                InterpolateDisp((uint8_t *)LeftUnDistTarget.data, (uint16_t *)ptrDisparity, 0);
                SaveRGBDisparity((uint16_t *)ptrDisparity, mSaveDirPath + "/disparity_rgb", imageName);
            }

            if (IsShow){
                sprintf(Strbuffer, "Final_%d_%d", 0, 0);
                InterpolateDisp((uint8_t *)LeftUnDistTarget.data, (uint16_t *)ptrDisparity, 0);
                mpStereoMatch->ShowDisparity(ptrDisparity, std::string("new_disparity") + Strbuffer, 0, false);
                cv::waitKey(0);
            }
        }
    }

    // Save state
    if (IsSave){
        char tempbuffer[256] = {0};
        memcpy(tempbuffer, imageName.c_str(), imageName.length() - 4);

        FILE * fp = fopen((mSaveDirPath + "/state1.txt").c_str(), "a");
        fprintf(fp, "%s %.5f %.5f %.5f %.5f %.5f %.5f\n", tempbuffer, finalState[0], finalState[1], finalState[2], finalState[3], finalState[4], finalState[5]);
        fclose(fp);

        // Save left undist image and right undist image
        cv::imwrite((mSaveDirPath + "/left/" + imageName).c_str(), LeftUnDistTarget);
        cv::imwrite((mSaveDirPath + "/right/" + imageName).c_str(), RightUnDistTarget);

        // save baseline map
        cv::Mat baselineMap;
        mpStereoMatch->GenerateBaselineMap(baselineMap);
        cv::imwrite((mSaveDirPath + "/baselineMap/" + imageName).c_str(), baselineMap);

        lastState = finalState;
    }

    return 0;
}

int System::Run3(const cv::Mat & leftImage, const cv::Mat & rightImage, double tframe, string imageName){

    vector<double> initialState;
    vector<double> finalState;

    vector<vector<double>> states;
    cv::Mat LeftUnDistTarget;
    cv::Mat RightUnDistTarget;

    cv::remap(leftImage,LeftUnDistTarget,unM1lG,unM2lG,cv::INTER_LINEAR);
    cv::remap(rightImage,RightUnDistTarget,unM1rG,unM2rG,cv::INTER_LINEAR);

    vector<vector<double>> SampsonStates = mpEstMotion->Est(leftImage, rightImage, tframe, imageName, false);
    vector<vector<double>> CloseFormStates = mpEstMotion->GetCloseFormRes();

    for (unsigned int iid = 0; iid < CloseFormStates.size(); ++iid){
        printf("%d: %f %f %f %f %f %f\n", iid, CloseFormStates[iid][0], CloseFormStates[iid][1], CloseFormStates[iid][2],
                                    CloseFormStates[iid][3], CloseFormStates[iid][4], CloseFormStates[iid][5]);
    }

    char tempbuffer[256] = {0};
    memcpy(tempbuffer, imageName.c_str(), imageName.length() - 4);
    SaveStates(mSaveDirPath + "/CloseForm/" + tempbuffer + ".txt", CloseFormStates);
    return 0;
}

void System::Reset()
{
}

void System::Shutdown()
{
}

} 
