
#include "ORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

// #include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>
#include "../util/globalCalib.h"

using namespace std;

namespace SFRSS
{

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998)
        return 2.5;
    else
        return 4.0;
}


int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize, cv::Mat imLeftRect, cv::Mat imRightRect)
{
    int nmatches=0;
    vnMatches12 = vector<int>(F1.mvKeysRec.size(),-1);
    float factor = 1.0/HISTO_LENGTH;

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    vector<int> vMatchedDistance(F2.mvKeysRec.size(),INT_MAX);
    vector<int> vnMatches21(F2.mvKeysRec.size(),-1);

    for(size_t i1=0, iend1=F1.mvKeysRec.size(); i1<iend1; i1++)
    {
        cv::KeyPoint kp1 = F1.mvKeysRec[i1];
        int level1 = kp1.octave;
        if(level1>0)
            continue;

        if (kp1.pt.y<2 || kp1.pt.x<2 || kp1.pt.x>=wG[0]-2 || kp1.pt.y>=hG[0]-2){
            continue;
        }

        // Get the detected corners near this position in the second image
        vector<size_t> vIndices2 = F2.GetFeaturesInRectedband(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize*1.2, windowSize*0.25, windowSize*0.3, windowSize*0.3, level1,level1);

        if (vIndices2.size() == 0){
            continue;
        }

        // Show : Find results
        bool IsShowFindResults = false;
        if (IsShowFindResults){

            cout << "x:" << vbPrevMatched[i1].x << endl;
            cout << "y:" << vbPrevMatched[i1].y << endl;
            cout << "Number:" << vIndices2.size() << endl;

            cv::Mat im0, im1;
            im0 = imLeftRect.clone();
            im1 = imRightRect.clone();
            cv::cvtColor(im0,im0,cv::COLOR_GRAY2BGR);
            cv::cvtColor(im1,im1,cv::COLOR_GRAY2BGR);

            cv::circle(im0, F1.mvKeysRec[i1].pt, 4, cv::Scalar(0,0,255),2);
            for (unsigned int ii = 0; ii < vIndices2.size(); ++ii){
                cv::circle(im1, F2.mvKeysRec[vIndices2[ii]].pt, 4, cv::Scalar(0,0,255),2);
            }
            cv::circle(im1, F1.mvKeysRec[i1].pt, 4, cv::Scalar(0,255,0),2);

            cv::imshow("left", im0);
            cv::imshow("right", im1);
            cv::waitKey(0);
        }


        if(vIndices2.empty())
            continue;

        cv::Mat d1 = F1.mDescriptors.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = F2.mDescriptors.row(i2);

            int dist = DescriptorDistance(d1,d2);

            if(vMatchedDistance[i2]<=dist)
                continue;

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }

        if(bestDist<=TH_LOW)
        {
            if(bestDist<(float)bestDist2*mfNNratio)
            {
                if(vnMatches21[bestIdx2]>=0)   // have found a correspondecs
                {
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;
                }
                vnMatches12[i1]=bestIdx2;
                vnMatches21[bestIdx2]=i1;
                vMatchedDistance[bestIdx2]=bestDist;
                nmatches++;

                if(mbCheckOrientation)
                {
                    float rot = F1.mvKeysRec[i1].angle-F2.mvKeysRec[bestIdx2].angle;
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1); // statistic 2D rotations
                }
            }
        }

    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)            // select these correspondences with same 2d rotations
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }
    }

    //Update prev matched
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
        if(vnMatches12[i1]>=0)
            vbPrevMatched[i1]=F2.mvKeysRec[vnMatches12[i1]].pt;    // Match results

    return nmatches;
}



void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

}
