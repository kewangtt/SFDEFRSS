
#include "Frame.h"
#include "ORBmatcher.h"
#include <thread>

namespace SFRSS
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mnMinXRec, Frame::mnMinYRec, Frame::mnMaxXRec, Frame::mnMaxYRec;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;
float Frame::mfGridElementWidthInvRec, Frame::mfGridElementHeightInvRec;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2),mTK(frame.mTK.clone()),
     mR(frame.mR.clone())
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}

Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, cv::Mat &R, cv::Mat &TargetK, cv::Size RecShape)
    :mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp),mK(K.clone()),mDistCoef(distCoef.clone()),mbf(bf),mThDepth(thDepth),mTK(TargetK.clone()),
     mR(R.clone()), mRecShape(RecShape)
{
    mImGray = imGray.clone();

    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    bool IsShowBefore = false;
    if (IsShowBefore){
        cv::Mat img;
        cv::cvtColor(imGray, img, cv::COLOR_GRAY2BGR);
        for (unsigned int ii = 0; ii < mvKeys.size(); ++ii){
            cv::circle(img, mvKeys[ii].pt, 2, cv::Scalar(0,0,255));
        }

        cv::imshow("left_before", img);
        cv::waitKey(0);
    }

    UndistortKeyPoints();

    RectifyPoints();

    NormalizedPoints();

    bool IsShowAfter = false;
    if (IsShowAfter){
        cv::Mat img;
        undistort(imGray, img, K, distCoef);
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
        for (unsigned int ii = 0; ii < mvKeysUn.size(); ++ii){
            cv::circle(img, mvKeysUn[ii].pt, 2, cv::Scalar(0,0,255));
        }

        cv::imshow("left_after", img);
        cv::waitKey(0);
    }


    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<double>(0,0);
        fy = K.at<double>(1,1);
        cx = K.at<double>(0,2);
        cy = K.at<double>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mnMinXRec = 0;
        mnMinYRec = 0;
        mnMaxXRec = mRecShape.width;
        mnMaxYRec = mRecShape.height;

        mfGridElementWidthInvRec = static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxXRec-mnMinXRec);
        mfGridElementHeightInvRec = static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxYRec-mnMinYRec);

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    mRecN = 0;
    AssignFeaturesToGrid();
    AssignFeaturesToGridRec();
}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}


void Frame::AssignFeaturesToGridRec()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGridRec[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysRec[i];

        int nGridPosX, nGridPosY;
        if(PosInGridRec(kp,nGridPosX,nGridPosY)){
            mGridRec[nGridPosX][nGridPosY].push_back(i);
            mRecN += 1;
        }
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}
// r = 100
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv)); // calc corresponding cellX
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

vector<size_t> Frame::GetFeaturesInRectedband(const float &x, const float &y, const float &rwl, const float &rwr, const float &rwu, const float &rwb, const int minLevel, const int maxLevel){
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellXRec = max(0,(int)floor((x-mnMinXRec-rwl)*mfGridElementWidthInvRec)); // calc corresponding cellX
    if(nMinCellXRec>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellXRec = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinXRec+rwr)*mfGridElementWidthInvRec));
    if(nMaxCellXRec<0)
        return vIndices;

    const int nMinCellYRec = max(0,(int)floor((y-mnMinYRec-rwu)*mfGridElementHeightInvRec));
    if(nMinCellYRec>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellYRec = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinYRec+rwb)*mfGridElementHeightInvRec));
    if(nMaxCellYRec<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellXRec; ix<=nMaxCellXRec; ix++)
    {
        for(int iy = nMinCellYRec; iy<=nMaxCellYRec; iy++)
        {
            const vector<size_t> vCell = mGridRec[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpRec = mvKeysRec[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpRec.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpRec.octave>maxLevel)
                            continue;
                }

                // cout << "x:" << x << "y:" << y << "xx:" << kpRec.pt.x << "yy:" << kpRec.pt.y << endl;

                const float distx = kpRec.pt.x-x;
                const float disty = kpRec.pt.y-y;

                if (distx < 0 && fabs(distx) > rwl){
                    continue;
                }

                if (distx > 0 && fabs(distx) > rwr){
                    continue;
                }

                if (disty < 0 && fabs(disty) > rwu){
                    continue;
                }

                if (disty > 0 && fabs(disty) > rwb){
                    continue;
                }

                vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}

bool Frame::PosInGridRec(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinXRec)*mfGridElementWidthInvRec);
    posY = round((kp.pt.y-mnMinYRec)*mfGridElementHeightInvRec);

    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<double>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK); // the same intrinsic mK = mK
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::RectifyPoints(){
    
    if(mR.at<double>(0,0)==0.0)
    {
        mvKeysRec = mvKeysUn;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(3,N,CV_64F);
    for(int i=0; i<N; i++)
    {
        mat.at<double>(0,i) = mvKeysUn[i].pt.x;
        mat.at<double>(1,i) = mvKeysUn[i].pt.y;
        mat.at<double>(2,i) = 1.0;
    }

    // cv:: Mat tmTK, tMR;
    // mTK.convertTo(tmTK, )
    mat = mTK * mR * mK.inv() * mat;  // transfer to new intrinsic space
    
    // Fill rectified keypoint vector
    mvKeysRec.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeysUn[i];
        kp.pt.x = mat.at<double>(0,i) / mat.at<double>(2,i);
        kp.pt.y = mat.at<double>(1,i) / mat.at<double>(2,i);
        mvKeysRec[i]=kp;
    }
}


void Frame::NormalizedPoints(){
    
    if(mK.at<double>(0,0)==0.0)
    {
        mvKeysNorm = mvKeysUn;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(3,N,CV_64F);
    for(int i=0; i<N; i++)
    {
        mat.at<double>(0,i) = mvKeysUn[i].pt.x;
        mat.at<double>(1,i) = mvKeysUn[i].pt.y;
        mat.at<double>(2,i) = 1.0;
    }

    mat = mK.inv() * mat; // Normalized coordinates
    
    // Fill rectified keypoint vector
    mvKeysNorm.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeysUn[i];
        kp.pt.x = mat.at<double>(0,i) / mat.at<double>(2,i);
        kp.pt.y = mat.at<double>(1,i) / mat.at<double>(2,i);
        mvKeysNorm[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<double>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}

cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

} 
