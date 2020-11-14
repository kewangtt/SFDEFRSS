
#include "Initializer.h"
#include "ORBmatcher.h"
#include <thread>
#include "MinimalSolver.h"
#include "../util/DUtils/Random.h"
#include "../util/globalCalib.h"

namespace SFRSS
{

Initializer::Initializer(Frame &ReferenceFrame, float sigma, int iterations, std::string path)
{
    mK = ReferenceFrame.mK.clone();
    targetK = ReferenceFrame.mTK.clone();

    mvKeys1 = ReferenceFrame.mvKeys;
    mvKeys1Un = ReferenceFrame.mvKeysUn;
    mvKeys1Norm = ReferenceFrame.mvKeysNorm;

    mSigma = sigma;
    mSigma2 = sigma * sigma;
    mMaxIterations = iterations;
    mPath = path;
}

int Initializer::SetUndistImagePair(cv::Mat &leftImg, cv::Mat &rightImg)
{
    mLeftImgRgb = leftImg.clone();
    mRightImgRgb = rightImg.clone();

    cv::cvtColor(mLeftImgRgb, mLeftImgRgb, cv::COLOR_GRAY2BGR);
    cv::cvtColor(mRightImgRgb, mRightImgRgb, cv::COLOR_GRAY2BGR);

    return 0;
}

bool Initializer::Initialize(Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                             vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, cv::Mat &R, cv::Mat &T, cv::Mat img0, cv::Mat img1)
{
    mvKeys2 = CurrentFrame.mvKeys;         
    mvKeys2Norm = CurrentFrame.mvKeysNorm; 
    mvKeys2Un = CurrentFrame.mvKeysUn;
    mK2 = CurrentFrame.mK;
    
    mR = R.clone();
    mT = T.clone();

    mvMatches12.clear();
    mvMatches12.reserve(mvKeys2.size()); // reserve
    mvbMatched1.resize(mvKeys1.size());
    for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
    {
        if (vMatches12[i] >= 0)
        {
            mvMatches12.push_back(make_pair(i, vMatches12[i]));
            mvbMatched1[i] = true;
        }
        else
            mvbMatched1[i] = false;
    }

    const int N = mvMatches12.size();

    // Indices for minimum set selection
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for (int i = 0; i < N; i++)
    {
        vAllIndices.push_back(i);
    }

    // Generate sets of 8 points for each RANSAC iteration
    mvSets = vector<vector<size_t>>(mMaxIterations, vector<size_t>(6, 0));

    // DUtils::Random::SeedRandOnce(0);

    for (int it = 0; it < mMaxIterations; it++)
    {
        vAvailableIndices = vAllIndices; // sample seeds in all indices every time

        // Select a minimum set
        for (size_t j = 0; j < 6; j++)
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);
            int idx = vAvailableIndices[randi];

            mvSets[it][j] = idx;

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    // Show
    bool IsShowSets = false;
    std::vector<cv::DMatch> matches;
    std::vector<cv::KeyPoint> keyvec0;
    std::vector<cv::KeyPoint> keyvec1;
    std::vector<cv::KeyPoint> keyvec3;
    std::vector<cv::KeyPoint> keyvec4;
    if (IsShowSets)
    {
        for (unsigned int ii = 0; ii < mvSets.size(); ++ii)
        {
            keyvec0.clear();
            keyvec1.clear();
            keyvec3.clear();
            keyvec4.clear();
            matches.clear();
            cv::Mat tmpLeftUn = img0.clone();
            cv::Mat tmpRightUn = img1.clone();
            for (unsigned int jj = 0; jj < mvSets[ii].size(); ++jj)
            {
                int leftIndx = mvMatches12[mvSets[ii][jj]].first;
                int rightIndx = mvMatches12[mvSets[ii][jj]].second;
                cv::KeyPoint lkp = mvKeys1Un[leftIndx];
                cv::KeyPoint rkp = mvKeys2Un[rightIndx];
                keyvec0.push_back(lkp);
                keyvec1.push_back(rkp);
                matches.push_back(cv::DMatch(jj, jj, 0.3));

                cv::KeyPoint lkpN = mvKeys1Norm[leftIndx];
                cv::KeyPoint rkpN = mvKeys2Norm[rightIndx];
                lkpN.pt.x = lkpN.pt.x * mK.at<double>(0, 0) + mK.at<double>(0, 2);
                lkpN.pt.y = lkpN.pt.y * mK.at<double>(1, 1) + mK.at<double>(1, 2);

                rkpN.pt.x = rkpN.pt.x * mK2.at<double>(0, 0) + mK2.at<double>(0, 2);
                rkpN.pt.y = rkpN.pt.y * mK2.at<double>(1, 1) + mK2.at<double>(1, 2);

                keyvec3.push_back(lkpN);
                keyvec4.push_back(rkpN);
            }

            cv::Mat out, out1;
            cv::drawMatches(tmpLeftUn, keyvec0, tmpRightUn, keyvec1, matches, out);
            cv::drawMatches(tmpLeftUn, keyvec3, tmpRightUn, keyvec4, matches, out1);
            cv::resize(out, out, cv::Size(), 0.4, 0.4, cv::INTER_LINEAR_EXACT);
            cv::resize(out1, out1, cv::Size(), 0.4, 0.4, cv::INTER_LINEAR_EXACT);
            cv::imshow("match0", out);
            cv::imshow("match1", out1);
            cv::waitKey(0);
        }
    }

    // SaveParam(mvSets);

    FindMotionMThread(mvSets);

    return true;
}

void ComposeRT(const CvMat *_wvec, const CvMat *_vvec, // rotation and velocity
               const CvMat *_rvec, const CvMat *_tvec,
               double leftrow, double rightrow,
               CvMat *_rvec3, CvMat *_tvec3,
               CvMat *dr3dw, CvMat *dr3dv,
               CvMat *dt3dw, CvMat *dt3dv)
{
    double _r[3], _w[3];
    double _R1[9], _d1[9 * 3], _R2[9], _d2[9 * 3];
    CvMat r = cvMat(3, 1, CV_64F, _r), w = cvMat(3, 1, CV_64F, _w);
    CvMat R1 = cvMat(3, 3, CV_64F, _R1), R2 = cvMat(3, 3, CV_64F, _R2);
    CvMat dR1dr1 = cvMat(9, 3, CV_64F, _d1), dR2dr2 = cvMat(9, 3, CV_64F, _d2);

    CV_Assert(_rvec->rows == 3 && _rvec->cols == 1 && CV_ARE_SIZES_EQ(_rvec, _wvec));

    cvConvert(_rvec, &r);
    cvConvert(_wvec, &w); // w * (rightrow - leftrow)
    _w[0] = _w[0] * (rightrow - leftrow);
    _w[1] = _w[1] * (rightrow - leftrow);
    _w[2] = _w[2] * (rightrow - leftrow);

    cvRodrigues2(&w, &R1, &dR1dr1);
    cvRodrigues2(&r, &R2, &dR2dr2);

    double _r3[3], _R3[9], _dR3dR1[9 * 9], _dR3dR2[9 * 9], _dr3dR3[9 * 3];
    double _W1[9 * 3], _W2[3 * 3];
    CvMat r3 = cvMat(3, 1, CV_64F, _r3), R3 = cvMat(3, 3, CV_64F, _R3);
    CvMat dR3dR1 = cvMat(9, 9, CV_64F, _dR3dR1), dR3dR2 = cvMat(9, 9, CV_64F, _dR3dR2);
    CvMat dr3dR3 = cvMat(3, 9, CV_64F, _dr3dR3);
    CvMat W1 = cvMat(3, 9, CV_64F, _W1), W2 = cvMat(3, 3, CV_64F, _W2);

    cvMatMul(&R2, &R1, &R3);
    // Deriv
    cvCalcMatMulDeriv(&R2, &R1, &dR3dR2, &dR3dR1);
    // derivative, SO3->so3
    cvRodrigues2(&R3, &r3, &dr3dR3);

    // _rvec3: an output variable
    if (_rvec3)
        cvConvert(&r3, _rvec3);

    if (dr3dw)
    {
        cvMatMul(&dr3dR3, &dR3dR1, &W1);
        cvMatMul(&W1, &dR1dr1, &W2);
        cvConvert(&W2, dr3dw);

        double *ptrdata = dr3dw->data.db;
        ptrdata[0] = ptrdata[0] * (rightrow - leftrow);
        ptrdata[1] = ptrdata[1] * (rightrow - leftrow);
        ptrdata[2] = ptrdata[2] * (rightrow - leftrow);
        ptrdata[3] = ptrdata[3] * (rightrow - leftrow);
        ptrdata[4] = ptrdata[4] * (rightrow - leftrow);
        ptrdata[5] = ptrdata[5] * (rightrow - leftrow);
        ptrdata[6] = ptrdata[6] * (rightrow - leftrow);
        ptrdata[7] = ptrdata[7] * (rightrow - leftrow);
        ptrdata[8] = ptrdata[8] * (rightrow - leftrow);
    }

    if (dr3dv)
        cvZero(dr3dv);

    if (dt3dv || dt3dw)
    {
        double _t1[3], _t1_v0[3], _t1_v1[3], _t2[3], _t3[3], _dxdR2[3 * 9], _dxdR1[3 * 9], _dxdt1[3 * 3], _dxdt1_v0[3 * 3], _W3[3 * 9], _W4[3 * 3];
        CvMat t1 = cvMat(3, 1, CV_64F, _t1), t2 = cvMat(3, 1, CV_64F, _t2);
        CvMat t1_v0 = cvMat(3, 1, CV_64F, _t1_v0), t1_v1 = cvMat(3, 1, CV_64F, _t1_v1);
        CvMat t3 = cvMat(3, 1, CV_64F, _t3);
        CvMat dxdR2 = cvMat(3, 9, CV_64F, _dxdR2);
        CvMat dxdt1 = cvMat(3, 3, CV_64F, _dxdt1);
        CvMat dxdR1 = cvMat(3, 9, CV_64F, _dxdR1);
        CvMat dxdt1_v0 = cvMat(3, 3, CV_64F, _dxdt1_v0);
        CvMat W3 = cvMat(3, 9, CV_64F, _W3);
        CvMat W4 = cvMat(3, 3, CV_64F, _W4);

        CV_Assert(CV_IS_MAT(_tvec) && CV_IS_MAT(_vvec));
        CV_Assert(CV_ARE_SIZES_EQ(_tvec, _vvec));

        cvConvert(_vvec, &t1_v0);
        cvConvert(_vvec, &t1_v1);
        _t1_v0[0] = -1 * _t1_v0[0] * leftrow;
        _t1_v0[1] = -1 * _t1_v0[1] * leftrow;
        _t1_v0[2] = -1 * _t1_v0[2] * leftrow;

        _t1_v1[0] = _t1_v1[0] * rightrow;
        _t1_v1[1] = _t1_v1[1] * rightrow;
        _t1_v1[2] = _t1_v1[2] * rightrow;

        cvMatMulAdd(&R1, &t1_v0, &t1_v1, &t1);
        cvConvert(_tvec, &t2);

        cvMatMulAdd(&R2, &t1, &t2, &t3);

        if (_tvec3)
            cvConvert(&t3, _tvec3);

        if (dt3dw || dt3dv)
        {
            cvCalcMatMulDeriv(&R2, &t1, &dxdR2, &dxdt1);
            cvCalcMatMulDeriv(&R1, &t1_v0, &dxdR1, &dxdt1_v0);
            if (dt3dw)
            {
                cvMatMul(&dxdt1, &dxdR1, &W3);
                cvMatMul(&W3, &dR1dr1, &W4);
                cvConvert(&W4, dt3dw); // * delta row

                double *ptrdata = dt3dw->data.db;
                ptrdata[0] = ptrdata[0] * (rightrow - leftrow);
                ptrdata[1] = ptrdata[1] * (rightrow - leftrow);
                ptrdata[2] = ptrdata[2] * (rightrow - leftrow);
                ptrdata[3] = ptrdata[3] * (rightrow - leftrow);
                ptrdata[4] = ptrdata[4] * (rightrow - leftrow);
                ptrdata[5] = ptrdata[5] * (rightrow - leftrow);
                ptrdata[6] = ptrdata[6] * (rightrow - leftrow);
                ptrdata[7] = ptrdata[7] * (rightrow - leftrow);
                ptrdata[8] = ptrdata[8] * (rightrow - leftrow);
            }
            if (dt3dv)
            {
                _dxdt1_v0[0] = -leftrow * _dxdt1_v0[0] + rightrow;
                _dxdt1_v0[1] = -leftrow * _dxdt1_v0[1];
                _dxdt1_v0[2] = -leftrow * _dxdt1_v0[2];
                _dxdt1_v0[3] = -leftrow * _dxdt1_v0[3];
                _dxdt1_v0[4] = -leftrow * _dxdt1_v0[4] + rightrow;
                _dxdt1_v0[5] = -leftrow * _dxdt1_v0[5];
                _dxdt1_v0[6] = -leftrow * _dxdt1_v0[6];
                _dxdt1_v0[7] = -leftrow * _dxdt1_v0[7];
                _dxdt1_v0[8] = -leftrow * _dxdt1_v0[8] + rightrow;

                cvMatMul(&dxdt1, &dxdt1_v0, &W4);
                cvConvert(&W4, dt3dv);
            }
        }
    }
}

double StateDist(vector<double> &state0, vector<double> &state1, double rw, double rv, double scale)
{
    double dist0 = sqrt((state0[0] - state1[0]) * (state0[0] - state1[0]) +
                        (state0[1] - state1[1]) * (state0[1] - state1[1]) +
                        (state0[2] - state1[2]) * (state0[2] - state1[2]));

    double dist1 = sqrt((state0[3] - state1[3]) * (state0[3] - state1[3]) +
                        (state0[4] - state1[4]) * (state0[4] - state1[4]) +
                        (state0[5] - state1[5]) * (state0[5] - state1[5]));

    return (rw * dist0 + rv * dist1) / scale;
}

int StateDist1(double *state0, double *state1, double scale, double *pdist)
{
    double dist0 = sqrt((state0[0] - state1[0] * scale) * (state0[0] - state1[0] * scale) +
                        (state0[1] - state1[1] * scale) * (state0[1] - state1[1] * scale) +
                        (state0[2] - state1[2] * scale) * (state0[2] - state1[2] * scale));

    double dist1 = sqrt((state0[3] - state1[3] * scale) * (state0[3] - state1[3] * scale) +
                        (state0[4] - state1[4] * scale) * (state0[4] - state1[4] * scale) +
                        (state0[5] - state1[5] * scale) * (state0[5] - state1[5] * scale));

    pdist[0] = dist0;
    pdist[1] = dist1;

    return 0;
}

int StateDist2(vector<double> &state0, vector<double> &state1, double scale, double *pdist)
{
    double dist0 = sqrt((state0[0] * scale - state1[0] * scale) * (state0[0] * scale - state1[0] * scale) +
                        (state0[1] * scale - state1[1] * scale) * (state0[1] * scale - state1[1] * scale) +
                        (state0[2] * scale - state1[2] * scale) * (state0[2] * scale - state1[2] * scale));

    double dist1 = sqrt((state0[3] * scale - state1[3] * scale) * (state0[3] * scale - state1[3] * scale) +
                        (state0[4] * scale - state1[4] * scale) * (state0[4] * scale - state1[4] * scale) +
                        (state0[5] * scale - state1[5] * scale) * (state0[5] * scale - state1[5] * scale));

    pdist[0] = dist0;
    pdist[1] = dist1;

    return 0;
}

double vecMed(std::vector<double> vec)
{
    if (vec.empty())
        return 0;
    else
    {
        std::sort(vec.begin(), vec.end());
        if (vec.size() % 2 == 0)
            return (vec[vec.size() / 2 - 1] + vec[vec.size() / 2]) / 2;
        else
            return vec[vec.size() / 2];
    }
}



template <typename T>
std::vector<int> argsort(const std::vector<T> &array)
{
    const int array_len(array.size());
    std::vector<int> array_index(array_len, 0);
    for (int i = 0; i < array_len; ++i)
        array_index[i] = i;

    std::sort(array_index.begin(), array_index.end(),
              [&array](int pos1, int pos2) { return (array[pos1] > array[pos2]); });

    return array_index;
}

template <typename T>
std::vector<int> argsorts(const std::vector<T> &array)
{
    const int array_len(array.size());
    std::vector<int> array_index(array_len, 0);
    for (int i = 0; i < array_len; ++i)
        array_index[i] = i;

    std::sort(array_index.begin(), array_index.end(),
              [&array](int pos1, int pos2) { return (array[pos1] < array[pos2]); });

    return array_index;
}



void R6M(vector<vector<double>> paramsList, std::vector<std::vector<std::vector<double>>> &resList, unsigned int pt_start, unsigned  int pt_end, int threadId)
{

    double buffer[256] = {0};
    double ptRes[256] = {0};


    std::vector<std::vector<double>> res;
    for (unsigned int ii = pt_start; ii < pt_end && ii < paramsList.size(); ++ii)
    {
        buffer[0] = 0;
        for (unsigned int jj = 0; jj < paramsList[ii].size(); ++jj)
        {
            if (paramsList[ii].size() != 60)
            {
                break;
            }
            // printf("%d\n",ii);
            buffer[jj + 1] = paramsList[ii][jj];
        }
        // resList[ii].clear();
        int cnt = MinSolver(buffer, ptRes, threadId);
        resList[ii].resize(cnt);
        for (int jj = 0; jj < cnt; ++jj)
        {
            resList[ii][jj].resize(6);
            for (int kk = 0; kk < 6; ++kk)
            {
                resList[ii][jj][kk] = ptRes[jj * 6 + kk];
            }
        }
    }

    return;
}

cv::Mat skew(cv::Mat &vector)
{

    cv::Mat matrix = cv::Mat(3, 3, CV_64F, 0.0);

    matrix.at<double>(0, 1) = -vector.at<double>(2);
    matrix.at<double>(0, 2) = vector.at<double>(1);
    matrix.at<double>(1, 2) = -vector.at<double>(0);

    matrix.at<double>(1, 0) = vector.at<double>(2);
    matrix.at<double>(2, 0) = -vector.at<double>(1);
    matrix.at<double>(2, 1) = vector.at<double>(0);

    return matrix;
}

int Initializer::FindMotionMThread(const vector<vector<size_t>> &Sets)
{
    // Build task list
    vector<vector<double>> paramsList;
    paramsList.resize(Sets.size());
    for (unsigned int ii = 0; ii < Sets.size(); ++ii)
    {
        paramsList[ii].push_back(mR.at<double>(0, 0));
        paramsList[ii].push_back(mR.at<double>(1, 0));
        paramsList[ii].push_back(mR.at<double>(2, 0));

        paramsList[ii].push_back(mR.at<double>(0, 1));
        paramsList[ii].push_back(mR.at<double>(1, 1));
        paramsList[ii].push_back(mR.at<double>(2, 1));

        paramsList[ii].push_back(mR.at<double>(0, 2));
        paramsList[ii].push_back(mR.at<double>(1, 2));
        paramsList[ii].push_back(mR.at<double>(2, 2));

        paramsList[ii].push_back(mT.at<double>(0));
        paramsList[ii].push_back(mT.at<double>(1));
        paramsList[ii].push_back(mT.at<double>(2));

        for (unsigned int jj = 0; jj < Sets[ii].size(); ++jj)
        {
            int index = Sets[ii][jj];
            int leftIndex = mvMatches12[index].first;
            int rightIndex = mvMatches12[index].second;

            paramsList[ii].push_back(mvKeys1[leftIndex].pt.y);
            paramsList[ii].push_back(mvKeys2[rightIndex].pt.y);
            paramsList[ii].push_back(mvKeys1Norm[leftIndex].pt.x);
            paramsList[ii].push_back(mvKeys1Norm[leftIndex].pt.y);
            paramsList[ii].push_back(1.0);

            paramsList[ii].push_back(mvKeys2Norm[rightIndex].pt.x);
            paramsList[ii].push_back(mvKeys2Norm[rightIndex].pt.y);
            paramsList[ii].push_back(1.0);
        }
    }

    int taskCnt = paramsList.size();
    bool multiThread = true;
    int threadCnt = THREAD_CNT;
    std::vector<std::vector<std::vector<double>>> resList;
    resList.resize(taskCnt);
    int thread_step = int(ceil(taskCnt / threadCnt));
    PreProcess();
    if (multiThread)
    {
        std::vector<std::thread> thread_pool;
        for (int it = 0; it < threadCnt; ++it)
        {
            int pt_start = thread_step * it;
            int pt_end = thread_step * (it + 1);

            std::thread this_thread(R6M, paramsList, std::ref(resList), pt_start, pt_end, it);
            thread_pool.push_back(std::move(this_thread));
        }
        for (unsigned int it = 0; it < thread_pool.size(); ++it)
        {
            if (thread_pool[it].joinable())
                thread_pool[it].join();
        }
    }
    else
    {
        R6M(paramsList, resList, 0, taskCnt, 0);
    }

    // Number of putative matches
    const int N = mvMatches12.size();

    vector<bool> vbMatchesInliers;
    vbMatchesInliers.resize(N);

    const float th = 0.841; // 3.841
    const float thScore = 5.991;
    double sigma = 1.0;
    double factor = 1.2;
    float invSigmaSquare[8];
    invSigmaSquare[0] = 1.0 / (sigma * sigma);
    invSigmaSquare[1] = invSigmaSquare[0] * factor * factor;
    invSigmaSquare[2] = invSigmaSquare[1] * factor * factor;
    invSigmaSquare[3] = invSigmaSquare[2] * factor * factor;
    invSigmaSquare[4] = invSigmaSquare[3] * factor * factor; 
    invSigmaSquare[5] = invSigmaSquare[5] * factor * factor;
    invSigmaSquare[6] = invSigmaSquare[6] * factor * factor;
    invSigmaSquare[7] = invSigmaSquare[7] * factor * factor;
    int bestCnt = 0;
    double bestScore = 0;
    int bestIndex = -1;
    int bestKK = -1;

    // For debug
    std::vector<std::vector<int>> inlierCntList;
    std::vector<std::vector<double>> scoreCntList;
    std::vector<std::vector<std::vector<double>>> disList;

    inlierCntList.resize(resList.size());
    scoreCntList.resize(resList.size());
    disList.resize(resList.size());

    // transfer all informations to a vector
    std::vector<std::vector<double>> distVec;
    std::vector<int> inlierCntVec;
    std::vector<double> scoreCntVec;
    std::vector<std::vector<double>> stateVec;
    std::vector<double> dists;

    int SumN = 0;

    double scale = RowTimeG;
    bool IsCalc;
    for (unsigned int it = 0; it < resList.size(); it++)
    {
        inlierCntList[it].resize(resList[it].size(), 0);
        scoreCntList[it].resize(resList[it].size(), 0);
        disList[it].resize(resList[it].size());

        for (unsigned int kk = 0; kk < resList[it].size(); ++kk)
        {
            SumN += 1;

            const double w0 = resList[it][kk][0];
            const double w1 = resList[it][kk][1];
            const double w2 = resList[it][kk][2];
            const double t0 = resList[it][kk][3];
            const double t1 = resList[it][kk][4];
            const double t2 = resList[it][kk][5];

            cv::Mat W(3, 1, CV_64F);
            cv::Mat V(3, 1, CV_64F);

            W.at<double>(0) = w0;
            W.at<double>(1) = w1;
            W.at<double>(2) = w2;

            V.at<double>(0) = t0;
            V.at<double>(1) = t1;
            V.at<double>(2) = t2;

            // FILTER
            double vec = sqrt(t0 * t0 + t1 * t1 + t2 * t2);
            // t0 / scale > 0 || 
            if (vec * 3.6 / scale > 100 || fabs(w0) / scale > 3 || fabs(w1) / scale > 3 || fabs(w2) / scale > 3)
            {
                IsCalc = false;
                continue;
            }

            cv::Mat Rb = skew(W);

            float score = 0;
            int inlierCnt = 0;
            vector<bool> vbCurrentInliers(N, false);

            dists.clear();
            IsCalc = true;
            for (int i = 0; i < N; i++)
            {
                int leftIndex = mvMatches12[i].first;
                int rightIndex = mvMatches12[i].second;

                const double lrowidx = mvKeys1[leftIndex].pt.y;
                const double rrowidx = mvKeys2[rightIndex].pt.y;
                int level = mvKeys1[leftIndex].octave;

                const cv::Point2f p1 = mvKeys1Norm[leftIndex].pt;
                const cv::Point2f p2 = mvKeys2Norm[rightIndex].pt;

                // Construct Essential matrix
                cv::Mat Ri = mR * (cv::Mat::eye(3, 3, CV_64F) + (rrowidx - lrowidx) * Rb);
                cv::Mat Ti = mT + rrowidx * mR * V - lrowidx * Ri * V;
                cv::Mat E = skew(Ti) * Ri;
                cv::Mat F = mK2.t().inv() * E * mK.inv();

                const double u1 = mK.at<double>(0, 0) * p1.x + mK.at<double>(0, 2);
                const double v1 = mK.at<double>(1, 1) * p1.y + mK.at<double>(1, 2);
                const double u2 = mK2.at<double>(0, 0) * p2.x + mK2.at<double>(0, 2);
                const double v2 = mK2.at<double>(1, 1) * p2.y + mK2.at<double>(1, 2);

                const double a2 = F.at<double>(0, 0) * u1 + F.at<double>(0, 1) * v1 + F.at<double>(0, 2);
                const double b2 = F.at<double>(1, 0) * u1 + F.at<double>(1, 1) * v1 + F.at<double>(1, 2);
                const double c2 = F.at<double>(2, 0) * u1 + F.at<double>(2, 1) * v1 + F.at<double>(2, 2);

                const double num2 = a2 * u2 + b2 * v2 + c2;
                const double squareDist1 = num2 * num2 / (a2 * a2 + b2 * b2);
                const double chiSquare1 = squareDist1 * invSigmaSquare[level];

                disList[it][kk].push_back(squareDist1);
                dists.push_back(squareDist1);

                if (chiSquare1 < th)
                {
                    inlierCnt += 1;
                    score += thScore - chiSquare1; // thscore is balence to
                    vbCurrentInliers[i] = true;
                }
                else
                {
                    vbCurrentInliers[i] = false;
                }
            }

            scoreCntList[it][kk] = score;
            inlierCntList[it][kk] = inlierCnt;

            if (IsCalc)
            {
                stateVec.push_back(resList[it][kk]);
                scoreCntVec.push_back(score);
                inlierCntVec.push_back(inlierCnt);
                distVec.push_back(dists);
            }

            if (bestScore < score)
            {
                vbMatchesInliers = vbCurrentInliers;
                bestScore = score;
                bestIndex = it;
                bestCnt = inlierCnt;
                bestKK = kk;
            }
        }
    }

    if (bestIndex != -1)
    {
        std::cout << "BestIt:" << bestIndex << " BestScore:" << bestScore << " BestCnt:" << bestCnt << " Ratio:" << float(bestCnt) / N << " TotalMatches:" << N << std::endl;
        std::cout << "CloseForm Solution:" <<  resList[bestIndex][bestKK][0] / scale << " " << resList[bestIndex][bestKK][1] / scale << " " << resList[bestIndex][bestKK][2] / scale
                  << " " << resList[bestIndex][bestKK][3] / scale << " " << resList[bestIndex][bestKK][4] / scale << " "
                  << resList[bestIndex][bestKK][5] / scale << std::endl;
    }
    else
    {
        printf("No solutions satisfy conditions!\n");
    }

    // Sort scores
    vector<int> sortedIndexs = argsort(scoreCntVec);

    mCloseFormInliers.clear();
    mCloseFormStates.clear();
    unsigned int maxCandidate = 20;
    for (unsigned int iid = 0; iid < sortedIndexs.size() && iid < maxCandidate; ++iid){
        mCloseFormStates.push_back(stateVec[sortedIndexs[iid]]);
        mCloseFormStates[iid][0] = mCloseFormStates[iid][0] / scale;
        mCloseFormStates[iid][1] = mCloseFormStates[iid][1] / scale;
        mCloseFormStates[iid][2] = mCloseFormStates[iid][2] / scale;
        mCloseFormStates[iid][3] = mCloseFormStates[iid][3] / scale;
        mCloseFormStates[iid][4] = mCloseFormStates[iid][4] / scale;
        mCloseFormStates[iid][5] = mCloseFormStates[iid][5] / scale;
    }

    return 0;
}

void Initializer::UpdateSolutionsFromCloseForm(vector<vector<int>> finalInliers, vector<vector<double>> finalStateGroup){
    mCloseFormInliers.clear();
    mCloseFormStates.clear();
    for (unsigned int iid = 0; iid < finalInliers.size(); ++iid){
        mCloseFormInliers.push_back(finalInliers[iid].size());
        mCloseFormStates.push_back(finalStateGroup[iid]);
    }
}

void Initializer::UpdateSolutionsFromProjError(vector<vector<int>> finalInliers, vector<vector<double>> finalStateGroup){
    mProjInliers.clear();
    mProjStates.clear();
    for (unsigned int iid = 0; iid < finalInliers.size(); ++iid){
        mProjInliers.push_back(finalInliers[iid].size());
        mProjStates.push_back(finalStateGroup[iid]);
    }
}

vector<vector<double>> Initializer::GetProjRes(){
    return mProjStates;
}

vector<vector<double>> Initializer::GetCloseFormRes(){
    return mCloseFormStates;
}

vector<string> ssplit(const string &s, const string &seperator)
{
    vector<string> result;
    typedef string::size_type string_size;
    string_size i = 0;

    while (i != s.size())
    {
        int flag = 0;
        while (i != s.size() && flag == 0)
        {
            flag = 1;
            for (string_size x = 0; x < seperator.size(); ++x)
                if (s[i] == seperator[x])
                {
                    ++i;
                    flag = 0;
                    break;
                }
        }

        flag = 0;
        string_size j = i;
        while (j != s.size() && flag == 0)
        {
            for (string_size x = 0; x < seperator.size(); ++x)
                if (s[j] == seperator[x])
                {
                    flag = 1;
                    break;
                }
            if (flag == 0)
                ++j;
        }
        if (i != j)
        {
            result.push_back(s.substr(i, j - i));
            i = j;
        }
    }
    return result;
}

int LoadMatchs(std::string matchPath, vector<double> &LeftIndexs, vector<double> &RightIndexs, vector<cv::Point2f> &LeftPts, vector<cv::Point2f> &RightPts)
{
    FILE *fp = fopen(matchPath.c_str(), "r");
    if (fp == NULL)
    {
        printf("Open %s error!\n", matchPath.c_str());
        return -1;
    }

    LeftIndexs.clear();
    RightIndexs.clear();
    LeftPts.clear();
    RightPts.clear();

    cv::Point2f pt0, pt1;
    char buffer[1024] = {0};
    while (fgets(buffer, 1024, fp))
    {
        buffer[strlen(buffer) - 1] = '\0';
        std::string tStr = std::string(buffer);
        std::vector<std::string> strVector = ssplit(tStr, " ");

        if (strVector.size() != 6)
        {
            printf("split error!!\n");
            break;
        }

        int lrowidx = atof(strVector[0].c_str());
        int rrowidx = atof(strVector[1].c_str());

        LeftIndexs.push_back(lrowidx);
        RightIndexs.push_back(rrowidx);

        pt0.x = atof(strVector[2].c_str());
        pt0.y = atof(strVector[3].c_str());

        pt1.x = atof(strVector[4].c_str());
        pt1.y = atof(strVector[5].c_str());

        LeftPts.push_back(pt0);
        RightPts.push_back(pt1);
    }
    fclose(fp);
    return 0;
}

int LoadResParam(std::string paramPath, vector<vector<double>> &paramsList)
{
    FILE *fp = fopen(paramPath.c_str(), "r");
    if (fp == NULL)
    {
        printf("Open %s error!\n", paramPath.c_str());
        return -1;
    }

    paramsList.clear();

    vector<double> params;
    char buffer[1024] = {0};
    while (fgets(buffer, 1024, fp))
    {
        buffer[strlen(buffer) - 1] = '\0';
        std::string tStr = std::string(buffer);
        std::vector<std::string> strVector = ssplit(tStr, " ");

        if (strVector.size() != 7)
        {
            printf("split error!!\n");
            break;
        }

        params.clear();
        params.push_back(atof(strVector[0].c_str()));
        params.push_back(atof(strVector[1].c_str()));
        params.push_back(atof(strVector[2].c_str()));

        params.push_back(atof(strVector[3].c_str()));
        params.push_back(atof(strVector[4].c_str()));
        params.push_back(atof(strVector[5].c_str()));

        params.push_back(atof(strVector[6].c_str()));

        paramsList.push_back(params);
    }
    fclose(fp);
    return 0;
}

int LoadParam(std::string paramPath, vector<vector<double>> &paramsList)
{
    FILE *fp = fopen(paramPath.c_str(), "r");
    if (fp == NULL)
    {
        printf("Open %s error!\n", paramPath.c_str());
        return -1;
    }

    paramsList.clear();

    vector<double> params;
    char buffer[1024] = {0};
    while (fgets(buffer, 1024, fp))
    {
        buffer[strlen(buffer) - 1] = '\0';
        std::string tStr = std::string(buffer);
        std::vector<std::string> strVector = ssplit(tStr, " ");

        if (strVector.size() != 60)
        {
            printf("split error!!\n");
            break;
        }

        params.clear();
        for (int ii = 0; ii < 60; ++ii)
        {
            params.push_back(atof(strVector[ii].c_str()));
        }

        paramsList.push_back(params);
    }
    fclose(fp);
    return 0;
}



void Initializer::CheckEssential(std::string resPath, std::string matchPath, cv::Mat R, cv::Mat T, cv::Mat K0, cv::Mat K1)
{
    // Load calculated results and matches
    vector<double> LeftIndexs;
    vector<double> RightIndexs;
    vector<cv::Point2f> LeftPts;
    vector<cv::Point2f> RightPts;
    vector<vector<double>> paramsList;

    LoadMatchs(matchPath, LeftIndexs, RightIndexs, LeftPts, RightPts);
    LoadResParam(resPath, paramsList);

    // Number of putative matches
    const int N = LeftIndexs.size();

    vector<bool> vbMatchesInliers;
    vbMatchesInliers.resize(N);

    const float th = 0.841; // 3.841
    const float thScore = 5.991;
    double sigma = 1.0;
    const float invSigmaSquare = 1.0 / (sigma * sigma);
    vector<double> scoreList;
    vector<int> inlierCntList;
    int bestCnt = 0;
    double bestScore = 0;
    int bestIndex = 0;

    for (unsigned int it = 0; it < paramsList.size(); it++)
    {
        const double w0 = paramsList[it][0];
        const double w1 = paramsList[it][1];
        const double w2 = paramsList[it][2];
        const double t0 = paramsList[it][3];
        const double t1 = paramsList[it][4];
        const double t2 = paramsList[it][5];

        double vec = sqrt(t0 * t0 + t1 * t1 + t2 * t2);
        double scale = RowTimeG;
        if (t0 / scale < 0 || vec * 3.6 / scale > 100 || fabs(w0) / scale > 1.5 || fabs(w1) / scale > 2.4 || fabs(w2) / scale > 1.5)
        {
            continue;
        }

        cv::Mat W(3, 1, CV_64F);
        cv::Mat V(3, 1, CV_64F);

        W.at<double>(0) = w0;
        W.at<double>(1) = w1;
        W.at<double>(2) = w2;

        V.at<double>(0) = t0;
        V.at<double>(1) = t1;
        V.at<double>(2) = t2;

        cv::Mat Rb = skew(W);

        // cout << Rb << endl;

        float score = 0;
        int inlierCnt = 0;
        vector<bool> vbCurrentInliers(N, false);

        for (int i = 0; i < N; i++)
        {
            const double lrowidx = LeftIndexs[i];
            const double rrowidx = RightIndexs[i];
            const cv::Point2f p1 = LeftPts[i];
            const cv::Point2f p2 = RightPts[i];

            // Construct Essential matrix
            cv::Mat Ri = R * (cv::Mat::eye(3, 3, CV_64F) + (rrowidx - lrowidx) * Rb);
            cv::Mat Ti = T + rrowidx * R * V - lrowidx * Ri * V;
            cv::Mat E = skew(Ti) * Ri;
            cv::Mat F = K1.t().inv() * E * K0.inv();

            const double u1 = K0.at<double>(0, 0) * p1.x + K0.at<double>(0, 2);
            const double v1 = K0.at<double>(1, 1) * p1.y + K0.at<double>(1, 2);
            const double u2 = K1.at<double>(0, 0) * p2.x + K1.at<double>(0, 2);
            const double v2 = K1.at<double>(1, 1) * p2.y + K1.at<double>(1, 2);

            const double a2 = F.at<double>(0, 0) * u1 + F.at<double>(0, 1) * v1 + F.at<double>(0, 2);
            const double b2 = F.at<double>(1, 0) * u1 + F.at<double>(1, 1) * v1 + F.at<double>(1, 2);
            const double c2 = F.at<double>(2, 0) * u1 + F.at<double>(2, 1) * v1 + F.at<double>(2, 2);

            const double num2 = a2 * u2 + b2 * v2 + c2;

            const double squareDist1 = num2 * num2 / (a2 * a2 + b2 * b2);

            const double chiSquare1 = squareDist1 * invSigmaSquare;

            if (chiSquare1 < th)
            {
                inlierCnt += 1;
                score += thScore - chiSquare1;
                vbCurrentInliers[i] = true;
            }
            else
            {
                vbCurrentInliers[i] = false;
            }
        }

        scoreList.push_back(score);
        inlierCntList.push_back(inlierCnt);

        if (bestScore < score)
        {
            vbMatchesInliers = vbCurrentInliers;
            bestScore = score;
            bestIndex = it;
            bestCnt = inlierCnt;
        }

    }

    std::cout << "BestIt:" << bestIndex << " BestScore:" << bestScore << " BestCnt:" << bestCnt << std::endl;
    std::cout << "Total Matches:" << N << std::endl;
}

int Initializer::TestR6M(std::string path, std::string matchPath, cv::Mat R, cv::Mat T, cv::Mat K0, cv::Mat K1)
{

    vector<vector<double>> paramsList;
    LoadParam(path, paramsList);

    int taskCnt = paramsList.size();
    bool multiThread = false;
    int threadCnt = THREAD_CNT;
    std::vector<std::vector<std::vector<double>>> resList;
    taskCnt = 200;
    resList.resize(taskCnt);
    int thread_step = int(ceil(taskCnt / threadCnt));
    PreProcess();
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    if (multiThread)
    {
        std::vector<std::thread> thread_pool;
        for (int it = 0; it < threadCnt; ++it)
        {
            int pt_start = thread_step * it;
            int pt_end = thread_step * (it + 1) - 1;

            std::thread this_thread(R6M, paramsList, std::ref(resList), pt_start, pt_end, it);
            thread_pool.push_back(std::move(this_thread));
        }
        for (unsigned int it = 0; it < thread_pool.size(); ++it)
        {
            if (thread_pool[it].joinable())
                thread_pool[it].join();
        }
    }
    else
    {
        R6M(paramsList, resList, 0, taskCnt, 0);
    }
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    printf("%f\n", ttrack);

    // Load calculated results and matches
    vector<double> LeftIndexs;
    vector<double> RightIndexs;
    vector<cv::Point2f> LeftPts;
    vector<cv::Point2f> RightPts;
    // vector<vector<double>> paramsList;

    LoadMatchs(matchPath, LeftIndexs, RightIndexs, LeftPts, RightPts);

    // Number of putative matches
    const int N = LeftIndexs.size();

    vector<bool> vbMatchesInliers;
    vbMatchesInliers.resize(N);

    const float th = 0.841; // 3.841
    const float thScore = 5.991;
    double sigma = 1.0;
    const float invSigmaSquare = 1.0 / (sigma * sigma);
    vector<double> scoreList;
    vector<int> inlierCntList;
    int bestCnt = 0;
    double bestScore = 0;
    int bestIndex = 0;
    int bestKK = 0;

    for (unsigned int it = 0; it < resList.size(); it++)
    {
        for (unsigned int kk = 0; kk < resList[it].size(); ++kk)
        {
            const double w0 = resList[it][kk][0];
            const double w1 = resList[it][kk][1];
            const double w2 = resList[it][kk][2];
            const double t0 = resList[it][kk][3];
            const double t1 = resList[it][kk][4];
            const double t2 = resList[it][kk][5];

            double vec = sqrt(t0 * t0 + t1 * t1 + t2 * t2);
            double scale = RowTimeG;
            if (t0 / scale < 0 || vec * 3.6 / scale > 100 || fabs(w0) / scale > 1.5 || fabs(w1) / scale > 2.4 || fabs(w2) / scale > 1.5)
            {
                continue;
            }

            cv::Mat W(3, 1, CV_64F);
            cv::Mat V(3, 1, CV_64F);

            W.at<double>(0) = w0;
            W.at<double>(1) = w1;
            W.at<double>(2) = w2;

            V.at<double>(0) = t0;
            V.at<double>(1) = t1;
            V.at<double>(2) = t2;

            cv::Mat Rb = skew(W);

            // cout << Rb << endl;

            float score = 0;
            int inlierCnt = 0;
            vector<bool> vbCurrentInliers(N, false);

            for (int i = 0; i < N; i++)
            {
                const double lrowidx = LeftIndexs[i];
                const double rrowidx = RightIndexs[i];
                const cv::Point2f p1 = LeftPts[i];
                const cv::Point2f p2 = RightPts[i];

                // Construct Essential matrix
                cv::Mat Ri = R * (cv::Mat::eye(3, 3, CV_64F) + (rrowidx - lrowidx) * Rb);
                cv::Mat Ti = T + rrowidx * R * V - lrowidx * Ri * V;
                cv::Mat E = skew(Ti) * Ri;
                cv::Mat F = K1.t().inv() * E * K0.inv();

                const double u1 = K0.at<double>(0, 0) * p1.x + K0.at<double>(0, 2);
                const double v1 = K0.at<double>(1, 1) * p1.y + K0.at<double>(1, 2);
                const double u2 = K1.at<double>(0, 0) * p2.x + K1.at<double>(0, 2);
                const double v2 = K1.at<double>(1, 1) * p2.y + K1.at<double>(1, 2);

                // Reprojection error in second image
                // l2=F21x1=(a2,b2,c2)

                const double a2 = F.at<double>(0, 0) * u1 + F.at<double>(0, 1) * v1 + F.at<double>(0, 2);
                const double b2 = F.at<double>(1, 0) * u1 + F.at<double>(1, 1) * v1 + F.at<double>(1, 2);
                const double c2 = F.at<double>(2, 0) * u1 + F.at<double>(2, 1) * v1 + F.at<double>(2, 2);

                const double num2 = a2 * u2 + b2 * v2 + c2;

                const double squareDist1 = num2 * num2 / (a2 * a2 + b2 * b2);

                const double chiSquare1 = squareDist1 * invSigmaSquare;

                if (chiSquare1 < th)
                {
                    inlierCnt += 1;
                    score += thScore - chiSquare1;
                    vbCurrentInliers[i] = true;
                }
                else
                {
                    vbCurrentInliers[i] = false;
                }
            }

            scoreList.push_back(score);
            inlierCntList.push_back(inlierCnt);

            if (bestScore < score)
            {
                vbMatchesInliers = vbCurrentInliers;
                bestScore = score;
                bestIndex = it;
                bestCnt = inlierCnt;
                bestKK = kk;
            }
        }
    }

    std::cout << "BestIt:" << bestIndex << " BestScore:" << bestScore << " BestCnt:" << bestCnt << std::endl;
    std::cout << "Total Matches:" << N << std::endl;
    double scale = RowTimeG;
    std::cout << resList[bestIndex][bestKK][0] / scale << " " << resList[bestIndex][bestKK][1] / scale << " " << resList[bestIndex][bestKK][2] / scale
              << " " << resList[bestIndex][bestKK][3] / scale << " " << resList[bestIndex][bestKK][4] / scale << " "
              << resList[bestIndex][bestKK][5] / scale << std::endl;

    return 0;
}


void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
{
    // Number of putative matches
    const int N = mvMatches12.size();

    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1, vPn1, T1);
    Normalize(mvKeys2, vPn2, T2);
    cv::Mat T2inv = T2.inv();

    score = 0.0;
    vbMatchesInliers = vector<bool>(N, false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat H21i, H12i;
    vector<bool> vbCurrentInliers(N, false);
    float currentScore;

    for (int it = 0; it < mMaxIterations; it++)
    {
        // Select a minimum set
        for (size_t j = 0; j < 8; j++)
        {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        cv::Mat Hn = ComputeH21(vPn1i, vPn2i);
        H21i = T2inv * Hn * T1;
        H12i = H21i.inv();

        currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);

        if (currentScore > score)
        {
            H21 = H21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    const int N = vbMatchesInliers.size();

    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1, vPn1, T1);
    Normalize(mvKeys2, vPn2, T2);
    cv::Mat T2t = T2.t();

    score = 0.0;
    vbMatchesInliers = vector<bool>(N, false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N, false);
    float currentScore;

    for (int it = 0; it < mMaxIterations; it++)
    {
        for (int j = 0; j < 8; j++)
        {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        cv::Mat Fn = ComputeF21(vPn1i, vPn2i);

        F21i = T2t * Fn * T1;

        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

        if (currentScore > score)
        {
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(2 * N, 9, CV_32F);

    for (int i = 0; i < N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2 * i, 0) = 0.0;
        A.at<float>(2 * i, 1) = 0.0;
        A.at<float>(2 * i, 2) = 0.0;
        A.at<float>(2 * i, 3) = -u1;
        A.at<float>(2 * i, 4) = -v1;
        A.at<float>(2 * i, 5) = -1;
        A.at<float>(2 * i, 6) = v2 * u1;
        A.at<float>(2 * i, 7) = v2 * v1;
        A.at<float>(2 * i, 8) = v2;

        A.at<float>(2 * i + 1, 0) = u1;
        A.at<float>(2 * i + 1, 1) = v1;
        A.at<float>(2 * i + 1, 2) = 1;
        A.at<float>(2 * i + 1, 3) = 0.0;
        A.at<float>(2 * i + 1, 4) = 0.0;
        A.at<float>(2 * i + 1, 5) = 0.0;
        A.at<float>(2 * i + 1, 6) = -u2 * u1;
        A.at<float>(2 * i + 1, 7) = -u2 * v1;
        A.at<float>(2 * i + 1, 8) = -u2;
    }

    cv::Mat u, w, vt;

    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    return vt.row(8).reshape(0, 3);
}

cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(N, 9, CV_32F);

    for (int i = 0; i < N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i, 0) = u2 * u1;
        A.at<float>(i, 1) = u2 * v1;
        A.at<float>(i, 2) = u2;
        A.at<float>(i, 3) = v2 * u1;
        A.at<float>(i, 4) = v2 * v1;
        A.at<float>(i, 5) = v2;
        A.at<float>(i, 6) = u1;
        A.at<float>(i, 7) = v1;
        A.at<float>(i, 8) = 1;
    }

    cv::Mat u, w, vt;

    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    cv::SVDecomp(Fpre, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    w.at<float>(2) = 0;

    return u * cv::Mat::diag(w) * vt;
}

float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = mvMatches12.size();

    const float h11 = H21.at<float>(0, 0);
    const float h12 = H21.at<float>(0, 1);
    const float h13 = H21.at<float>(0, 2);
    const float h21 = H21.at<float>(1, 0);
    const float h22 = H21.at<float>(1, 1);
    const float h23 = H21.at<float>(1, 2);
    const float h31 = H21.at<float>(2, 0);
    const float h32 = H21.at<float>(2, 1);
    const float h33 = H21.at<float>(2, 2);

    const float h11inv = H12.at<float>(0, 0);
    const float h12inv = H12.at<float>(0, 1);
    const float h13inv = H12.at<float>(0, 2);
    const float h21inv = H12.at<float>(1, 0);
    const float h22inv = H12.at<float>(1, 1);
    const float h23inv = H12.at<float>(1, 2);
    const float h31inv = H12.at<float>(2, 0);
    const float h32inv = H12.at<float>(2, 1);
    const float h33inv = H12.at<float>(2, 2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 5.991;

    const float invSigmaSquare = 1.0 / (sigma * sigma);

    for (int i = 0; i < N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        const float w2in1inv = 1.0 / (h31inv * u2 + h32inv * v2 + h33inv);
        const float u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv;
        const float v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv;

        const float squareDist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1);

        const float chiSquare1 = squareDist1 * invSigmaSquare;

        if (chiSquare1 > th)
            bIn = false;
        else
            score += th - chiSquare1;

        const float w1in2inv = 1.0 / (h31 * u1 + h32 * v1 + h33);
        const float u1in2 = (h11 * u1 + h12 * v1 + h13) * w1in2inv;
        const float v1in2 = (h21 * u1 + h22 * v1 + h23) * w1in2inv;

        const float squareDist2 = (u2 - u1in2) * (u2 - u1in2) + (v2 - v1in2) * (v2 - v1in2);

        const float chiSquare2 = squareDist2 * invSigmaSquare;

        if (chiSquare2 > th)
            bIn = false;
        else
            score += th - chiSquare2;

        if (bIn)
            vbMatchesInliers[i] = true;
        else
            vbMatchesInliers[i] = false;
    }

    return score;
}

float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = mvMatches12.size();

    const float f11 = F21.at<float>(0, 0);
    const float f12 = F21.at<float>(0, 1);
    const float f13 = F21.at<float>(0, 2);
    const float f21 = F21.at<float>(1, 0);
    const float f22 = F21.at<float>(1, 1);
    const float f23 = F21.at<float>(1, 2);
    const float f31 = F21.at<float>(2, 0);
    const float f32 = F21.at<float>(2, 1);
    const float f33 = F21.at<float>(2, 2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 3.841;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0 / (sigma * sigma);

    for (int i = 0; i < N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        const float a2 = f11 * u1 + f12 * v1 + f13;
        const float b2 = f21 * u1 + f22 * v1 + f23;
        const float c2 = f31 * u1 + f32 * v1 + f33;

        const float num2 = a2 * u2 + b2 * v2 + c2;

        const float squareDist1 = num2 * num2 / (a2 * a2 + b2 * b2);

        const float chiSquare1 = squareDist1 * invSigmaSquare;

        if (chiSquare1 > th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        const float a1 = f11 * u2 + f21 * v2 + f31;
        const float b1 = f12 * u2 + f22 * v2 + f32;
        const float c1 = f13 * u2 + f23 * v2 + f33;

        const float num1 = a1 * u1 + b1 * v1 + c1;

        const float squareDist2 = num1 * num1 / (a1 * a1 + b1 * b1);

        const float chiSquare2 = squareDist2 * invSigmaSquare;

        if (chiSquare2 > th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if (bIn)
            vbMatchesInliers[i] = true;
        else
            vbMatchesInliers[i] = false;
    }

    return score;
}

bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                               cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N = 0;
    for (size_t i = 0, iend = vbMatchesInliers.size(); i < iend; i++)
        if (vbMatchesInliers[i])
            N++;

    cv::Mat E21 = K.t() * F21 * K;

    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses
    DecomposeE(E21, R1, R2, t);

    cv::Mat t1 = t;
    cv::Mat t2 = -t;

    // Reconstruct with the 4 hyphoteses and check
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<bool> vbTriangulated1, vbTriangulated2, vbTriangulated3, vbTriangulated4;
    float parallax1, parallax2, parallax3, parallax4;

    int nGood1 = CheckRT(R1, t1, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3D1, 4.0 * mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2, t1, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3D2, 4.0 * mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1, t2, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3D3, 4.0 * mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2, t2, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3D4, 4.0 * mSigma2, vbTriangulated4, parallax4);

    int maxGood = max(nGood1, max(nGood2, max(nGood3, nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();

    int nMinGood = max(static_cast<int>(0.9 * N), minTriangulated);

    int nsimilar = 0;
    if (nGood1 > 0.7 * maxGood)
        nsimilar++;
    if (nGood2 > 0.7 * maxGood)
        nsimilar++;
    if (nGood3 > 0.7 * maxGood)
        nsimilar++;
    if (nGood4 > 0.7 * maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    if (maxGood < nMinGood || nsimilar > 1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    if (maxGood == nGood1)
    {
        if (parallax1 > minParallax)
        {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }
    else if (maxGood == nGood2)
    {
        if (parallax2 > minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }
    else if (maxGood == nGood3)
    {
        if (parallax3 > minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }
    else if (maxGood == nGood4)
    {
        if (parallax4 > minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}

bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                               cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N = 0;
    for (size_t i = 0, iend = vbMatchesInliers.size(); i < iend; i++)
        if (vbMatchesInliers[i])
            N++;

    cv::Mat invK = K.inv();
    cv::Mat A = invK * H21 * K;

    cv::Mat U, w, Vt, V;
    cv::SVD::compute(A, w, U, Vt, cv::SVD::FULL_UV);
    V = Vt.t();

    float s = cv::determinant(U) * cv::determinant(Vt);

    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);

    if (d1 / d2 < 1.00001 || d2 / d3 < 1.00001)
    {
        return false;
    }

    vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    float aux1 = sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3));
    float aux3 = sqrt((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3));
    float x1[] = {aux1, aux1, -aux1, -aux1};
    float x3[] = {aux3, -aux3, aux3, -aux3};

    //case d'=d2
    float aux_stheta = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 + d3) * d2);

    float ctheta = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    for (int i = 0; i < 4; i++)
    {
        cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
        Rp.at<float>(0, 0) = ctheta;
        Rp.at<float>(0, 2) = -stheta[i];
        Rp.at<float>(2, 0) = stheta[i];
        Rp.at<float>(2, 2) = ctheta;

        cv::Mat R = s * U * Rp * Vt;
        vR.push_back(R);

        cv::Mat tp(3, 1, CV_32F);
        tp.at<float>(0) = x1[i];
        tp.at<float>(1) = 0;
        tp.at<float>(2) = -x3[i];
        tp *= d1 - d3;

        cv::Mat t = U * tp;
        vt.push_back(t / cv::norm(t));

        cv::Mat np(3, 1, CV_32F);
        np.at<float>(0) = x1[i];
        np.at<float>(1) = 0;
        np.at<float>(2) = x3[i];

        cv::Mat n = V * np;
        if (n.at<float>(2) < 0)
            n = -n;
        vn.push_back(n);
    }

    //case d'=-d2
    float aux_sphi = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 - d3) * d2);

    float cphi = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    for (int i = 0; i < 4; i++)
    {
        cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
        Rp.at<float>(0, 0) = cphi;
        Rp.at<float>(0, 2) = sphi[i];
        Rp.at<float>(1, 1) = -1;
        Rp.at<float>(2, 0) = sphi[i];
        Rp.at<float>(2, 2) = -cphi;

        cv::Mat R = s * U * Rp * Vt;
        vR.push_back(R);

        cv::Mat tp(3, 1, CV_32F);
        tp.at<float>(0) = x1[i];
        tp.at<float>(1) = 0;
        tp.at<float>(2) = x3[i];
        tp *= d1 + d3;

        cv::Mat t = U * tp;
        vt.push_back(t / cv::norm(t));

        cv::Mat np(3, 1, CV_32F);
        np.at<float>(0) = x1[i];
        np.at<float>(1) = 0;
        np.at<float>(2) = x3[i];

        cv::Mat n = V * np;
        if (n.at<float>(2) < 0)
            n = -n;
        vn.push_back(n);
    }

    int bestGood = 0;
    int secondBestGood = 0;
    int bestSolutionIdx = -1;
    float bestParallax = -1;
    vector<cv::Point3f> bestP3D;
    vector<bool> bestTriangulated;

    for (size_t i = 0; i < 8; i++)
    {
        float parallaxi;
        vector<cv::Point3f> vP3Di;
        vector<bool> vbTriangulatedi;
        int nGood = CheckRT(vR[i], vt[i], mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3Di, 4.0 * mSigma2, vbTriangulatedi, parallaxi);

        if (nGood > bestGood)
        {
            secondBestGood = bestGood;
            bestGood = nGood;
            bestSolutionIdx = i;
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        else if (nGood > secondBestGood)
        {
            secondBestGood = nGood;
        }
    }

    if (secondBestGood < 0.75 * bestGood && bestParallax >= minParallax && bestGood > minTriangulated && bestGood > 0.9 * N)
    {
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        vP3D = bestP3D;
        vbTriangulated = bestTriangulated;

        return true;
    }

    return false;
}

void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4, 4, CV_32F);

    A.row(0) = kp1.pt.x * P1.row(2) - P1.row(0);
    A.row(1) = kp1.pt.y * P1.row(2) - P1.row(1);
    A.row(2) = kp2.pt.x * P2.row(2) - P2.row(0);
    A.row(3) = kp2.pt.y * P2.row(2) - P2.row(1);

    cv::Mat u, w, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
}

void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    for (int i = 0; i < N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    meanX = meanX / N;
    meanY = meanY / N;

    float meanDevX = 0;
    float meanDevY = 0;

    for (int i = 0; i < N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    meanDevX = meanDevX / N;
    meanDevY = meanDevY / N;

    float sX = 1.0 / meanDevX;
    float sY = 1.0 / meanDevY;

    for (int i = 0; i < N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    T = cv::Mat::eye(3, 3, CV_32F);
    T.at<float>(0, 0) = sX;
    T.at<float>(1, 1) = sY;
    T.at<float>(0, 2) = -meanX * sX;
    T.at<float>(1, 2) = -meanY * sY;
}

int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                         const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                         const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    // Calibration parameters
    const float fx = K.at<float>(0, 0);
    const float fy = K.at<float>(1, 1);
    const float cx = K.at<float>(0, 2);
    const float cy = K.at<float>(1, 2);

    vbGood = vector<bool>(vKeys1.size(), false);
    vP3D.resize(vKeys1.size());

    vector<float> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K[I|0]
    cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
    K.copyTo(P1.rowRange(0, 3).colRange(0, 3));

    cv::Mat O1 = cv::Mat::zeros(3, 1, CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    cv::Mat P2(3, 4, CV_32F);
    R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
    t.copyTo(P2.rowRange(0, 3).col(3));
    P2 = K * P2;

    cv::Mat O2 = -R.t() * t;

    int nGood = 0;

    for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
    {
        if (!vbMatchesInliers[i])
            continue;

        const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
        const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
        cv::Mat p3dC1;

        Triangulate(kp1, kp2, P1, P2, p3dC1);

        if (!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[vMatches12[i].first] = false;
            continue;
        }

        // Check parallax
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);

        float cosParallax = normal1.dot(normal2) / (dist1 * dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if (p3dC1.at<float>(2) <= 0 && cosParallax < 0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R * p3dC1 + t;

        if (p3dC2.at<float>(2) <= 0 && cosParallax < 0.99998)
            continue;

        // Check reprojection error in first image
        float im1x, im1y;
        float invZ1 = 1.0 / p3dC1.at<float>(2);
        im1x = fx * p3dC1.at<float>(0) * invZ1 + cx;
        im1y = fy * p3dC1.at<float>(1) * invZ1 + cy;

        float squareError1 = (im1x - kp1.pt.x) * (im1x - kp1.pt.x) + (im1y - kp1.pt.y) * (im1y - kp1.pt.y);

        if (squareError1 > th2)
            continue;

        // Check reprojection error in second image
        float im2x, im2y;
        float invZ2 = 1.0 / p3dC2.at<float>(2);
        im2x = fx * p3dC2.at<float>(0) * invZ2 + cx;
        im2y = fy * p3dC2.at<float>(1) * invZ2 + cy;

        float squareError2 = (im2x - kp2.pt.x) * (im2x - kp2.pt.x) + (im2y - kp2.pt.y) * (im2y - kp2.pt.y);

        if (squareError2 > th2)
            continue;

        vCosParallax.push_back(cosParallax);
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
        nGood++;

        if (cosParallax < 0.99998)
            vbGood[vMatches12[i].first] = true;
    }

    if (nGood > 0)
    {
        sort(vCosParallax.begin(), vCosParallax.end());

        size_t idx = min(50, int(vCosParallax.size() - 1));
        parallax = acos(vCosParallax[idx]) * 180 / CV_PI;
    }
    else
        parallax = 0;

    return nGood;
}

void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    cv::Mat u, w, vt;
    cv::SVD::compute(E, w, u, vt);

    u.col(2).copyTo(t);
    t = t / cv::norm(t);

    cv::Mat W(3, 3, CV_32F, cv::Scalar(0));
    W.at<float>(0, 1) = -1;
    W.at<float>(1, 0) = 1;
    W.at<float>(2, 2) = 1;

    R1 = u * W * vt;
    if (cv::determinant(R1) < 0)
        R1 = -R1;

    R2 = u * W.t() * vt;
    if (cv::determinant(R2) < 0)
        R2 = -R2;
}

} 
