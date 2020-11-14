#include "CoarseInitializer.h"
#include "PixelSelector2.h"
#include "PixelSelector.h"
#include "../util/nanoflann.h"
#include "../util/globalFuncs.h"
#include "../util/globalCalib.h"
#include "../util/ImageDisplay.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <thread>  

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

using namespace cv;

namespace SFRSS
{

FramePym* leftFrameG = NULL;
FramePym* rightFrameG = NULL;
#define MAX_THREAD 8
Accumulator9 acc9MT[MAX_THREAD];
Accumulator9 acc9SCMT[MAX_THREAD];
Vec10f* JbBuffer;			// 0-7: sum(dd * dp). 8: sum(res*dd). 9: 1/(1+sum(dd*dd))=inverse hessian entry.
Vec10f* JbBuffer_new;

float alphaW;
float alphaV;
float regWeight;
float couplingWeight;

Mat88f Hmt[MAX_THREAD], Hscmt[MAX_THREAD];
Vec8f bmt[MAX_THREAD], bscmt[MAX_THREAD];
Vec3f resmt[MAX_THREAD];

CoarseInitializer::CoarseInitializer(int ww, int hh)
{

	JbBuffer = new Vec10f[ww * hh];
	JbBuffer_new = new Vec10f[ww * hh];

	wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = 1.0; // SCALE_XI_ROT
	wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = 1.0; // for rotation

	float densities_scale = 1;
	densities[0] = 0.01 * densities_scale;
	densities[1] = 0.004 * densities_scale;
	densities[2] = 0.0016 * densities_scale;
	densities[3] = 0.00064 * densities_scale;
	densities[4] = 0.00064 * 0.25 * densities_scale;
	densities[5] = 0.00064 * 0.25 * densities_scale;
	for (int ii = 0; ii < PyrLevelsUsedG; ++ii)
	{
		points[ii] = new Pnt[4 * int(densities[ii] * wG[0] * hG[0])];
		numPoints[ii] = 0;
	}

	// Create a new pixel selector...
	statusMap = new float[wG[0] * hG[0]];
	statusMapB = new bool[wG[0] * hG[0]];
	sel = new PixelSelector(wG[0], hG[0]);
}

CoarseInitializer::~CoarseInitializer()
{
	for (int lvl = 0; lvl < PyrLevelsUsedG; lvl++)
	{
		if (points[lvl] != 0)
			delete[] points[lvl];
	}

	delete[] JbBuffer;
	delete[] JbBuffer_new;
	delete sel;
	delete[] statusMap;
	delete[] statusMapB;
}


double FindRightRow(Vec3 RawLeftPoint, double leftRow, double guessRightRow, double idepth, double AngSpeed, Vec3 AngAxis, Vec3 Velocity, Mat33 RLRG, Vec3 mTLR, int lvl){

	Vec3 curLeftPoint = KiG[lvl] * RawLeftPoint;

	// can be accelerated by precomputing
	int MaxTryTimes = 10;
	int tryTimes = 0;
	int scale = 1 << lvl;

	Vec3 curRightPoint;
	Mat33 Ridd;
	Vec3 tidd;
	while (tryTimes < MaxTryTimes)
	{
		Ridd = RLRG * Eigen::AngleAxisd(AngSpeed * (guessRightRow - leftRow), AngAxis).matrix();
		tidd = mTLR + guessRightRow * RLRG * Velocity - leftRow * Ridd * Velocity;

		curRightPoint = Ridd * curLeftPoint + tidd * idepth;

		// Just by a rotation!!MaxDisparity
		double u = curRightPoint[0] / curRightPoint[2];
		double v = curRightPoint[1] / curRightPoint[2];
		double Ku = fxG[lvl] * u + cxG[lvl];
		double Kv = fyG[lvl] * v + cyG[lvl];
		double newidpeth = idepth / curRightPoint[2];

		if (!(Ku > 2 && Kv > 2 && Ku < (wG[lvl] - 3.5) && Kv < (hG[lvl] - 3.5) && newidpeth > 0)){
			tryTimes = MaxTryTimes;
			break;
		}

		// Interpolate
		double KuOri = (Ku + 0.5) * scale - 0.5;
		double KvOri = (Kv + 0.5) * scale - 0.5;
		double rightRow = getInterpolatedElement(ptrRightRemapYG, KuOri, KvOri, wG[0]);
		if (fabs(rightRow - guessRightRow) < 0.25)		{
			break;
		}
		else{
			guessRightRow = rightRow;
		}

		tryTimes += 1;
	}

	if (tryTimes >= MaxTryTimes){
		return -1;
	}
	else{
		return guessRightRow;
	}

}



Vec2 RsProj(Vec3 RawLeftPoint, double leftRow, double guessRightRow, double idepth, double AngSpeed, Vec3 AngAxis, Vec3 Velocity, Mat33 RLRG, Vec3 mTLR, int lvl){

	Vec3 curLeftPoint = KiG[lvl] * RawLeftPoint;
	Vec3 curRightPoint;

	Mat33 Ridd = RLRG * Eigen::AngleAxisd(AngSpeed * (guessRightRow - leftRow), AngAxis).matrix();
	Vec3 tidd = mTLR + guessRightRow * RLRG * Velocity - leftRow * Ridd * Velocity;

	curRightPoint = Ridd * curLeftPoint + tidd * idepth;

	// Just by a rotation!!MaxDisparity
	double u = curRightPoint[0] / curRightPoint[2];
	double v = curRightPoint[1] / curRightPoint[2];
	double Ku = fxG[lvl] * u + cxG[lvl];
	double Kv = fyG[lvl] * v + cyG[lvl];
	double newidpeth = idepth / curRightPoint[2];

	if (!(Ku > 2 && Kv > 2 && Ku < (wG[lvl] - 3.5) && Kv < (hG[lvl] - 3.5) && newidpeth > 0)){
		return Vec2(-1,-1);
	}

	return Vec2(Ku, Kv);
}



int ComposeUV(double *_wvec, double *_vvec, CvMat *RLR, CvMat *tvec, CvMat *leftPoint, double idepth, double leftrow, double rightrow, double *ptrUV, double *_dUVdw, double *_dUVdv, double *_dUVdid, int lvl)
{
	double _wTrows[3]; 
	double _R1[9], _dR1dwrows[9 * 3];
	double _dR3dw[9 * 3]; 
	CvMat wTrows = cvMat(3, 1, CV_64F, _wTrows);
	CvMat R1 = cvMat(3, 3, CV_64F, _R1);
	CvMat dR1dwrows = cvMat(9, 3, CV_64F, _dR1dwrows);
	CvMat dR3dw = cvMat(9, 3, CV_64F, _dR3dw);

	// convert data from a mat to a mat, sometimes, will change data type
	memcpy(_wTrows, _wvec, sizeof(double) * 3);
	_wTrows[0] = _wTrows[0] * (rightrow - leftrow); // w * (rightrow - leftrow)
	_wTrows[1] = _wTrows[1] * (rightrow - leftrow);
	_wTrows[2] = _wTrows[2] * (rightrow - leftrow);

	cvRodrigues2(&wTrows, &R1, &dR1dwrows);

	double *ptrdata = dR1dwrows.data.db;
	for (int ii = 0; ii < 3 * 9; ++ii){
		ptrdata[ii] = ptrdata[ii] * (rightrow - leftrow);
	}

	double _R3[9], _dR3dR1[9 * 9], _dR3dR2[9 * 9];
	CvMat R3 = cvMat(3, 3, CV_64F, _R3);
	CvMat dR3dR1 = cvMat(9, 9, CV_64F, _dR3dR1), dR3dR2 = cvMat(9, 9, CV_64F, _dR3dR2);

	cvMatMul(RLR, &R1, &R3);
	// Deriv
	cvCalcMatMulDeriv(RLR, &R1, &dR3dR2, &dR3dR1);
	cvMatMul(&dR3dR1, &dR1dwrows, &dR3dw);


    double _t1[3], _t1_v0[3], _t1_v1[3], _t2[3], _t3[3], _dt3dR2[3 * 9], _dt1dR1[3 * 9];
    double _dt3dt1[3 * 3], _dt1dt1_v0[3 * 3], _W3[3 * 9], _dt3dw[3 * 3], _dt3dv[3 * 3];
    CvMat t1 = cvMat(3, 1, CV_64F, _t1), t2 = cvMat(3, 1, CV_64F, _t2);
    CvMat t1_v0 = cvMat(3, 1, CV_64F, _t1_v0), t1_v1 = cvMat(3, 1, CV_64F, _t1_v1);
    CvMat t3 = cvMat(3, 1, CV_64F, _t3);
    CvMat dt3dR2 = cvMat(3, 9, CV_64F, _dt3dR2);
    CvMat dt3dt1 = cvMat(3, 3, CV_64F, _dt3dt1);
    CvMat dt1dR1 = cvMat(3, 9, CV_64F, _dt1dR1);
    CvMat dt1dt1_v0 = cvMat(3, 3, CV_64F, _dt1dt1_v0);
    CvMat W3 = cvMat(3, 9, CV_64F, _W3);
    CvMat dt3dw = cvMat(3, 3, CV_64F, _dt3dw);
    CvMat dt3dv = cvMat(3, 3, CV_64F, _dt3dv);

    memcpy(_t1_v0, _vvec, sizeof(double) * 3);
    memcpy(_t1_v1, _vvec, sizeof(double) * 3);
    _t1_v0[0] = -1 * _t1_v0[0] * leftrow;
    _t1_v0[1] = -1 * _t1_v0[1] * leftrow;
    _t1_v0[2] = -1 * _t1_v0[2] * leftrow;

    _t1_v1[0] = _t1_v1[0] * rightrow;
    _t1_v1[1] = _t1_v1[1] * rightrow;
    _t1_v1[2] = _t1_v1[2] * rightrow;

    cvConvert(tvec, &t2);
    cvMatMulAdd(&R1, &t1_v0, &t1_v1, &t1);
    cvMatMulAdd(RLR, &t1, &t2, &t3);

    cvCalcMatMulDeriv(&R1, &t1_v0, &dt1dR1, &dt1dt1_v0);
    cvCalcMatMulDeriv(RLR, &t1, &dt3dR2, &dt3dt1);

    cvMatMul(&dt3dt1, &dt1dR1, &W3);
    cvMatMul(&W3, &dR1dwrows, &dt3dw);

    _dt1dt1_v0[0] = -leftrow * _dt1dt1_v0[0] + rightrow;
    _dt1dt1_v0[1] = -leftrow * _dt1dt1_v0[1];
    _dt1dt1_v0[2] = -leftrow * _dt1dt1_v0[2];
    _dt1dt1_v0[3] = -leftrow * _dt1dt1_v0[3];
    _dt1dt1_v0[4] = -leftrow * _dt1dt1_v0[4] + rightrow;
    _dt1dt1_v0[5] = -leftrow * _dt1dt1_v0[5];
    _dt1dt1_v0[6] = -leftrow * _dt1dt1_v0[6];
    _dt1dt1_v0[7] = -leftrow * _dt1dt1_v0[7];
    _dt1dt1_v0[8] = -leftrow * _dt1dt1_v0[8] + rightrow;
    cvMatMul(&dt3dt1, &dt1dt1_v0, &dt3dv);

	double _XYZ[3], _dXYZdw[9], _dXYZdv[9], _dXYZdR3[27], _dXYZdt3[9];
	double _t3id[3];
	CvMat XYZ = cvMat(3, 1, CV_64F, _XYZ);
	CvMat dXYZdw = cvMat(3, 3, CV_64F, _dXYZdw);
	CvMat dXYZdv = cvMat(3, 3, CV_64F, _dXYZdv);
	CvMat dXYZdR3 = cvMat(3, 9, CV_64F, _dXYZdR3);
	CvMat dXYZdt3 = cvMat(3, 3, CV_64F, _dXYZdt3);
	CvMat t3id = cvMat(3, 1, CV_64F, _t3id);
	_t3id[0] = _t3[0] * idepth;
	_t3id[1] = _t3[1] * idepth;
	_t3id[2] = _t3[2] * idepth;
	cvMatMulAdd(&R3, leftPoint, &t3id, &XYZ);
    cvCalcMatMulDeriv(&R3, leftPoint, &dXYZdR3, &dXYZdt3);
	_dXYZdt3[0] = idepth;
	_dXYZdt3[1] = 0;
	_dXYZdt3[2] = 0;
	_dXYZdt3[3] = 0;
	_dXYZdt3[4] = idepth;
	_dXYZdt3[5] = 0;
	_dXYZdt3[6] = 0;
	_dXYZdt3[7] = 0;
	_dXYZdt3[8] = idepth;


	cvMatMul(&dXYZdR3, &dR3dw, &dXYZdw);
	cvMatMulAdd(&dXYZdt3, &dt3dw, &dXYZdw, &dXYZdw);
	cvMatMul(&dXYZdt3, &dt3dv, &dXYZdv);


	ptrUV[0] = fxG[lvl] * _XYZ[0] / _XYZ[2] + cxG[lvl];
	ptrUV[1] = fyG[lvl] * _XYZ[1] / _XYZ[2] + cyG[lvl];

	// Construct 2 * 3 matrix
	double _duvdxyz[6];
	CvMat duvdxyz = cvMat(2, 3, CV_64F, _duvdxyz);
	_duvdxyz[0] = fxG[lvl] / _XYZ[2];
	_duvdxyz[1] = 0;
	_duvdxyz[2] = -fxG[lvl] * _XYZ[0] / (_XYZ[2] * _XYZ[2]);
	_duvdxyz[3] = 0;
	_duvdxyz[4] = fyG[lvl] / _XYZ[2];
	_duvdxyz[5] = -fyG[lvl] * _XYZ[1] / (_XYZ[2] * _XYZ[2]);

	CvMat dUVdw = cvMat(2, 3, CV_64F, _dUVdw);
	CvMat dUVdv = cvMat(2, 3, CV_64F, _dUVdv);
	CvMat dUVdid = cvMat(2, 1, CV_64F, _dUVdid);

	cvMatMul(&duvdxyz, &dXYZdw, &dUVdw);	
	cvMatMul(&duvdxyz, &dXYZdv, &dUVdv);
	cvMatMul(&duvdxyz, &t3, &dUVdid);

	return 0;
}


void calcResAndGS_new4_MT(
	int lvl, Vec3 Angular, Vec3 Velocity, bool IsInitialDepth, bool FixTranslation, Vec3 InitialVelocity, 
	int threadID, int start, int end, Pnt *ptsl)
{
	Mat88f H_out;
	Vec8f b_out;
	Mat88f H_out_sc;
	Vec8f b_out_sc;
	Vec3f res;


	int wl = wG[lvl];
	Eigen::Vector3f *colorRef = leftFrameG->dIp[lvl];
	Eigen::Vector3f *colorNew = rightFrameG->dIp[lvl];

	Mat33 RKi;

	int scale = 1 << lvl;

	double AngSpeed = Angular.norm();
	Vec3 AngAxis;
	if (AngSpeed < 0.0000001){
		AngAxis[0] = 1.0;
		AngAxis[1] = 1.0;
		AngAxis[2] = 1.0;
		AngAxis = AngAxis / AngAxis.norm();
	}
	else{
		AngAxis = Angular / AngSpeed;
	}

	Accumulator11 E;
	acc9MT[threadID].initialize();
	E.initialize();

	bool isGood;
	double energy;
	Vec3 curLeftPoint, mpt, rpt;
	double Ku, Kv;
	int SumNotGood0 = 0;

	RKi = RLRG * KiG[lvl];

	double _cvRLR[9], _cvTLR[3];
	CvMat cvRLR = cvMat(3,3,CV_64F,_cvRLR);
	CvMat cvTLR = cvMat(3,1,CV_64F,_cvTLR);
	_cvRLR[0] = RLRG(0,0);
	_cvRLR[1] = RLRG(0,1);
	_cvRLR[2] = RLRG(0,2);
	_cvRLR[3] = RLRG(1,0);
	_cvRLR[4] = RLRG(1,1);
	_cvRLR[5] = RLRG(1,2);
	_cvRLR[6] = RLRG(2,0);
	_cvRLR[7] = RLRG(2,1);
	_cvRLR[8] = RLRG(2,2);

	_cvTLR[0] = TLRG[0];
	_cvTLR[1] = TLRG[1];
	_cvTLR[2] = TLRG[2];


	double SumResidual = 0;
	double LocalResidual = 0;
	double SumEnergy = 0;
	int SumNumber = 0;

	for (int i = start; i < end; i++)
	{
		Pnt *point = ptsl + i;
		point->maxstep = 1e10;
		if (!point->isGood){
			SumNotGood0 += 1;
			E.updateSingle((float)(point->energy[0]));
			point->energy_new = point->energy; 
			point->isGood_new = false;		   
			continue;						   
		}

		double score[4];
		double initialRightRow[4];

		Vec3 rawLeftPoint;
		rawLeftPoint << point->u, point->v, 1;
		double leftRow = getInterpolatedElement(ptrLeftRemapYG, (rawLeftPoint[0]+0.5)*scale-0.5, (rawLeftPoint[1]+0.5) * scale-0.5, wG[0]);

		double idepth;
		double guessRightRow;
		if (IsInitialDepth){
			// select best idepth candidate
			for (int kk = 0; kk < point->valid_idepth_num; ++kk){
				score[kk] = -1;

				idepth = point->idepth_candidates[kk];
				// calc initial right cooridinate,
				rpt = KG[lvl] * (RKi * rawLeftPoint + TLRG * idepth);
				Ku = rpt[0] / rpt[2];
				Kv = rpt[1] / rpt[2];

				if (!(Ku > 2 && Kv > 2 && Ku < (wG[lvl] - 3.5) && Kv < (hG[lvl] - 3.5) && rpt[2] > 0)){
					continue;						   
				}

				guessRightRow = getInterpolatedElement(ptrRightRemapYG, (Ku+0.5)*scale-0.5, (Kv+0.5)*scale-0.5, wG[0]);
				guessRightRow = FindRightRow(rawLeftPoint, leftRow, guessRightRow, idepth, AngSpeed, AngAxis, Velocity,RLRG, TLRG,lvl);
				initialRightRow[kk] = guessRightRow;

				if (point->valid_idepth_num == 1){
					if (guessRightRow > 0){
						score[kk] = 1;
					}
					break;
				}

				if (guessRightRow < 0){
					continue;
				}
				else{
					int ww;
					score[kk] = 0; 
					for (ww = 0; ww < patternNum; ++ww){
						int dx = patternP[ww][0];
						int dy = patternP[ww][1];
						rawLeftPoint << point->u + dx, point->v + dy, 1;

						leftRow = getInterpolatedElement(ptrLeftRemapYG, (rawLeftPoint[0] + 0.5) * scale, (rawLeftPoint[1] + 0.5) * scale, wG[0]);
						guessRightRow = FindRightRow(rawLeftPoint, leftRow, guessRightRow, idepth, AngSpeed, AngAxis, Velocity,RLRG, TLRG,lvl);

						Vec2 UV = RsProj(rawLeftPoint, leftRow, guessRightRow, idepth, AngSpeed, AngAxis, Velocity, RLRG, TLRG, lvl);
						if (UV[0] < 0){
							break;
						}
						Vec3f hitColor = getInterpolatedElement33(colorNew, UV[0], UV[1], wl);
						double rlR = getInterpolatedElement31(colorRef, point->u + dx, point->v + dy, wl);
						score[kk] += fabs(hitColor[0] - rlR);  // ToDo:: add weights based on gradient
					}
					if (ww < patternNum){
						score[kk] = -1;
					}
				}
			}

			int idepth_idx = -1;
			double bestScore = DBL_MAX;
			for (int kk = 0; kk < point->valid_idepth_num; ++kk){
				if (score[kk] > 0){
					if (bestScore > score[kk]){
						bestScore = score[kk];
						idepth_idx = kk;
					}
				}
			}

			if (idepth_idx == -1){
				E.updateSingle((float)(point->energy[0]));
				point->energy_new = point->energy; 
				point->isGood_new = false;		   
				continue;						   
			}

			idepth = point->idepth_candidates[idepth_idx];
            point->idepth_initial = point->idepth_new = point->idepth = point->ori_idepth = point->idepth_candidates[0];
			guessRightRow = initialRightRow[idepth_idx];
			point->guessRightRow = guessRightRow;
		}
		else{
			idepth = point->idepth_new;
			guessRightRow = point->guessRightRow;
		}


		VecNRf dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7, dd, r;
		JbBuffer_new[i].setZero();

		isGood = true;
		energy = 0;

		Mat33 Ridd;
		Vec3 tidd, inc;
		Vec3 curRightPoint;

		// sum over all residuals.
		LocalResidual = 0;
		int gradZeros = 0;
		for (int idx = 0; idx < patternNum; idx++){
			int dx = patternP[idx][0];
			int dy = patternP[idx][1];

			// Projection
			curLeftPoint << point->u + dx, point->v + dy, 1;
			leftRow = getInterpolatedElement(ptrLeftRemapYG, (curLeftPoint[0]+0.5)*scale-0.5, (curLeftPoint[1]+0.5)*scale-0.5, wG[0]);
			guessRightRow = FindRightRow(curLeftPoint, leftRow, guessRightRow, idepth, AngSpeed, AngAxis, Velocity, RLRG, TLRG,lvl);

			if (guessRightRow < 0){
				isGood = false;
				break;
			}

			if (dx == 0 && dy == 0){
				point->guessRightRow = guessRightRow;
				point->deltaRow = guessRightRow - leftRow;
			}

			double UV[2], dUVdw[6], dUVdv[6], dUVdid[2];
			double _leftPoint[3];
			curLeftPoint = KiG[lvl] * curLeftPoint;
			_leftPoint[0] = curLeftPoint[0];
			_leftPoint[1] = curLeftPoint[1];
			_leftPoint[2] = curLeftPoint[2];

			CvMat cvleftPoint = cvMat(3,1,CV_64F,_leftPoint);
			ComposeUV(Angular.data(), Velocity.data(), &cvRLR, &cvTLR, &cvleftPoint, idepth, leftRow, guessRightRow, UV, dUVdw, dUVdv, dUVdid, lvl);

			Vec3f hitColor = getInterpolatedElement33(colorNew, UV[0], UV[1], wl);
			double rlR = getInterpolatedElement31(colorRef, point->u + dx, point->v + dy, wl);
			if (!std::isfinite(rlR) || !std::isfinite((float)hitColor[0])){
				isGood = false;
				break;
			}

			double residual = hitColor[0] - rlR;
			double hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
			energy += hw * residual * residual * (2 - hw);
			LocalResidual += hw * residual * residual * hw;

			double drdu = hw*hitColor[1];
			double drdv = hw*hitColor[2];

			// // Derivative for rotation
			dp0[idx] = drdu * dUVdw[0] + drdv * dUVdw[3];
			dp1[idx] = drdu * dUVdw[1] + drdv * dUVdw[4];
			dp2[idx] = drdu * dUVdw[2] + drdv * dUVdw[5];

			// Derivative for translation
			dp3[idx] = drdu * dUVdv[0] + drdv * dUVdv[3];
			dp4[idx] = drdu * dUVdv[1] + drdv * dUVdv[4];
			dp5[idx] = drdu * dUVdv[2] + drdv * dUVdv[5];

			// Null
			dp6[idx] = 0.0;
			dp7[idx] = 0.0;

			// Derivative for the inverse depth of this point
			dd[idx] = drdu * dUVdid[0]  + drdv * dUVdid[1];
			// Residual
			r[idx] = hw*residual;

			float maxstep = 1.0f / Vec2(dUVdid[0], dUVdid[1]).norm();
			if (maxstep < point->maxstep)
				point->maxstep = maxstep;

			JbBuffer_new[i][0] += dp0[idx] * dd[idx];
			JbBuffer_new[i][1] += dp1[idx] * dd[idx];
			JbBuffer_new[i][2] += dp2[idx] * dd[idx];
			JbBuffer_new[i][3] += dp3[idx] * dd[idx];
			JbBuffer_new[i][4] += dp4[idx] * dd[idx];
			JbBuffer_new[i][5] += dp5[idx] * dd[idx];
			JbBuffer_new[i][6] += dp6[idx] * dd[idx];
			JbBuffer_new[i][7] += dp7[idx] * dd[idx];
			JbBuffer_new[i][8] += r[idx] * dd[idx];
			JbBuffer_new[i][9] += dd[idx] * dd[idx];

			if (dd[idx] == 0){
				gradZeros += 1;
 			}

			if (gradZeros >= 5){
				isGood = false;
				break;
			}
		}

		if (!isGood || energy > point->outlierTH * 20)
		{
			E.updateSingle((float)(point->energy[0]));
			point->isGood_new = false;
			point->energy_new = point->energy;
			continue;
		}

		E.updateSingle(energy);
		point->isGood_new = true;
		point->energy_new[0] = energy; 

		for (int i = 0; i + 3 < patternNum; i += 4)
			acc9MT[threadID].updateSSE(
				_mm_load_ps(((float *)(&dp0)) + i),
				_mm_load_ps(((float *)(&dp1)) + i),
				_mm_load_ps(((float *)(&dp2)) + i),
				_mm_load_ps(((float *)(&dp3)) + i),
				_mm_load_ps(((float *)(&dp4)) + i),
				_mm_load_ps(((float *)(&dp5)) + i),
				_mm_load_ps(((float *)(&dp6)) + i),
				_mm_load_ps(((float *)(&dp7)) + i),
				_mm_load_ps(((float *)(&r)) + i));

		for (int i = ((patternNum >> 2) << 2); i < patternNum; i++)
			acc9MT[threadID].updateSingle(
				(float)dp0[i], (float)dp1[i], (float)dp2[i], (float)dp3[i],
				(float)dp4[i], (float)dp5[i], (float)dp6[i], (float)dp7[i],
				(float)r[i]);
		

		SumResidual += LocalResidual;
		SumEnergy += energy;
		SumNumber += 1;
	}


	E.finish();
	acc9MT[threadID].finish();

	acc9SCMT[threadID].initialize();
	for (int i = start; i < end; i++)
	{
		Pnt *point = ptsl + i;
		if (!point->isGood_new)
			continue;

		point->lastHessian_new = JbBuffer_new[i][9];

		JbBuffer_new[i][8] += couplingWeight * (point->idepth_new - point->idepth_initial);
		JbBuffer_new[i][9] += couplingWeight;

		JbBuffer_new[i][9] =  1 / JbBuffer_new[i][9];
		acc9SCMT[threadID].updateSingleWeighted(
			(float)JbBuffer_new[i][0], (float)JbBuffer_new[i][1], (float)JbBuffer_new[i][2], (float)JbBuffer_new[i][3],
			(float)JbBuffer_new[i][4], (float)JbBuffer_new[i][5], (float)JbBuffer_new[i][6], (float)JbBuffer_new[i][7],
			(float)JbBuffer_new[i][8], (float)JbBuffer_new[i][9]);
	}

	acc9SCMT[threadID].finish();

	H_out = acc9MT[threadID].H.topLeftCorner<8, 8>();
	b_out = acc9MT[threadID].H.topRightCorner<8, 1>();
	H_out_sc = acc9SCMT[threadID].H.topLeftCorner<8, 8>();
	b_out_sc = acc9SCMT[threadID].H.topRightCorner<8, 1>();

	SumNumber = end - start;
	if (!FixTranslation){
		H_out(0, 0) += alphaW * SumNumber; //(end - start);
		H_out(1, 1) += alphaW * SumNumber; // (end - start);
		H_out(2, 2) += alphaW * SumNumber; // (end - start);

		b_out[0] += Angular[0] * alphaW * SumNumber; // (end - start);
		b_out[1] += Angular[1] * alphaW * SumNumber; // (end - start);
		b_out[2] += Angular[2] * alphaW * SumNumber; // (end - start);
	}
	else{
		// InitialVelocity
		H_out(3, 3) += alphaV * SumNumber; // (end - start);
		H_out(4, 4) += alphaV * SumNumber; // (end - start);
		H_out(5, 5) += alphaV * SumNumber; // (end - start);

		b_out[3] += (Velocity[0] - InitialVelocity[0]) * alphaV * SumNumber; // (end - start);
		b_out[4] += (Velocity[1] - InitialVelocity[1]) * alphaV * SumNumber; // (end - start);
		b_out[5] += (Velocity[2] - InitialVelocity[2]) * alphaV * SumNumber; // (end - start);
	}

	int TEST_V_PRIOR = 1;
	// double alphaV2[4] = {150*150*4e2*4e2, 150*150*1e2*1e2, 0, 0};
	double alphaV2[4] = {0, 0, 0, 0};
	if (TEST_V_PRIOR){
		// InitialVelocity
		H_out(3, 3) += alphaV2[lvl] * SumNumber; // (end - start);
		H_out(4, 4) += alphaV2[lvl] * SumNumber; // (end - start);
		H_out(5, 5) += alphaV2[lvl] * SumNumber; // (end - start);

		b_out[3] += Velocity[0] * alphaV2[lvl] * SumNumber; // (end - start);
		b_out[4] += Velocity[1] * alphaV2[lvl] * SumNumber; // (end - start);
		b_out[5] += Velocity[2] * alphaV2[lvl] * SumNumber; // (end - start);
	}

	Hmt[threadID] = H_out;
	bmt[threadID] = b_out;
	Hscmt[threadID] = H_out_sc;
	bscmt[threadID] = b_out_sc;
	resmt[threadID] = Vec3f(E.A, SumResidual, E.num);
}


void doStep_MT(int lvl, float lambda, Vec6f inc, int start, int end, Pnt *pts)
{
	const float maxPixelStep = 0.25;
	const float idMaxStep = 1e10;
	int sum = 0;
	for (int i = start; i < end; ++i){
		if (!pts[i].isGood)
			continue;

		sum += 1;

		float b = JbBuffer[i][8] + JbBuffer[i].head<6>().dot(inc);
		float step = -b * JbBuffer[i][9] / (1 + lambda);

		float maxstep = maxPixelStep * pts[i].maxstep;
		if (maxstep > idMaxStep)
			maxstep = idMaxStep;

		if (step > maxstep)
			step = maxstep;
		if (step < -maxstep)
			step = -maxstep;

		float newIdepth = pts[i].idepth + step;
		if (newIdepth < 1e-3)
			newIdepth = 1e-3;
		if (newIdepth > 50)
			newIdepth = 50;
		pts[i].idepth_new = newIdepth;
	}
}


std::vector<double> CoarseInitializer::Optimize3_withdepths_MT(std::vector<double> motionState, int pyramidLvl, bool FixTranslation)
{
	printDebug = false;
	bool printObservability = false;

	int maxIterations[] = {100, 100, 100, 100, 100}; // 100 iterations

	alphaW = 150 * 150 * 3e4 * 3e4;				 //*freeDebugParam2*freeDebugParam2;
	alphaV = 150 * 150 * 3e4 * 3e4;
	regWeight = 0.8;				 //*freeDebugParam4;
	couplingWeight = 0;// 150 * 150 * 28; //*freeDebugParam5;

	Vec3 Angular(motionState[0] * RowTimeG, motionState[1] * RowTimeG, motionState[2] * RowTimeG);
	Vec3 Velocity(motionState[3] * RowTimeG, motionState[4] * RowTimeG, motionState[5] * RowTimeG);
	Vec3 InitialVelocity(motionState[3] * RowTimeG, motionState[4] * RowTimeG, motionState[5] * RowTimeG);

	Vec3f latestRes = Vec3f::Zero();

	int lvl = pyramidLvl;
	// sc, schur
	Mat88f H, Hsc;
	Vec8f b, bsc;
	Vec3f resOld;

    bool multiThread = true;
    int threadCnt = THREAD_CNT;
    int taskCnt = numPoints[lvl];
    int thread_step = int(ceil(taskCnt/threadCnt));
    if(multiThread){
        std::vector<std::thread> thread_pool;
        for (int it = 0; it < threadCnt; ++it){
            int pt_start = thread_step * it;
            int pt_end = thread_step * (it + 1);

			if (pt_end > taskCnt){
				pt_end = taskCnt;
			}
            std::thread this_thread(calcResAndGS_new4_MT, lvl, Angular, Velocity, true, FixTranslation, InitialVelocity, it, pt_start, pt_end, points[lvl]);
			thread_pool.push_back(std::move(this_thread));
        }
        for (unsigned int it = 0; it < thread_pool.size(); ++it){
            if(thread_pool[it].joinable())
                thread_pool[it].join();
        }
		H = Hmt[0];
		b = bmt[0];
		Hsc = Hscmt[0];
		bsc = bscmt[0];
		resOld = resmt[0];
		for (int it = 1; it < threadCnt; ++it){
			H = H + Hmt[it];
			b = b + bmt[it];
			Hsc = Hsc + Hscmt[it];
			bsc = bsc + bscmt[it];
			resOld = resOld + resmt[it];
		}
    }
    else{
        calcResAndGS_new4_MT(lvl, Angular, Velocity, true, FixTranslation, InitialVelocity, 0,
		0, numPoints[lvl], points[lvl]);
		H = Hmt[0];
		b = bmt[0];
		Hsc = Hscmt[0];
		bsc = bscmt[0];
		resOld = resmt[0];
    }

	applyStep(lvl);

	float lambda = 0.1;
	float eps = 1e-7;
	int fails = 0;

	if (printDebug)
	{
		printf("lvl %d, it %d (l=%f) %s: %.5f (|inc| = %f) Velocity:%f! \n",
			   lvl, 0, lambda,
			   "AverageEnergy",
			   sqrtf((float)(resOld[0] / resOld[2])),
			   0.0f, Velocity.norm());
	}

	int iteration = 0;
	while (true)
	{
		Mat66f Hl = H.topLeftCorner<6, 6>();
		for (int i = 0; i < 6; i++)
			Hl(i, i) *= (1 + lambda);

		Hl -= Hsc.topLeftCorner<6, 6>() * (1 / (1 + lambda));
		Vec6f bl = b.head<6>() - bsc.head<6>() * (1 / (1 + lambda));

		Hl = wM * Hl * wM * (0.01f/(wG[lvl]*hG[lvl]));
		bl = wM * bl * (0.01f/(wG[lvl]*hG[lvl]));

		// The observability
		if (printObservability)
		{
			Eigen::EigenSolver<Mat66f> es(Hl);

			Mat66f D = es.pseudoEigenvalueMatrix();
			Mat66f V = es.pseudoEigenvectors();

			std::cout << "The pseudo-eigenvalue matrix D is:" << std::endl
					  << D << std::endl;
			std::cout << "The pseudo-eigenvector matrix V is:" << std::endl
					  << V << std::endl;
		}

		Vec6f inc;
		inc = -(wM * (Hl.ldlt().solve(bl))); //=-H^-1 * b.

		// Notice new update method
		Vec3 Angular_new = Angular + inc.head<3>().cast<double>();
		Vec3 Velocity_new = Velocity + inc.tail<3>().cast<double>();


		if(multiThread){
			// printf("B\n");
			std::vector<std::thread> thread_pool;
			for (int it = 0; it < threadCnt; ++it){
				int pt_start = thread_step * it;
				int pt_end = thread_step * (it + 1);

				if (pt_end > taskCnt){
					pt_end = taskCnt;
				}

				std::thread this_thread(doStep_MT, lvl, lambda, inc, pt_start, pt_end, points[lvl]);
				thread_pool.push_back(std::move(this_thread));
			}
			for (unsigned int it = 0; it < thread_pool.size(); ++it){
				if(thread_pool[it].joinable())
					thread_pool[it].join();
			}
		}
		else{
			doStep_MT(lvl, lambda, inc, 0, numPoints[lvl], points[lvl]);
		}


		Mat88f H_new, Hsc_new;
		Vec8f b_new, bsc_new;

		Vec3f resNew;

		if(multiThread){
			// printf("B\n");
			std::vector<std::thread> thread_pool;
			for (int it = 0; it < threadCnt; ++it){
				int pt_start = thread_step * it;
				int pt_end = thread_step * (it + 1);

				if (pt_end > taskCnt){
					pt_end = taskCnt;
				}

				std::thread this_thread(calcResAndGS_new4_MT, lvl, Angular_new, Velocity_new, false, FixTranslation, InitialVelocity, it, pt_start, pt_end, points[lvl]);
				thread_pool.push_back(std::move(this_thread));
			}
			for (unsigned int it = 0; it < thread_pool.size(); ++it){
				if(thread_pool[it].joinable())
					thread_pool[it].join();
			}
			H_new = Hmt[0];
			b_new = bmt[0];
			Hsc_new = Hscmt[0];
			bsc_new = bscmt[0];
			resNew = resmt[0];
			for (int it = 1; it < threadCnt; ++it){
				H_new = H_new + Hmt[it];
				b_new = b_new + bmt[it];
				Hsc_new = Hsc_new + Hscmt[it];
				bsc_new = bsc_new + bscmt[it];
				resNew = resNew + resmt[it];
			}
		}
		else{
			calcResAndGS_new4_MT(lvl, Angular_new, Velocity_new, false, FixTranslation, InitialVelocity, 0,
			0, numPoints[lvl], points[lvl]);
			H_new = Hmt[0];
			b_new = bmt[0];
			Hsc_new = Hscmt[0];
			bsc_new = bscmt[0];
			resNew = resmt[0];
		}


		float eTotalNew = resNew[0];
		float eTotalOld = resOld[0];

		bool accept = eTotalOld > eTotalNew;

		if (printDebug)
		{
			printf("lvl %d, it %d (l=%f) %s: AverageEnergy %.5f -> %.5f (|inc| = %f) Ang:%f Velocity:%f! \n",
				   lvl, iteration, lambda,
				   (accept ? "ACCEPT" : "REJECT"),
				   sqrtf((float)(resOld[0] / resOld[2])),
				   sqrtf((float)(resNew[0] / resNew[2])),
				   inc.norm(), Angular_new.norm() * 180 / 3.1415926, Velocity_new.norm());
		}

		if (accept)
		{
			H = H_new;
			b = b_new;
			Hsc = Hsc_new;
			bsc = bsc_new;
			resOld = resNew;
			Angular = Angular_new;
			Velocity = Velocity_new;
			applyStep(lvl); // new to constant
			lambda *= 0.5;
			fails = 0;
			if (lambda < 0.0001)
				lambda = 0.0001;
		}
		else
		{
			fails++;
			lambda *= 4;
			if (lambda > 10000)
				lambda = 10000;
		}

		bool quitOpt = false;
		if (!(inc.norm() > eps) || iteration >= maxIterations[lvl])
		{
			// Mat88f H,Hsc; Vec8f b,bsc;
			quitOpt = true;
		}

		if (quitOpt)
			break;
		iteration++;
	}
	latestRes = resOld;

	printf("LvL:%d iter:%d Before State: %f %f %f %f %f %f\n", pyramidLvl, iteration, motionState[0], motionState[1], motionState[2], motionState[3], motionState[4], motionState[5]);
	printf("Lvl:%d iter:%d After State: %f %f %f %f %f %f\n", pyramidLvl, iteration, Angular[0]/RowTimeG, Angular[1]/RowTimeG, Angular[2]/RowTimeG, Velocity[0]/RowTimeG, Velocity[1]/RowTimeG, Velocity[2]/RowTimeG);

	std::vector<double> new_motionState;

	new_motionState.push_back(Angular[0]/RowTimeG);
	new_motionState.push_back(Angular[1]/RowTimeG);
	new_motionState.push_back(Angular[2]/RowTimeG);

	new_motionState.push_back(Velocity[0]/RowTimeG);
	new_motionState.push_back(Velocity[1]/RowTimeG);
	new_motionState.push_back(Velocity[2]/RowTimeG);


	return new_motionState;
}





void CoarseInitializer::setLeftFramePym(FramePym *newFrameHessian)
{
	// Create the intrinsic matrix for pyramid
	leftFrameG = newFrameHessian;

	for (int lvl = 0; lvl < PyrLevelsUsedG; lvl++)
	{
		sel->currentPotential = 3;
		int npts;

		if (lvl == 0)
			npts = sel->makeMaps(leftFrameG, statusMap, densities[lvl] * wG[0] * hG[0], 1, false, 2);
		else
			npts = makePixelStatus(leftFrameG->dIp[lvl], statusMapB, wG[lvl], hG[lvl], densities[lvl] * wG[0] * hG[0]);

		// set idepth map to initially 1 everywhere.
		int wl = wG[lvl], hl = hG[lvl];
		Pnt *pl = points[lvl];

		int nl = 0;

		for (int y = DisparityEdgeG + 2; y < hl - DisparityEdgeG - 3; y++)
		{
			for (int x = DisparityEdgeG + 2; x < wl - DisparityEdgeG - 3; x++)
			{
				//if(x==2) printf("y=%d!\n",y);
				if ((lvl != 0 && statusMapB[x + y * wl]) || (lvl == 0 && statusMap[x + y * wl] != 0))
				{
					// printf("%d: %f %f\n", nl, pl[nl].u, pl[nl].v);
					//assert(patternNum==9);
					pl[nl].u = x; // !!!
					pl[nl].v = y;
					pl[nl].idepth = 1;
					pl[nl].isGood = true;
					pl[nl].energy.setZero(); // energy 的初始值为0
					pl[nl].lastHessian = 0;
					pl[nl].lastHessian_new = 0;
					pl[nl].my_type = (lvl != 0) ? 1 : statusMap[x + y * wl];
					pl[nl].valid_idepth_num = 0;

					// memset(pl[nl].dest_ori_rowindex, 0, sizeof(pl[nl].dest_ori_rowindex[0]) * patternNum);
					Eigen::Vector3f *cpt = leftFrameG->dIp[lvl] + x + y * wG[lvl]; // this is a pointer
					float sumGrad2 = 0;
					// patternP: 8 for SSE efficiency
					for (int idx = 0; idx < patternNum; idx++)
					{
						int dx = patternP[idx][0];
						int dy = patternP[idx][1];
						// squaredNorm()：the squared l2 norm of *this
						// For vectors, this is also equals to the dot product of *this with itself.
						float absgrad = cpt[dx + dy * wG[lvl]].tail<2>().squaredNorm();
						sumGrad2 += absgrad;
					}

					pl[nl].outlierTH = patternNum * setting_outlierTH; // setting_outlierTH： 12*12;

					nl++;
					assert(nl <= npts);
				}
			}

		}

		numPoints[lvl] = nl;
	}

}

void CoarseInitializer::setRightFramePym(FramePym *newFrameHessian)
{
	rightFrameG = newFrameHessian;
}



void CoarseInitializer::doStep(int lvl, float lambda, Vec6f inc)
{

	const float maxPixelStep = 0.25;
	const float idMaxStep = 1e10;
	Pnt *pts = points[lvl];
	int npts = numPoints[lvl];
	int sum = 0;
	for (int i = 0; i < npts; i++)
	{
		if (!pts[i].isGood)
			continue;

		sum += 1;

		
		float b = JbBuffer[i][8] + JbBuffer[i].head<6>().dot(inc);
		float step = -b * JbBuffer[i][9] / (1 + lambda);

		float maxstep = maxPixelStep * pts[i].maxstep;
		if (maxstep > idMaxStep)
			maxstep = idMaxStep;

		if (step > maxstep)
			step = maxstep;
		if (step < -maxstep)
			step = -maxstep;

		float newIdepth = pts[i].idepth + step;
		if (newIdepth < 1e-3)
			newIdepth = 1e-3; 
		if (newIdepth > 50)
			newIdepth = 50;
		pts[i].idepth_new = newIdepth;
	}

	// printf("Depth Update Sum:%d!\n",sum);
}
void CoarseInitializer::applyStep(int lvl)
{
	Pnt *pts = points[lvl];
	int npts = numPoints[lvl];
	for (int i = 0; i < npts; i++)
	{
		if (!pts[i].isGood)
		{
			pts[i].idepth = pts[i].idepth_new;
			continue;
		}
		pts[i].energy = pts[i].energy_new;
		pts[i].isGood = pts[i].isGood_new; 
		pts[i].idepth = pts[i].idepth_new;
		pts[i].lastHessian = pts[i].lastHessian_new; 
	}
	std::swap<Vec10f *>(JbBuffer, JbBuffer_new);
}

void CoarseInitializer::makeNN()
{
	const float NNDistFactor = 0.05;

	typedef nanoflann::KDTreeSingleIndexAdaptor<
		nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud>,
		FLANNPointcloud, 2>
		KDTree;

	// build indices
	FLANNPointcloud pcs[PYR_LEVELS];
	KDTree *indexes[PYR_LEVELS];
	for (int i = 0; i < PyrLevelsUsedG; i++)
	{
		pcs[i] = FLANNPointcloud(numPoints[i], points[i]);
		indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5));
		indexes[i]->buildIndex();
	}

	const int nn = 10;

	// find NN & parents
	for (int lvl = 0; lvl < PyrLevelsUsedG; lvl++)
	{
		Pnt *pts = points[lvl];
		int npts = numPoints[lvl];

		int ret_index[nn];
		float ret_dist[nn];
		nanoflann::KNNResultSet<float, int, int> resultSet(nn);
		nanoflann::KNNResultSet<float, int, int> resultSet1(1);

		for (int i = 0; i < npts; i++)
		{
			//resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
			resultSet.init(ret_index, ret_dist);
			Vec2f pt = Vec2f(pts[i].u, pts[i].v);
			indexes[lvl]->findNeighbors(resultSet, (float *)&pt, nanoflann::SearchParams());
			int myidx = 0;
			float sumDF = 0;
			for (int k = 0; k < nn; k++)
			{
				pts[i].neighbours[myidx] = ret_index[k];
				float df = expf(-ret_dist[k] * NNDistFactor); 
				sumDF += df;
				pts[i].neighboursDist[myidx] = df;
				assert(ret_index[k] >= 0 && ret_index[k] < npts);
				myidx++;
			}
			for (int k = 0; k < nn; k++)
				pts[i].neighboursDist[k] *= 10 / sumDF; 

			if (lvl < PyrLevelsUsedG - 1) 
			{
				resultSet1.init(ret_index, ret_dist);
				pt = pt * 0.5f - Vec2f(0.25f, 0.25f);
				indexes[lvl + 1]->findNeighbors(resultSet1, (float *)&pt, nanoflann::SearchParams());

				pts[i].parent = ret_index[0];
				pts[i].parentDist = expf(-ret_dist[0] * NNDistFactor);

				assert(ret_index[0] >= 0 && ret_index[0] < numPoints[lvl + 1]);
			}
			else
			{
				pts[i].parent = -1;
				pts[i].parentDist = -1;
			}
		}
	}

	// done.

	for (int i = 0; i < PyrLevelsUsedG; i++)
		delete indexes[i];
}

} // namespace SFRSS
