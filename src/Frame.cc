/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}

// 双目SLAM中的Frame帧构造函数
// 
// Input:
//      imLeft:             左目图像
//      imRight:            右目图像
//      timeStamp:          时间戳
//      extractorLeft:      左目图像的ORB特征提取器
//      extractorRight:     右目图像的ORB特征提取器
//      voc:                Vocabulary
//      distCoef:           畸变参数
//      bf:                 基线长度
//      thDepth:            深度阈值
// Output:
//      构造函数，无返回值
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    // 同单目一样，总帧数加一，nNextId是一个静态成员变量
    mnId=nNextId++;

    // Scale Level Info
    // 同单目一样，设定一些图像金字塔的参数
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    // 这里为了加速，开启两个线程分别计算左目和右目的ORB特征
    thread threadLeft(&Frame::ExtractORB,           // 线程函数，就是之前单目用到的ExtractORB
                    this,                           // 该类的函数
                    0,                              // 0 是ExtractORB的参数，表示提取左目图像特征
                    imLeft);                        // 也是ExtractORB参数，表示左目图像
    thread threadRight(&Frame::ExtractORB,this,1,imRight);

    // 这里的join()方法，需要主线程等待threadLeft执行完才可以向下进行
    threadLeft.join();
    // 同理
    threadRight.join();

    // mvKeys保存了左图提取的特征点，N是左图特征点的数量
    N = mvKeys.size();

    // 如果特征点数量为0，则返回
    if(mvKeys.empty())
        return;

    // 对左图提取的特征进行去畸变操作
    UndistortKeyPoints();

    // 计算左右图的双目立体匹配，与单目帧的主要区别
    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
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

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

// 单目帧的构造函数
// 
// Input:
//      imGray:         待构造帧对象的灰度图像
//      timeStamp:      时间戳
//      extractor:      该帧采用的特征提取器
//      voc:            该帧使用的词典
//      K:              内参矩阵
//      distCoef:       畸变系数矩阵
//      bf:             基线        （对于单目SLAM没用）
//      thDepth:        深度阈值    （对于单目SLAM没用）
// Output:
//      None
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    // 此处nNextId是一个静态成员变量，用于保存全局的帧的个数，或者说下一个帧的ID
    mnId=nNextId++;

    // Scale Level Info
    // 从提取器中获取图像金字塔的信息，这些都是提取器构造函数中设置好的
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    // 提取ORB特征
    // 0-左图， 1-右图，由于是单目，所以只有左图
    ExtractORB(0,imGray);

    // N记录了该帧提取的特征点个数
    N = mvKeys.size();

    // 如果没有提取到特征点，那么返回
    if(mvKeys.empty())
        return;

    // 对提取的特征点坐标进行去畸变的操作
    UndistortKeyPoints();

    // Set no stereo information
    // 因为这是针对单目相机的帧构造函数，所以没有右图和深度图
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    // 预留的地图点空间, N
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    // 预留的外点空间, N， 用来标记某个点是否是外点
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    // 这部分代码仅仅在第一帧的时候被使用（或者在内参改变之后）
    if(mbInitialComputations)
    {
        // 计算畸变后图像的边界，例子见笔记（七） 2
        ComputeImageBounds(imGray);

        // 这里计算的是一个像素对应的grid的列数和行数是多少
        // 相当于对一个grid列和行占的像素数做了一个倒数
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        // 把相机的内参矩阵参数提取出来，提高计算的效率
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        // 在运行完这段代码的时候，将标记变量设置为false，保证只有在第一帧以及内参改变之后运行这段代码
        mbInitialComputations=false;
    }

    // 计算baseline，实际在单目中没有用到
    mb = mbf/fx;

    // 将去畸变后的特征点放到去畸变后的grid中
    AssignFeaturesToGrid();
}

// 将去畸变后的特征点，放到去畸变后的grid中
// 
// Input:
//      成员变量, mvKeysUn，去畸变后的特征点的vector
// Output:
//      成员变量, mGrid，保存不同位置网格中包含的特征点的索引
void Frame::AssignFeaturesToGrid()
{
    // 为每一个网格预分配需要保存的特征点数量
    // 这里乘0.5，应该是为了节省分配的空间，因为有好多实际用不了那么多空间
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);  // 为每个网格预分配空间

    // 将去畸变后的特征点分配到去畸变后的网格中
    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        // 如果kp在nGridPosX, nGridPosY中，那么就将kp的索引加入mGrid[nGridPosX][nGridPosY]
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

// 提取ORB特征，保存到该帧对应的成员变量mvKeys和mDescriptors中
// 
// Input:
//      flag:           标记，为0表示是左图，为1表示是右图
//      im:             输入的特征提取的图像
// Output:
//      这里的输出通过mvKeys和mDescriptors返回
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

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

// 找到帧中从minLevel到maxLevel图像金字塔中，以(x,y)为中心，r为half_patch内的所有特征点
// 
// Input:
//      x:          窗口中心的x坐标
//      y:          窗口中心的y坐标
//      r:          窗口的half_patch的大小
//      minLevel:   包括的最小金字塔层数
//      maxLevel:   包括的最大金字塔层数
// Output:
//      返回帧中从minLevel到maxLevel图像金字塔中，以(x,y)为中心，r为half_patch内的所有特征点
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    // 返回值
    vector<size_t> vIndices;
    vIndices.reserve(N);        // N是该帧所有的特征点数目，预留了足够大的空间

    // 关于函数原理可以参见笔记（九）2

    // mfGridElementWidthInv = FRAME_GRID_COLS / (mnMaxX-mnMinX)，表示的是一个像素所占据的Grid列数
    // 这里 x - mnMinX - r其实计算的是窗口左边界到图像左边界的像素数，然后与mfGridElementWidthInv相乘的到Grid的列数
    // 这里与0求最大值，表明窗口最左侧是0
    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    // 保证窗口左边界小于Grid最大列数
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    // 与上面相同，求得是窗口右边界
    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    // 窗口上边界
    // mfGridElementHeightInv = FRAME_GRID_ROWS/ (mnMaxY-mnMinY)，与上面一样
    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    // 窗口下边界
    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    // ？？bug：视频说这个是bug，但我感觉也不是，只不过没有明白为什么要这么写？
    // 可以看一下，或运算是只两个之一满足或者都满足
    // 满足第一个条件，则后面调用进入判断后，会判断第一个判断语句，跳过第二个
    // 满足第二个条件，则后面调用进入判断后，会判断第二个判断语句，跳过第一个
    // 
    // 这样的好处可能也是为了加速，因为我如果bCheckLevels为false的话，下面就都不需要判断了
    // 如果不用这个bool变量的话，那么无论怎样都要经过两个判断语句
    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    // 对窗口内得网格进行遍历
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            // 获取网格
            const vector<size_t> vCell = mGrid[ix][iy];

            // 如果网格中得特征点为空，那么跳过
            if(vCell.empty())
                continue;

            // 如果网格中有特征点，那么遍历这些特征点，将满足要求得特征点索引加入结果
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                // 获取去畸变特征点kpUn
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];

                // 对应上面bCheckLevels定义处的解释
                if(bCheckLevels)
                {
                    // 如果该特征点所在金字塔层数小于要求的最小层数，跳过
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        // 如果该特征点所在金字塔层数大于要求的最大层数，跳过
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                // 如果满足要求，则计算该特征点x方向和y方向距窗口中心的距离
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                // 如果距离均小于r，则将该特征点索引加入结果
                // 实际窗口应该是一个正方形（而视频说的是圆形）
                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    // 返回结果
    return vIndices;
}

// 计算kp是否在posX，posY处的网格中，posX, posY是计算出的网格索引，如果在，返回true，否则返回false
// 
// Input:
//      kp:         特征点对象
//      posX, posY: 根据特征点对象计算出的应该属于的网格的位置坐标
// Output:
//      计算出网格坐标如果正常，则返回true，否则返回false
bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    // mfGridElementWidthInv是之前计算出的每一个像素对应的网格列数，所以这里求的是对应列的索引
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    // 与上面相同
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    // 如果网格列数的索引或者行数的索引超过了最大值或小于0，则是不正常的，返回false
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    // 如果正常，返回true
    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

// 对特征点进行去畸变的操作，去畸变后的特征点结果存储在mvKeysUn中
// 
// Input: 
//      成员变量 mvKeys，提取的图像特征点
//      成员变量 mDistCoeft，畸变参数
//      成员变量 mK，内参矩阵
//      成员变量 N，图像特征点数目
// Output:
//      成员变量 mvKeysUn，坐标去畸变后的特征点
void Frame::UndistortKeyPoints()
{
    // 如果畸变参数中的第一个参数为0，即k1为0，那么则不用进行去畸变操作
    // [k1, k2, p1, p2, k3]
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    // 为了调用openCV中的 undistortPoints 函数，将坐标处理为2通道矩阵的形式
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    // mat.reshape(channels, rows)
    mat=mat.reshape(2);         // 这步是在把矩阵变成2通道的形式，这是undistortPoints函数要求的
                                // 函数要求src格式必须为CV_64F2C或CV_32F2C
    cv::undistortPoints(mat,mat,        // 原图像和目标图像
                        mK,             // 原内参矩阵
                        mDistCoef,      // 畸变参数
                        cv::Mat(),      // 没有用，暂时置为空矩阵
                        mK);            // 去畸变后的内参矩阵
    mat=mat.reshape(1);         // 将矩阵恢复为1通道，方便后面处理

    // Fill undistorted keypoint vector
    // 将mat矩阵中去畸变的结果赋值给成员变量 mvKeysUn
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

// 计算去畸变后图像的边界，其实就是图像的四个顶点
// 
// Input:
//      imLeft:     原图像
// Output:
//      成员变量 mnMinX, mnMaxX, mnMinY, mnMaxY，即去畸变图像后的四个顶点
void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    // 如果畸变参数的第一个数不为0，那么要对四个顶点进行去畸变，找到去畸变后图像的四个顶点的坐标
    if(mDistCoef.at<float>(0)!=0.0)
    {
        // 与之前的去畸变步骤相同
        // 首先将去畸变的点组织成2通道矩阵的形式
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);         // 将矩阵变为2通道的形式
        // 利用opencv的undistortPoints对四个顶点的去畸变坐标进行计算
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);         // 恢复矩阵的通道数为1

        // 保存边界
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    // 如果畸变参数的第一个数为0，那么就不用对顶点进行变换了
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

// 进行双目帧构造时的双目立体匹配
// 
// Input:
//      均是利用成员变量进行操作
// Output:
//      输出也是到成员变量中
// 原理见笔记（八）中1
void Frame::ComputeStereoMatches()
{
    // mvuRight[iL]保存的是，左图中第i个特征点所对应的右图特征点的索引
    // mvDepth[iL]保存的是，左图中第i个特征点所对应的深度
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    // ORB特征描述子的距离阈值，用于进行粗匹配
    // TH_HIGH是100，TH_LOW是50，这里取了个平均
    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    // 取左图原图的行数，也就是高
    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    // vRowIndices即是在统计行特征点，vRowIndices[i]保存了右图中第i行所有特征点的列坐标，也就是x坐标
    // 比如 vRowIndices[0] = [1, 4, 5], 则说明右图第0行第1列有一个特征点，第0行第4列有一个特征点，第0行第5列有一个特征点
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    // 预分配空间
    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    // 右图提取特征点的总数，左图的特征点总数的成员变量N
    const int Nr = mvKeysRight.size();

    // 遍历右图所有特征点，统计行特征点
    for(int iR=0; iR<Nr; iR++)
    {
        // 取得右图中的特征点
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;         // 特征点的y坐标，也就是行坐标

        // 因为左右目图像经过了立体匹配，那么可能在y方向会有不准确的地方
        // 那么我们设定一个r，如果kpY处有特征点p，那么[kpY - r, kpY + r]，这些行，都有该特征点
        // r随着特征点金字塔层数的不同而不同，金字塔层数越高，不确定度越高，r越大
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];    
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        // 这里就是将[minr, maxr]，这些行都加入索引为iR的特征点
        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    // 这里不知道为啥将深度的最小值设置为基线长度？？？
    const float minZ = mb;
    // 视差的最小值肯定是0
    const float minD = 0;
    // 根据上面设置的最小深度，由视差公式得到最大视差，视差公式见笔记（八）2
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    // vDistIdx[iL].first是索引为iL的左图特征点匹配右图特征点对应的描述子距离
    // vDistIdx[IL].second是索引为iL的左图特征点匹配的右图特征点的索引
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);        // 预分配内存

    // 遍历左图特征点
    for(int iL=0; iL<N; iL++)
    {
        // 取得左图特征点
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;     // 所处金字塔层数
        const float &vL = kpL.pt.y;         // 左图特征点的y坐标，行坐标
        const float &uL = kpL.pt.x;         // 左图特征点的x坐标，列坐标

        // 刚才统计的行坐标点，就是右图中和左图特征点同行(vL)的所有特征点
        const vector<size_t> &vCandidates = vRowIndices[vL];

        // 如果没有同行的，那么就没有匹配点，下一个
        if(vCandidates.empty())
            continue;

        // 上面计算的最大视差，也就是确定枚举对比的理论范围
        // 右图的视差肯定在左边，所以最大的U就设定为uL-minD
        const float minU = uL-maxD;
        const float maxU = uL-minD;

        // 如果最大范围小于0，则没有
        if(maxU<0)
            continue;

        // 用于求取最小值，一开始设为TH_HIGH的意思就是，如果距离比这个还大，那肯定不可能
        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;    // 保存粗匹配的索引

        // 获得左图特征点对应的描述子
        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        // 遍历右图中候选的特征点
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            // 获取特征点索引以及特征点
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            // 如果左图特征点所在的金字塔层数和右图特征点所在的金字塔层数相差超过2，则被认为误差太大，舍弃
            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            // 右图特征点的x坐标，列坐标
            const float &uR = kpR.pt.x;

            // 如果特征点的x坐标，列坐标，超过理论的范围，那么舍弃
            // 如果符合理论的范围，则通过计算左右图特征点描述子之间的距离，选取最小的距离结果作为粗匹配的结果
            if(uR>=minU && uR<=maxU)
            {
                // 取出右图特征点对应的描述子
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);     // 计算左右图描述子之间的距离

                // 计算最大值，并保存最大值的索引，作为粗匹配结果
                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        // 精确匹配
        // 如果上面粗匹配得到的最好的匹配距离大于等于阈值，则舍弃
        // 否则进行接下来的处理
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;       // 上述粗匹配得到的特征点的x坐标，列坐标
            const float scaleFactor = mvInvScaleFactors[kpL.octave];    // 左图特征点对应的金字塔层比例因子的倒数
            const float scaleduL = round(kpL.pt.x*scaleFactor);     // 左图特征点x坐标乘比例因子，换算为该层金字塔图像的坐标
            const float scaledvL = round(kpL.pt.y*scaleFactor);     // 左图特征点y坐标乘比例因子，换算为该层金字塔图像的坐标
            const float scaleduR0 = round(uR0*scaleFactor);     // 将粗匹配得到的右图特征点的x坐标也换算为金字塔图像中的坐标

            // sliding window search
            // w是窗口边长的一半，实际窗口大小是2w × 2w
            const int w = 5;
            // 获取左图中图像金字塔该层图像，以左图特征点为中心，2w为边长的窗口
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);        // 转换格式

            // 进行归一化，去除亮度的影响，其实就是将窗口内所有像素减去一个左图特征点的像素值
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            // 同样是为了求最小距离设置的变量
            int bestDist = INT_MAX;
            int bestincR = 0;       // 最小距离对应的特征点索引

            // 窗口要滑动的长度是L，即开始的窗口是[scaleduR0-L-w, scaleduR0-L+w]，最终的窗口是[scaleduR0+L-w, scaleduR0+L+w]
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            // 这里iniu是不是写错了？？？ 应该是scaleduR0-L-w？？？
            // 其实就是为了限制下面取子矩阵的时候不越界
            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            // 如果会越界，就跳过
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            // 滑动窗口，找到最优的匹配点
            for(int incR=-L; incR<=+L; incR++)
            {
                // 获取滑动窗口，右图中的窗口
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                // 同样去除亮度的影响，和上述左图是一样的操作
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                // 这里是比较两个图像块的相似度，用sad计算
                float dist = cv::norm(IL,IR,cv::NORM_L1);
                // 同样是求取距离最小，并记录滑动的步数
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;  // 记录每个滑动步数对应的像素相似度（sad距离）
            }

            // 如果最小值出现在最左和最右，则舍弃
            // 这里也不知道为什么要舍弃这两种情况？？？？
            if(bestincR==-L || bestincR==L)
                continue;

            // 亚像素插值
            // 这部分见笔记（八）3后的公式
            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            // 应该也是要求，直观上来理解，亚像素插值这个结果应该小于1，亚像素就是不足一个像素
            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            // 将粗匹配的右图特征点位置加上精确计算后的窗口滑动步数，再加上亚像素匹配的结果
            // 然后恢复到原图的x坐标，列坐标
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            // 求视差
            float disparity = (uL-bestuR);

            // 如果视差又满足一开始我们设定的理论视差范围，那么则记录视差
            if(disparity>=minD && disparity<maxD)
            {
                // 如果视差小于0，那么限定视差为0.01
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                // 设置索引为iL的特征点深度
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;      // 设置匹配的特征点列坐标
                vDistIdx.push_back(pair<int,int>(bestDist,iL));     // 设置iL特征点匹配的最小sad距离，用于去除离群点
            }
        }
    }

    // 按照距离进行排序
    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;     // 找到中位数
    const float thDist = 1.5f*1.4f*median;                      // 设置距离阈值

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)        // 如果距离小于阈值，则无操作
            break;
        else                                // 如果距离大于等于阈值，则取消匹配关系
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
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

} //namespace ORB_SLAM
