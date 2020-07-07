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

#include "Initializer.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include "Optimizer.h"
#include "ORBmatcher.h"

#include<thread>

namespace ORB_SLAM2
{

Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
{
    mK = ReferenceFrame.mK.clone();

    mvKeys1 = ReferenceFrame.mvKeysUn;

    mSigma = sigma;
    mSigma2 = sigma*sigma;
    mMaxIterations = iterations;
}

// 
bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                             vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
{
    // Fill structures with current keypoints and matches with reference frame
    // Reference Frame: 1, Current Frame: 2
    // mvKeys2保存当前帧的去畸变特征点
    mvKeys2 = CurrentFrame.mvKeysUn;

    // 
    mvMatches12.clear();
    mvMatches12.reserve(mvKeys2.size());
    mvbMatched1.resize(mvKeys1.size());

    // 遍历初始化匹配中求出的初始帧和当前帧的匹配关系，vMatches12[i]保存初始帧第i个特征点匹配的当前帧特征点的索引
    for(size_t i=0, iend=vMatches12.size();i<iend; i++)
    {
        // 如果初始帧第i个特征点在当前帧中有匹配
        if(vMatches12[i]>=0)
        {   
            // 将匹配关系放到mvMatches12中，mvMatches12保存一个pair
            // 第一个元素为一个初始帧特征点索引，第二个元素为初始帧特征点匹配的当前帧特征点索引
            mvMatches12.push_back(make_pair(i,vMatches12[i]));
            // 标记初始帧第i个元素是否有匹配
            mvbMatched1[i]=true;
        }
        else
            mvbMatched1[i]=false;
    }

    // N记录有匹配的特征点对数
    const int N = mvMatches12.size();

    // Indices for minimum set selection
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for(int i=0; i<N; i++)
    {
        // vAllIndices[i]就保存的是i
        vAllIndices.push_back(i);
    }

    // Generate sets of 8 points for each RANSAC iteration
    // mvSets保存所有的8点对数
    mvSets = vector< vector<size_t> >(mMaxIterations,     // mMaxIterations默认为200，即迭代200次，随机生成200个8点对
                                    vector<size_t>(8,0)); // 一开始都初始化8点对为0

    DUtils::Random::SeedRandOnce(0);        // 用于随机取数的种子设置

    // 迭代构成8点对集合
    for(int it=0; it<mMaxIterations; it++)
    {
        // 每次迭代将可用的点对数重新设置为所有匹配点对
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        // 选择8对点
        for(size_t j=0; j<8; j++)
        {
            // 在可用点对长度内随机选择一个数
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);

            // 拿到选择的这个数的索引
            // 因为在每次迭代中，选择完的点对索引会被删掉，所以虽然一开始vAvailableIndices中的下标和元素是相等的
            // 但有些点对被删除之后，就不相等了
            int idx = vAvailableIndices[randi];

            // 将上面取到的点对对应索引放到集合中
            mvSets[it][j] = idx;

            // 其实就是在可用点对中删除了刚刚选到的点对
            vAvailableIndices[randi] = vAvailableIndices.back();    // 把刚才选到点对的位置替换为最后元素
            vAvailableIndices.pop_back();                           // 删除最后元素（其实刚才被选到的点对索引已经没了）
        }
    }

    // Launch threads to compute in parallel a fundamental matrix and a homography
    vector<bool> vbMatchesInliersH, vbMatchesInliersF;
    float SH, SF;
    cv::Mat H, F;

    // 开启两个线程分别计算H（单应矩阵）和F（基础矩阵）
    thread threadH(&Initializer::FindHomography,this,ref(vbMatchesInliersH), ref(SH), ref(H));
    thread threadF(&Initializer::FindFundamental,this,ref(vbMatchesInliersF), ref(SF), ref(F));

    // Wait until both threads have finished
    // 等待上面两个进程结束
    threadH.join();
    threadF.join();

    // Compute ratio of scores
    // 计算RH，然后要根据RH选择用单应矩阵来初始化还是选择基础矩阵来初始化
    float RH = SH/(SH+SF);

    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    // 如果RH > 0.4，那么选择用单应矩阵来进行初始化
    if(RH>0.40)
        return ReconstructH(vbMatchesInliersH,H,mK,R21,t21,vP3D,vbTriangulated,1.0,50);

    // 否则选择用基础矩阵来进行初始化
    else //if(pF_HF>0.6)
        return ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50);

    return false;
}

// 用来计算单应矩阵的函数
// 
// Input:
//      vbMatchesInliers:   vbMatchesInliers[i]为true，表示mvMatches12中索引为i的匹配点对为内点
//      score:              计算的単应矩阵对应的评分
//      H21:                计算的単应矩阵的结果
// Output:
//      通过输入参数返回计算得到的単应矩阵，以及単应矩阵的评分和对应的内点情况
void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
{
    // Number of putative matches
    // N 初始帧和当前帧的匹配点对数
    const int N = mvMatches12.size();

    // Normalize coordinates
    // 将初始帧和当前帧的特征点进行归一化，详细解释见笔记（十一）1.
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    // 归一化操作
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2inv = T2.inv();       // 将当前帧特征点求得的归一化矩阵求逆

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    // 这两个就是每次使用的一组8点对
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    // 
    cv::Mat H21i, H12i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    // 迭代所有8点对的结果，并且保存最好的一次
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        // 把8点对取出并赋值给vPn1i和vPn2i
        for(size_t j=0; j<8; j++)
        {
            // 取出集合中it保存的8点对
            int idx = mvSets[it][j];

            // 这里赋值的是归一化后的点坐标
            // mvMatches12[i]保存了一个pair，其中第一个元素是初始帧的特征点索引，第二个元素是与之匹配的当前帧特征点索引
            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        // 用8点法计算单应矩阵，原理见笔记（十一）2.3.
        // 这里求得的单应矩阵是vPn1i中的点乘单应矩阵可以转化为vPn2i中的点
        cv::Mat Hn = ComputeH21(vPn1i,vPn2i);

        // 因为上面求単应矩阵是用的归一化后的点求得，但是我们想要的是原特征点之间的単应矩阵关系
        // vPn2i = Hn * vPn1i
        // vPn2i = T2 * mvKeys2, vPn1i = T1 * mvKeys1
        // (T2 * mvKeys2) = Hn * (T1 * mvKeys1)
        // mvKeys2 = T2inv * Hn * T1 * mvKeys1
        // 所以我们需要的単应矩阵 H = T2inv * Hn * T1
        H21i = T2inv*Hn*T1;
        H12i = H21i.inv();      // 求一下単应矩阵的逆，方便后面评分

        // 对RANSAC的每一组点对求得的単应矩阵进行评分
        currentScore = CheckHomography(H21i, H12i,          // 利用8点对求得的単应矩阵
                                    vbCurrentInliers,       // 用于输出当前単应矩阵对应的内点,
                                                            // vbCurrentInliers[i]为true，证明第i对匹配点为内点
                                    mSigma);                // 卡方检验的权重

        // 因为是要用RANSAC找到最好的単应矩阵，所以这里我们要保存最好的评分对应的単应矩阵和其对应的内点
        if(currentScore>score)
        {
            // 保存最好的単应矩阵
            H21 = H21i.clone();
            // 保存内点情况
            vbMatchesInliers = vbCurrentInliers;
            // 用于找到最大值分数
            score = currentScore;
        }
    }
}

// 计算初始帧和当前帧之间的基础矩阵
// 
// Input:
//      vbMatchesInliers:       vbMatchesInliers[i]为true，表示vbMatches12索引为i的匹配点为内点
//      score:                  求得的基础矩阵对应的卡方检验的评分
//      F21:                    求得的基础矩阵
// Output:
//      通过输入参数返回求得的基础矩阵，以及评分和内点情况
void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    // Number of putative matches
    // ！！我觉得这里有点问题，vbMatchesInliers并没有初始化，所以size应该是0
    // 这里我想应该和単应矩阵那里一样，是mvMatches12的size
    const int N = vbMatchesInliers.size();

    // Normalize coordinates
    // 对特征点坐标进行归一化
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);

    // 因为是基础矩阵，所以归一化后恢复和単应矩阵有所不同，需要用到T2的转职
    cv::Mat T2t = T2.t();

    // Best Results variables
    // 初始化存储最好分数和内点情况的结果变量
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);       // 初始帧中的8个点
    vector<cv::Point2f> vPn2i(8);       // 当前帧中的8个点
    cv::Mat F21i;                       // 求得基础矩阵的临时变量
    vector<bool> vbCurrentInliers(N,false);     // 求得内点情况的临时变量
    float currentScore;                 // 球儿的当前分数的临时变量

    // Perform all RANSAC iterations and save the solution with highest score
    // 遍历每个8点对
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        // 获取到8点对，不过是归一化坐标
        for(int j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        // 根据8点对求得基础矩阵
        cv::Mat Fn = ComputeF21(vPn1i,vPn2i);

        // 与単应矩阵同理，因为我们上面求得的Fn是特征点归一化后的基础矩阵，所以需要进行转换
        // vPn2i^t * Fn * vPn1i = 0
        // vPn2i = T2 * mvKeys2, vPn1i = T1 * mvKeys1
        // (T2 * mvKeys2)^t * Fn * (T1 * mvKeys1) = 0
        // mvKeys2^t * T2^t * Fn * T1 * mvKeys1 = 0
        // 所以实际特征点之间的基础矩阵为 T2^t * Fn * T1
        F21i = T2t*Fn*T1;

        // 利用卡方检验对求得的基础矩阵进行评分，并且标记内点和外点
        currentScore = CheckFundamental(F21i,           // 求得的基础矩阵
                                    vbCurrentInliers,   // 记录当前基础矩阵对应的内点情况
                                    mSigma);            // 卡方检验用到的权值

        // 记录评分最高的基础矩阵及其对应的内点情况
        if(currentScore>score)
        {
            // 保存评分最高的基础矩阵
            F21 = F21i.clone();
            // 该基础矩阵对应的内点情况
            vbMatchesInliers = vbCurrentInliers;
            // 用于计算最高分
            score = currentScore;
        }
    }
}

// 8点法计算单应矩阵,用DLT方法，求最小二乘解
// 
// Input:
//      vP1:    初始帧中归一化后的8个点
//      vP2:    与初始帧中对应的，当前帧中归一化后的8个点
// Output:
//      返回值返回单应矩阵H21,是p2 = H21 * p1
cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    // N 点对的数目，应该用4个点对就可以求，但是这里面用了8个
    const int N = vP1.size();

    // 建立2N×9的系数矩阵，对应于单应矩阵的原理
    cv::Mat A(2*N,9,CV_32F);

    // 按照原理推算，给系数矩阵赋值
    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;

    }

    // 求解SVD
    cv::Mat u,w,vt;

    cv::SVDecomp(A,         // 待SVD分解的系数矩阵
                w,          // 分解后的奇异值矩阵
                u,          // 分解后的左正交阵
                vt,         // 分解后的右正交阵
                cv::SVD::MODIFY_A | cv::SVD::FULL_UV);      // OpenCV SVD函数的选项（具体查看文档）

    // SVD结果中，vt的第9行即为我们所求的单应矩阵，原因见笔记（十一）3.
    // 求出来一行是9个值，我们把它reshape成3×3的矩阵
    return vt.row(8).reshape(0, 3);
}

// 利用8点对求得基础矩阵，原理 p2^t * F21 * p1 = 0
// l2 = F21 * p1，可以根据p1点和基础矩阵求得在image2中对应的极线
//
// Input:
//      vP1:    Image1中的8个特征点
//      vP2:    Image2中的8个特征点
// Output:
//      返回求得的基础矩阵
cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
{
    // N 所使用的特征点对的数量
    const int N = vP1.size();

    // A为系数矩阵
    cv::Mat A(N,9,CV_32F);

    // 根据基础矩阵的原理构造其系数矩阵
    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    cv::Mat u,w,vt;

    // 对系数矩阵进行SVD分解，从而求得基础矩阵
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    // 与之前的単应矩阵同理，求得的最后一个特征向量的误差最小，即最优值
    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    // 但由于基础矩阵一个重要的性质是其秩为2，所以我们需要再次对上面求得的Fpre进行SVD分解
    // 并且将w的最后一个特征值置为0，使其秩为2，然后再重新构建更精确的基础矩阵
    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    // 将w的最后一个特征值置为2
    w.at<float>(2)=0;

    // 返回重新构建的更加准确的基础矩阵
    return  u*cv::Mat::diag(w)*vt;
}

// 利用卡方检验对求得的単应矩阵评分
// 
// Input: 
//      H21:        p2 = H21 * p1
//      H12:        p1 = H12 * p2
//      vbMatchesInliers:       vbMatchesInliers[i]为true，表示mvMatches12中索引为i的那对匹配点是内点
//      sigma:      卡方检验用到的权重
// Output:
//      返回评分结果，通过vbMatchesInliers返回这个単应矩阵所对应的内点   
// 原理：
// 通过求取|| p2 - H21 * p1 ||来得到误差，然后通过卡方检验来对误差进行评分
// 同时也要求取反方向的评分，即|| p1 - H12 * p2 ||
float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
{   
    // N 当前帧和初始帧所有匹配的点对数
    const int N = mvMatches12.size();

    // 将単应矩阵H21中的值取出来
    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

    // 将単应矩阵H12中的值取出来
    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);

    // 预分配内存
    vbMatchesInliers.resize(N);

    // 记录得分，要累加两个方向的得分
    float score = 0;

    // 这是自由度为2的卡方分布，显著性水平为0.05的临界阈值
    const float th = 5.991;

    // 权重
    const float invSigmaSquare = 1.0/(sigma*sigma);

    // 对于所有匹配点，累计误差评分
    for(int i=0; i<N; i++)
    {
        // 一开始假设是内点
        bool bIn = true;

        // 获取匹配点的坐标
        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        // 初始帧特征点坐标(u1, v1)
        // 当前帧特征点坐标(u2, v2)
        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in first image
        // x2in1 = H12*x2

        // p1 = H12 * p2
        // |u1|   |h11 h12 h13||u2|   |u2in1|
        // |v1| = |h21 h22 h23||v2| = |v2in1| * w2in1
        // |1 |   |h31 h32 h33||1 |   |1    |
        // 其实这段代码就是在求当前帧特征点在初始帧上的投影，即p2在Image1上的投影
        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        // 重投影误差的平方
        // (p1 - H12 * p2)^2
        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

        // 乘权重转换为卡方的误差
        const float chiSquare1 = squareDist1*invSigmaSquare;

        // 如果误差大于阈值，则标记为外点
        if(chiSquare1>th)
            bIn = false;
        else
            // 如果误差小于等于阈值，则可以接受，计算评分
            // chiSquare1代表了误差，误差越大，则chiSquare1越接近th，则评分越小
            // 这里是累加，用所有点的评分和来选择好的単应矩阵
            score += th - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1

        // 下面的过程和上面一样，只不过是在算反方向，即|| p2 - H21 * p1 ||
        // p1在image2上的重投影误差
        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        // 计算p1到image2上的重投影误差
        // 即初始帧特征点到当前帧的重投影误差
        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

        // 乘卡方检验的权重
        const float chiSquare2 = squareDist2*invSigmaSquare;

        // 如果误差大于阈值，则标记为外点
        if(chiSquare2>th)
            bIn = false;
        else
            // 如果误差小于等于阈值，则计算分数
            // chiSquare2代表了误差，误差越大，则chiSquare2越接近th，则评分越小
            // 这里是累加，用所有点的评分和来选择好的単应矩阵
            score += th - chiSquare2;

        // 利用vbMatchesInliers来记录当前単应矩阵对应的所有内点和外点
        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    // 返回卡方检验计算的分数
    return score;
}

// 对计算得到的基础矩阵，利用卡方检验进行评分，并且计算其对应的内点情况
// 
// Input:
//      F21:    求得的基础矩阵      
//      vbMatchesInliers:   用于反映该基础矩阵对应的内点情况
//      sigma:  卡方检验用到的权值
// Output:
//      返回该基础矩阵的评分，并且通过输入参数返回内点情况
float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    // N 所有两帧之间的匹配点个数
    const int N = mvMatches12.size();

    // 提取基础矩阵中的值
    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    // 预分配内存
    vbMatchesInliers.resize(N);

    // 用于累加分数
    float score = 0;

    // 这里设定了两个阈值，一个是用于判断内外点，另一个是用于计算分数
    // 3.841是自由度为1的卡方分布，显著性水平为0.05的临界阈值
    const float th = 3.841;

    // 这里又使用了这个阈值来计算分数，应该是为了保证和単应矩阵分数计算的一致性
    const float thScore = 5.991;

    // 卡方检验用到的权值
    const float invSigmaSquare = 1.0/(sigma*sigma);

    // 遍历所有的匹配点对
    for(int i=0; i<N; i++)
    {
        // 开始认为都是内点
        bool bIn = true;

        // 获取匹配点的两个特征点
        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        // 初始帧特征点(u1, v1)
        // 当前帧特征点(u2, v2)
        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)

        // 这里通过计算点到直线距离实现误差的测量

        // 利用基础矩阵计算极线的系数
        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        // 点到直线距离公式
        const float num2 = a2*u2+b2*v2+c2;

        const float squareDist1 = num2*num2/(a2*a2+b2*b2);

        // 误差乘卡方检验需要的权值
        const float chiSquare1 = squareDist1*invSigmaSquare;

        // 如果误差大于阈值，则标记为外点
        if(chiSquare1>th)
            bIn = false;
        else
            // 如果误差小于等于权值，为了和単应矩阵对比保持分数统一，所以使用了thScore这个阈值
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)

        // 这里同样求反方向的误差
        // 因为基础矩阵原理：p2^t * F21 * p1 = 0
        // 则：p1^t * F21^t * p2 = 0
        // 所以反方向的基础矩阵直接转置就可以

        // 利用基础矩阵求得极线系数，注意与上面求的是相反的，这里的F使用了上面F的转置
        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        // 点到直线的距离公式
        const float num1 = a1*u1+b1*v1+c1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        // 将误差乘卡方检验需要的权值
        const float chiSquare2 = squareDist2*invSigmaSquare;

        // 如果误差大于阈值，那么标记为外点
        if(chiSquare2>th)
            bIn = false;
        else
            // 如果误差小于等于阈值，则计算分数
            score += thScore - chiSquare2;

        // 利用vbMatchesInliers来记录当前基础矩阵对应的所有内点和外点
        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    // 返回基础矩阵的评分
    return score;
}

bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // Compute Essential Matrix from Fundamental Matrix
    cv::Mat E21 = K.t()*F21*K;

    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses
    DecomposeE(E21,R1,R2,t);  

    cv::Mat t1=t;
    cv::Mat t2=-t;

    // Reconstruct with the 4 hyphoteses and check
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
    float parallax1,parallax2, parallax3, parallax4;

    int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();

    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);

    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    if(maxGood<nMinGood || nsimilar>1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    if(maxGood==nGood1)
    {
        if(parallax1>minParallax)
        {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
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
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988

    cv::Mat invK = K.inv();
    cv::Mat A = invK*H21*K;

    cv::Mat U,w,Vt,V;
    cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
    V=Vt.t();

    float s = cv::determinant(U)*cv::determinant(Vt);

    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);

    if(d1/d2<1.00001 || d2/d3<1.00001)
    {
        return false;
    }

    vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};

    //case d'=d2
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=ctheta;
        Rp.at<float>(0,2)=-stheta[i];
        Rp.at<float>(2,0)=stheta[i];
        Rp.at<float>(2,2)=ctheta;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=-x3[i];
        tp*=d1-d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    //case d'=-d2
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=cphi;
        Rp.at<float>(0,2)=sphi[i];
        Rp.at<float>(1,1)=-1;
        Rp.at<float>(2,0)=sphi[i];
        Rp.at<float>(2,2)=-cphi;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=x3[i];
        tp*=d1+d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }


    int bestGood = 0;
    int secondBestGood = 0;    
    int bestSolutionIdx = -1;
    float bestParallax = -1;
    vector<cv::Point3f> bestP3D;
    vector<bool> bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    for(size_t i=0; i<8; i++)
    {
        float parallaxi;
        vector<cv::Point3f> vP3Di;
        vector<bool> vbTriangulatedi;
        int nGood = CheckRT(vR[i],vt[i],mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K,vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);

        if(nGood>bestGood)
        {
            secondBestGood = bestGood;
            bestGood = nGood;
            bestSolutionIdx = i;
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        else if(nGood>secondBestGood)
        {
            secondBestGood = nGood;
        }
    }


    if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
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
    cv::Mat A(4,4,CV_32F);

    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}

// 归一化函数，原理见笔记（十一）1.
//
// Input:
//      vKeys:              特征点向量
//      vNormalizedPoints:  归一化特征点的结果
//      T:                  归一化矩阵（后面复原要用到）
// Output:
//      输出由vNormalizedPoints和T带出，就是归一化特征点的结果和归一化矩阵
void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    // 用来保存X,Y方向均值
    float meanX = 0;
    float meanY = 0;
    // 保存特征点数量
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    // 遍历所有特征点，为了求均值
    for(int i=0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    // 先求和再除以数目求均值
    meanX = meanX/N;
    meanY = meanY/N;

    // 用来保存 Σ|x - meanx| / N
    float meanDevX = 0;
    float meanDevY = 0;

    // 遍历所有特征点，求|x - meanx|的和
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    // 求得Σ|x - meanx| / N
    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    // 因为后面要进行除法，所以先取倒数，后面执行乘法
    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    // 遍历所有特征点，得到归一化结果
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    // 这里其实是上面一样的过程，只不过是变成了矩阵的形式，其实上面的过程就是 x' = Tx, y' = Ty
    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}


int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    // Calibration parameters
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    vbGood = vector<bool>(vKeys1.size(),false);
    vP3D.resize(vKeys1.size());

    vector<float> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K[I|0]
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3));

    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;

    cv::Mat O2 = -R.t()*t;

    int nGood=0;

    for(size_t i=0, iend=vMatches12.size();i<iend;i++)
    {
        if(!vbMatchesInliers[i])
            continue;

        const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
        const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
        cv::Mat p3dC1;

        Triangulate(kp1,kp2,P1,P2,p3dC1);

        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[vMatches12[i].first]=false;
            continue;
        }

        // Check parallax
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);

        float cosParallax = normal1.dot(normal2)/(dist1*dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R*p3dC1+t;

        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check reprojection error in first image
        float im1x, im1y;
        float invZ1 = 1.0/p3dC1.at<float>(2);
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);

        if(squareError1>th2)
            continue;

        // Check reprojection error in second image
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2.at<float>(2);
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

        if(squareError2>th2)
            continue;

        vCosParallax.push_back(cosParallax);
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
        nGood++;

        if(cosParallax<0.99998)
            vbGood[vMatches12[i].first]=true;
    }

    if(nGood>0)
    {
        sort(vCosParallax.begin(),vCosParallax.end());

        size_t idx = min(50,int(vCosParallax.size()-1));
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;

    return nGood;
}

void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    cv::Mat u,w,vt;
    cv::SVD::compute(E,w,u,vt);

    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;
    if(cv::determinant(R1)<0)
        R1=-R1;

    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}

} //namespace ORB_SLAM
