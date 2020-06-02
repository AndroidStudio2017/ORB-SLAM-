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


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(argv[3])+"/rgb.txt";                // rgb.txt中格式应该为 timestamp imageFileName
    LoadImages(strFile, vstrImageFilenames, vTimestamps);       // 将rgb.txt中列出的时间戳和文件名分别保存到向量中

    int nImages = vstrImageFilenames.size();                    // 图像的总数量

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);  // 创建SLAM系统，并初始化所有进程

    // Vector for tracking time statistics
    vector<float> vTimesTrack;                                      // track时间统计，用于显示信息
    vTimesTrack.resize(nImages);                                    // track时间统计向量长度设置为图像总数量

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        im = cv::imread(string(argv[3])+"/"+vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);   // 读取图片，IMAGE_UNCHANGED
        double tframe = vTimestamps[ni];                // 读取对应图片的时间戳

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();  // 记录时间
#endif

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im,tframe);     // 传递给SLAM系统进行track

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();  // 记录时间
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count(); // 记录本次track时间

        vTimesTrack[ni]=ttrack;     // 把track时间加入到vector中用于统计

        // Wait to load the next frame
        double T=0;                         // 等待加载下一帧，如果当前帧不是最后一帧，等待时间T = 下一帧时间戳 - 当前帧时间戳
                                            // 如果当前帧是最后一帧，等待时间T = 当前帧时间戳 - 上一帧时间戳
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        // 如果track时间小于等待时间，那么就等待剩余时间
        // 估计是为了保证模拟的和真实情况一致
        if(ttrack<T)                        
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();        // 关掉SLAM系统中的进程

    // Tracking time statistics
    // track时间统计
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;         // 中位数track时间
    cout << "mean tracking time: " << totaltime/nImages << endl;                // 平均track时间

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    // 遍历rgb.txt中的行
    // 将时间戳和图像文件名分别保存到 vTimestamps 和 vstrImageFilenames 中
    // 前者是vector<double>，后者是vector<string>
    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;                   // 时间戳
            string sRGB;                // 图像文件名
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}
