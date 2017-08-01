#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include "H5Cpp.h"
#include <cstdlib>
#include <sstream>
#include <opencv2/core/utility.hpp> 

using namespace H5;
using namespace std;
using namespace cv;
using namespace cv::cuda;

void calc_optical_flow(char* file1, char* file2, Mat& flowx, Mat& flowy, Size new_size)
{
    //read images
    
    Mat img1 = imread(file1, IMREAD_GRAYSCALE);
    Mat img2 = imread(file2, IMREAD_GRAYSCALE);
	
    resize(img1, img1, new_size);
	resize(img2, img2, new_size);
    Mat optFlow;
    //start time
    //const int64 start = getTickCount();
    //upload images to gpu device
    GpuMat gpu_img1(img1);
    GpuMat gpu_img2(img2);
    GpuMat gpu_flow(img1.size(), CV_32FC2);
    //declare methods of calculating optical flow
    Ptr<cuda::BroxOpticalFlow> brox = cuda::BroxOpticalFlow::create(0.197f,50.0f,0.8f,10,77,10);
    //Ptr<cuda::DensePyrLKOpticalFlow> lk = cuda::DensePyrLKOpticalFlow::create(Size(7, 7));
    //Ptr<cuda::FarnebackOpticalFlow> farn = cuda::FarnebackOpticalFlow::create();
    //Ptr<cuda::OpticalFlowDual_TVL1> tvl1 = cuda::OpticalFlowDual_TVL1::create();

    //calc optical flow map
    //Brox
    {
        GpuMat gpu_img1f;
        GpuMat gpu_img2f;
        gpu_img1.convertTo(gpu_img1f,CV_32F,1.0/255.0);
        gpu_img2.convertTo(gpu_img2f,CV_32F,1.0/255.0);
        
        const int64 start = getTickCount();
        brox->calc(gpu_img1f,gpu_img2f,gpu_flow);
        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << " time: " << timeSec << " sec" << endl;
    }

    GpuMat planes[2];
    cuda::split(gpu_flow, planes);

    planes[0].download(flowx);
    planes[1].download(flowy);

    //end time
    //const double timeSec = (getTickCount() - start) / getTickFrequency();
   // cout << "Brox: " << timeSec << " seconds" << endl;
}



int main(int argc,char*argv[])
{
	if (argc < 4) {
        cerr << "the number of arguments must be 4, exe file, down, train_type, gpu_id" << endl;
        exit(1);
    }

    int down = atoi(argv[1]);
    string down_s(argv[1]);
    string data_type(argv[2]);  // "train" or "val"
    int gpu_id = atoi(argv[3]);

    setDevice(gpu_id);
    
    //dataset path
    string dataset_path = "/media/data/bingbing/";
    string list_file = dataset_path + "leftImg8bit_sequence/" + data_type + "Frame.lst";
    string save_path = dataset_path + "opticalFlow/opencv/down-" + down_s + "/" + data_type + "/";

    Mat flowx, flowy;
    int new_size[] = {1024/down, 2048/down}; // {height, width}
    Size cv_size(new_size[1], new_size[0]);   // Size(_width, _height)
    

    // basic hdf5 configuration 
    const H5std_string DATASET_NAME("optical_flow");
	hsize_t fdim[4] = {1, 2, new_size[0], new_size[1]};
	hsize_t mdim[4] = {1, 1, new_size[0], new_size[1]};
	DataSpace fspace(4, fdim);
	DataSpace mspace(4, mdim);
	hsize_t start[4];
	hsize_t stride[4];
	hsize_t count[4];
	hsize_t block[4];
	start[0] = 0; start[1] = 0; start[2] = 0; start[3] = 0;
	stride[0] = 1; stride[1] = 1; stride[2] = 1; stride[3] = 1;
	count[0] = 1; count[1] = 1; count[2] = 1; count[3] = 1;
	block[0] = 1; block[1] = 1; block[2] = new_size[0]; block[3] = new_size[1];
	mspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
    

    ifstream fin(list_file.c_str());
    string str;
    int cnt = 0;
    while (getline(fin, str)) {
        istringstream sstr(str);
        string prev, cur;
        sstr >> prev;
        sstr >> cur;
        cout << "flow: " << cnt++;
        
        // one h5 file for each flow
        string tmp(cur);
        size_t pos = tmp.find_first_of('/');
        tmp.erase(0, pos+1);
        
        pos = tmp.find_first_of('.');
        tmp.erase(pos, 4);
        //tmp.erase(tmp.end()-4, tmp.end());
        const H5std_string FILE_NAME(save_path + tmp + "Flow.h5"); // ****Flow.h5
        H5File file(FILE_NAME, H5F_ACC_TRUNC);
        DataSet dataset = file.createDataSet(DATASET_NAME, PredType::NATIVE_FLOAT, fspace);

        // calculate optical flow        
        string prev_file = dataset_path + "leftImg8bit_sequence/" + data_type + "/" + prev;
        string cur_file = dataset_path + "leftImg8bit_sequence/" + data_type + "/" + cur;
        calc_optical_flow((char *)prev_file.c_str(), (char *)cur_file.c_str(), flowx, flowy, cv_size);
		start[1] = 0;
        fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
		dataset.write((void*)flowx.ptr(), PredType::NATIVE_FLOAT, mspace, fspace);
		fspace.selectNone();
        start[1] = 1;
		fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
		dataset.write((void*)flowy.ptr(), PredType::NATIVE_FLOAT, mspace, fspace);
        fspace.selectNone();
        //break;
    }

    return 0;
}

