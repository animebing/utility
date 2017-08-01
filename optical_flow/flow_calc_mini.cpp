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
#include <boost/filesystem.hpp>

using namespace H5;
using namespace std;
using namespace cv;
using namespace cv::cuda;
namespace fs = boost::filesystem;

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



int main(int argc,char**argv) {

    int gpu_id = atoi(argv[1]);

    setDevice(gpu_id);
    
    //dataset path
    string dataset_dir = "/media/data/bingbing/leftImg8bit_sequence/train";
    string city_list_name = dataset_dir + "/" + "city.lst";
    string opencv = "/media/data/bingbing/opencv";
    string opencv_flow = opencv + "/" + "opencv_flow";
    Mat flowx, flowy;
    int new_size[] = {1024/2, 2048/2}; // {height, width}
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
    

    ifstream f_city(city_list_name.c_str());
    string city;
    while (getline(f_city, city)) {
        
        string city_frame = dataset_dir + "/" + city + "/" + "frame.lst";
        
        string city_flow = opencv_flow + "/" + city;
        fs::path city_flow_path(city_flow);
        if (!(fs::exists(city_flow_path))) {
            if (fs::create_directory(city_flow_path)) {
                cout << city_flow << " Created Successfully" << endl;
            }
        }

        ifstream f_frame(city_frame);
        string line;
        int i = 0;
        while (getline(f_frame, line)) {

            istringstream sstr(line);
            string prev, cur;
            sstr >> prev;
            sstr >> cur;
        
            // one h5 file for each flow
            int idx_1 = i/10 + 1;
            int idx_2 = i%10 + 1;
            string h5file = city_flow + "/" + to_string(idx_1) + "_" + to_string(idx_2) + ".h5";
            const H5std_string FILE_NAME(h5file);
            H5File file(FILE_NAME, H5F_ACC_TRUNC);
            DataSet dataset = file.createDataSet(DATASET_NAME, PredType::NATIVE_FLOAT, fspace);

            // calculate optical flow        
            cout << "Wrting: " << h5file << endl;
            calc_optical_flow((char *)prev.c_str(), (char *)cur.c_str(), flowx, flowy, cv_size);
		    start[1] = 0;
            fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
		    dataset.write((void*)flowx.ptr(), PredType::NATIVE_FLOAT, mspace, fspace);
		    fspace.selectNone();
            start[1] = 1;
		    fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
		    dataset.write((void*)flowy.ptr(), PredType::NATIVE_FLOAT, mspace, fspace);
            fspace.selectNone();
            i++;
            //break;
        }
    }

    return 0;
}

