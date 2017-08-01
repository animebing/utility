#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
//#include "opencv2/cudaoptflow.hpp"
//#include "opencv2/cudaarithm.hpp"
//#include <dirent.h>
//#include <unistd.h>
#include <sys/stat.h>
#include "H5Cpp.h"
#include <cstdlib>
#include <sstream>
#include <opencv2/core/utility.hpp> 

using namespace H5;
using namespace std;
using namespace cv;
using namespace cv::cuda;

inline bool isFlowCorrect(Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static Vec3b computeColor(float fx, float fy)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float) CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    Vec3b pix;

    for (int b = 0; b < 3; b++)
    {
        const float col0 = colorWheel[k0][b] / 255.0f;
        const float col1 = colorWheel[k1][b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(255.0 * col);
    }

    return pix;
}

static void drawOpticalFlow(const Mat_<float>& flowx, const Mat_<float>& flowy, Mat& dst, float maxmotion = -1)
{
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flowx.rows; ++y)
        {
            for (int x = 0; x < flowx.cols; ++x)
            {
                Point2f u(flowx(y, x), flowy(y, x));

                if (!isFlowCorrect(u))
                    continue;

                maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

    for (int y = 0; y < flowx.rows; ++y)
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            Point2f u(flowx(y, x), flowy(y, x));

            if (isFlowCorrect(u))
                dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}

int main(int argc,char*argv[])
{
	if (argc < 3) {
        cerr << "the number of arguments must be 3, exe file, down, data_type" << endl;
        exit(1);
    }

    int down = atoi(argv[1]);
    string down_s(argv[1]);
    string data_type(argv[2]);  // "train" or "val"
    
    //dataset path
    string dataset_path = "/media/data/bingbing/";
    string list_file = dataset_path + "opticalFlow/spynet/down-" + down_s+ "/" + data_type + "/"+"Flow.lst";
    
    int new_size[] = {1024/down, 2048/down}; // {height, width}
    Mat flowx(new_size[0], new_size[1], CV_32F); 
    Mat flowy(new_size[0], new_size[1], CV_32F); 
    
    // basic hdf5 configuration
	const H5std_string DATASET_NAME("optical_flow");
	hsize_t start[4];
	hsize_t stride[4];
	hsize_t count[4];
	hsize_t block[4];
	start[0] = 0; start[1] = 0; start[2] = 0; start[3] = 0;
	stride[0] = 1; stride[1] = 1; stride[2] = 1; stride[3] = 1;
	count[0] = 1; count[1] = 1; count[2] = 1; count[3] = 1;
	block[0] = 1; block[1] = 1; block[2] = new_size[0]; block[3] = new_size[1];
    
    hsize_t mdim[2] = {new_size[0], new_size[1]};
    DataSpace mspace(2, mdim);
    hsize_t m_count[2];
    hsize_t m_start[2];
    m_count[0] = new_size[0]; m_count[1] = new_size[1];
    m_start[0] = 0; m_start[1] = 0;
    mspace.selectHyperslab(H5S_SELECT_SET, m_count, m_start);

    ifstream fin(list_file.c_str());
    string h5_file;
    while (getline(fin, h5_file)) {
        
        const H5std_string FILE_NAME(h5_file); 
        H5File file(FILE_NAME, H5F_ACC_RDONLY);
        DataSet dataset = file.openDataSet(DATASET_NAME);
        DataSpace fspace = dataset.getSpace();
        

		start[1] = 0;
        fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
		dataset.read((void*)flowx.ptr(), PredType::NATIVE_FLOAT, mspace, fspace);
		fspace.selectNone();
        start[1] = 1;
		fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
		dataset.read((void*)flowy.ptr(), PredType::NATIVE_FLOAT, mspace, fspace);
        fspace.selectNone();
        
        Mat out;
        drawOpticalFlow(flowx, flowy, out, 10);
        // out color file
        string out_file(h5_file);
        size_t pos = out_file.find_first_of('.');
        out_file.erase(pos, 3);
        out_file += "Color.png";
        cout << "Writing: " << out_file << endl;
        imwrite(out_file, out);
    }

    return 0;
}

