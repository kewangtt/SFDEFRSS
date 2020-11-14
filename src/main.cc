#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "./System/System.h"
#include <dirent.h>
#include <vector>

using namespace std;


// Load Images from Dir
inline int getdir (std::string dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL)
    {
        return -1;
    }

    while ((dirp = readdir(dp)) != NULL) {
    	std::string name = std::string(dirp->d_name);

    	if(name != "." && name != ".." && name[0] != '.')
    		files.push_back(name);
    }
    closedir(dp);
    std::sort(files.begin(), files.end());
    if(dir.at( dir.length() - 1 ) != '/') dir = dir+"/";
	for(unsigned int i=0;i<files.size();i++)
 	{
		// printf("%d %s\n",i, files[i].c_str());
		if(files[i].at(0) != '/')
			files[i] = dir + files[i];
	}

    return files.size();
}


inline int getfiles (std::string dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL)
    {
        return -1;
    }
    while ((dirp = readdir(dp)) != NULL) {
    	std::string name = std::string(dirp->d_name);

    	if(name != "." && name != ".." && name[0] != '.')
    		files.push_back(name);
    }
    closedir(dp);
    std::sort(files.begin(), files.end());

    return files.size();
}

int main(int argc, char **argv)
{
    if(argc != 3)
    {
        cerr << endl << "Usage: ./build/SFRSS target_path mode" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> leftImagePaths;
    vector<string> rightImagePaths;
    vector<string> imageNames;
    vector<double> vTimestamps;
    string leftDirPath = string(argv[1]) + "/frames/cam0";
    string rightDirPath = string(argv[1]) + "/frames/cam1";
    int mode = atoi(argv[2]);

    getdir(leftDirPath, leftImagePaths);
    getdir(rightDirPath, rightImagePaths);
    getfiles(leftDirPath, imageNames);

    int nPairs = leftImagePaths.size();
    char buffer[256] = {0};
    for (unsigned int ii = 0; ii < leftImagePaths.size(); ++ii){
        memset(buffer, 0, 256);
        memcpy(buffer, imageNames[ii].c_str(), imageNames[ii].size() - 4);
        vTimestamps.push_back(atof(buffer));
    }

    SFRSS::System sys(string(argv[1]) + "/SFRSS.yaml", argv[1], 2);

    vector<float> vTimesTrack;
    vTimesTrack.resize(nPairs);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nPairs << endl << endl;

    // Main loop
    cv::Mat leftImage, rightImage;
    for(int ni = 0; ni < nPairs; ni+=1) // 42
    {
        printf("///////////////////////////////////////////////////////////////////\n");
        printf("frameId:%d\n", ni);
        // Read image from file
        leftImage = cv::imread(leftImagePaths[ni], 0);
        rightImage = cv::imread(rightImagePaths[ni], 0);
        double tframe = vTimestamps[ni];

        if (leftImage.empty() || rightImage.empty()){
            cerr << endl << "Failed to load image at: "
                 << leftImagePaths[ni] << "/n" << rightImagePaths[ni] << endl;
            return 1;
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Pass the image to the SLAM system
        if (mode == 0){
            sys.Run2(leftImage, rightImage, tframe, imageNames[ni]);
        }
        else{
            sys.Run3(leftImage, rightImage, tframe, imageNames[ni]);
        }

        // Check
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        // vTimesTrack record the timing cost of this tracking.
        vTimesTrack[ni] = ttrack;
    }

    return 0;
}
