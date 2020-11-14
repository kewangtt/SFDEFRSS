# SFDEFRSS: Single Frmae Depth Estimation for Rolling Shutter Stereo

## 1. Installation
git clone https://github.com/kewangtt/SFDEFRSS.git

### 1.1 Required Dependencies
**suitesparse and eigen3 (required).** Install with  
`sudo apt-get install libsuitesparse-dev libeigen3-dev libboost-all-dev`  

**OpenCV (>=3.4)** and **CUDA**

### 1.2 Build
```
cd SFDEFRSS  
mkdir build
cd build
cmake ..
make
```  
this will compile an executable file `SFRSS_EXE` in the build directory.

## 2. Usage
### 2.1 Create sub-discretionary 
```  
sh create_subdir.sh
```  
### 2.2 Run  
Run an example sequence by  
```  
./build/SFRSS_EXE ./example 0
``` 
this will generate all results of the example sequence. The computed results are saved in 
```  
./Result/disparity_rgb  "rgb disparity for visualization"
./Result/undited_depth  "undistorted depth map"
./Result/motion_states  "estimated motion states (a_x, a_y, a_z, v_x, v_y, v_z)"
./Result/baseline_map  "baseline map"
``` 
`a_x, a_y, a_z` and `v_x, v_y, v_z` are the estimated angular speed and velocity respectively.  
The depth map is saved as a binary file and data type is double type. Thus, the size of depth map is 
```  
sizeof(double)*height*width.
```

All external parameters and calibration parameters are saved in
`./example/SFRSS.yaml`.

We also provide the close-form motion state solver proposed in "From two rolling shutters to one global shutter". Which can be run by
```  
./build/SFRSS_EXE ./example 1
``` 
We save the 20 candidates solutions with the most inliers in `.\example\Result\close_form`.

## 3. Demo video
![image](https://github.com/kewangtt/SFDEFRSS/tree/main/demo/demo.gif)

## 4. License
The open-source version is licensed under the GNU General Public License Version 3 (GPLv3).