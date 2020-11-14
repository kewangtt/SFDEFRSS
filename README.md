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
