InfrasCal
=========

Introduction
------------

This C++ library supports the following tasks:

1. Extrinsic infrastructure-based calibration of a multi-camera rig
2. Intrinsic and extrinsic infrastructure-based calibration of a multi-camera rig

The following two camera models are supported in this library:
* Pinhole camera model with radial and tangential distortion
* Equidistant fish-eye model (J. Kannala, and S. Brandt, A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses, PAMI 2006)

The infrastructure-based calibration runs in near real-time, and is strongly recommended if you are calibrating multiple rigs with mapping datasets.

The workings of the library are described in the three papers:

        Yukai Lin, Viktor Larsson, Marcel Geppert, Zuzana Kukelova, Marc Pollefeys, Torsten Sattler,
        Infrastructure-based Multi-Camera Calibration using Radial Projections, ECCV 2020.
    
        Lionel Heng, Mathias Bürki, Gim Hee Lee, Paul Furgale, Roland Siegwart, and Marc Pollefeys,
        Infrastructure-Based Calibration of a Multi-Camera Rig,
        In Proc. IEEE International Conference on Robotics and Automation (ICRA), 2014.
        
        Lionel Heng, Paul Furgale, and Marc Pollefeys,
        Leveraging Image-based Localization for Infrastructure-based Calibration of a Multi-camera Rig,
        Journal of Field Robotics (JFR), 2015.

If you use this library in an academic publication, please cite at least the following paper:
```
@InProceedings{Lin2020ECCV,
    author = {Yukai Lin and Viktor Larsson and Marcel Geppert and Zuzana Kukelova and Marc Pollefeys and Torsten Sattler},
    title = {{Infrastructure-based Multi-Camera Calibration using Radial Projections}},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2020},
}
```

Depending on which parts of the library you use, please cite the appropriate papers from the list above.

#### Acknowledgements ####

The InfrasCal library includes third-party code from the following sources:

        1. Lionel Heng, Bo Li, and Marc Pollefeys,
           CamOdoCal: Automatic Intrinsic and Extrinsic Calibration of a Rig
           with Multiple Generic Cameras and Odometry,
           https://github.com/hengli/camodocal
    
        2. Sameer Agarwal, Keir Mierle, and Others,
           Ceres Solver.
           https://code.google.com/p/ceres-solver/
        
        3. D. Galvez-Lopez, and J. Tardos,
           Bags of Binary Words for Fast Place Recognition in Image Sequences,
           IEEE Transactions on Robotics, 28(5):1188-1197, October 2012.
    
        4. L. Kneip, D. Scaramuzza, and R. Siegwart,
           A Novel Parametrization of the Perspective-Three-Point Problem for a
           Direct Computation of Absolute Camera Position and Orientation,
           In Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2011.
    
        5. pugixml
           http://pugixml.org/

        6. Changchang wu,
           SiftGPU: A GPU implementation of David Lowe's Scale Invariant Feature Transform
           http://cs.unc.edu/~ccwu
           
        7. Viktor Larsson, Torsten Sattler, Zuzana Kukelova and Marc Pollefeys,
           Revisiting Radial Distortion Absolute Pose.
           https://github.com/vlarsson/radialpose

Build Instructions for Ubuntu
-----------------------------

*Required dependencies*
* BLAS (Ubuntu package: libblas-dev)
* Boost (Ubuntu package: libboost-all-dev)
* Eigen3 (Ubuntu package: libeigen3-dev)
* SuiteSparse (Ubuntu package: libsuitesparse-dev)
* Ceres-solver (Ubuntu package: libceres-dev)
* CUDA
* OpenCV+contrib

*Optional dependencies*
* GTest
* Glog (Ubuntu package: libgoogle-glog-dev)

*Tested configuration versions*
* Ubuntu 18.04
* Ceres 1.13.0
* Eigen 3.3.4
* OpenCV & opencv_contrib 3.4.2, 4.1.1
* Boost 1.65.1
* Cuda 9.1, 10.1

1. Before you compile the repository code, you need to install the required
   dependencies, and install the optional dependencies if required.
    * Install Cuda
   
    * Build required dependencies
      ```
      sudo apt-get install cmake git gcc-6 g++-6 libopenblas-dev libblas-dev libeigen3-dev libgoogle-glog-dev 
      sudo apt-get install build-essential libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev libglew-dev
      sudo apt-get install libatlas-base-dev libsuitesparse-dev libsqlite3-dev libceres-dev libboost-all-dev
      ```
    * Build Opencv
   
      ```
      mkdir -p ~/dev && cd ~/dev
      git clone --depth 1 --branch 3.4.2 https://github.com/opencv/opencv.git
      git clone --depth 1 --branch 3.4.2 https://github.com/opencv/opencv_contrib.git
      cd opencv && mkdir build && cd build
      CC=/usr/bin/gcc-6 CXX=/usr/bin/g++-6 cmake .. -DWITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
      -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -DOPENCV_ENABLE_NONFREE:BOOL=ON \
      -DCUDA_NVCC_FLAGS=--expt-relaxed-constexpr
      make -j8
      sudo make install
      ```
2. Build the code.
    
    ```
    mkdir -p ~/dev && cd ~/dev
    git clone https://github.com/youkely/InfrasCal.git 
    cd InfrasCal && mkdir build && cd build
    CC=/usr/bin/gcc-6 CXX=/usr/bin/g++-6 cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j8
    ```
    
    

Examples
--------

Go to the source folder. To see all allowed options for each executable, use the --help option which shows a description of all available options.

1. Infrastructure-based calibration

        ./build/bin/infrastr_calib --camera-count 5 \
        --output ./data/demo/results \
        --map ./data/demo/map \
        --database ./data/demo/map/database.db \
        --input ./data/demo/ \
        --vocab ./data/vocabulary/sift128.bin \
        -v --camera-model pinhole-radtan --save

   The camera-model parameter takes one of the following two values: pinhole-radtan, and pinhole-equi(kannala-brandt).
   
   The calibration mode takes one of the following options: InRaSU(default, corresponds to Inf+1DR+RA in the ECCV2020 paper), In(Inf+K), InRI(Inf+K+RI), InRa(Inf+RD), InRaS(Inf+RD+RA)
