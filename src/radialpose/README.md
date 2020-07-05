# radialpose

This is a C++ implementation of different minimal solvers for absolute pose estimation with radial distortion.
Currently, the following solvers are implemented
 * 1 - D(1,0) - 5p -- Larsson et al.  ICCV 2019
 * 2 - D(2,0) - 5p -- Larsson et al.  ICCV 2019
 * 3 - D(3,0) - 5p -- Larsson et al.  ICCV 2019  (Minimal)
 * 4 - D(3,3) - 8p -- Larsson et al.  ICCV 2019
 * 5 - U(1,0) - 5p -- Larsson et al.  ICCV 2019
 * 6 - U(0,1) - 4p -- Larsson et al.  ICCV 2017  (Minimal, Non-planar)
 * 7 - U(0,1) - 4p -- Bujnak et al.   ACCV 2010  (Minimal, Non-planar)
 * 8 - U(0,1) - 5p -- Kukelova et al. ICCV 2013
 * 9 - U(0,2) - 5p -- Kukelova et al. ICCV 2013
 * 10 - U(0,3) - 5p -- Kukelova et al. ICCV 2013  (Minimal)
 * 11 - U(0,1) - 4p -- Oskarsson       arxiv 2018 (Minimal, Planar)
 * 12 - N/A    - 5p -- Kukelova et al. ICCV 2013  (Minimal, 1D Radial)
 
where D(2,0) corresponds to a distortion model with two polynomial parameters and zero division parameters, see the paper for more details. Note that the runtime experiments in the paper were done with a more barebones implementation of these solvers.

## Installation
The solvers are available as a library. To build run
```
mkdir build
cd build/
cmake ../
make -j
```
After building you should be able to run `radialpose_test` and `ransac_test`.

There are currently two dependencies:
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
* [RansacLib](https://github.com/tsattler/RansacLib)

There are also Matlab mex interfaces. Check `matlab/compile_mex.m`.


## Using radialpose
The different solvers are available through the classes
* `radialpose::larsson_iccv19::Solver<Np,Nd,DistortionModel>()`
* `radialpose::larsson_iccv17::NonPlanarSolver()`
* `radialpose::bujnak_accv10::NonPlanarSolver()`
* `radialpose::kukelova_iccv13::Solver()`
* `radialpose::kukelova_iccv13::Radial1DSolver(Nd)`
* `radialpose::oskarsson_arxiv18::PlanarSolver`

Each of these solvers implement a function 
`int  estimate(const Points2D &image_points, const Points3D &world_points, std::vector<Camera>  *poses) const;`
which should be used to call the solvers. See `solvers/pose_estimator.h` for definitions of `Points2D` and `Points3D`, and see `misc/camera.h` for the definition of `Camera`.

**Important**: You should always call `Solver.estimate()` instead of `Solver.solve()`. The estimate function takes care of rescaling the input data before passing it to the solver as well as filtering some bad solutions and other nice things.

### RansacLib wrappers
For each solver there is also a RansacLib wrapper in `radialpose::RansacEstimator<Solver>`.

For example, to do LO-RANSAC with the D(2,0) solver from the ICCV19 paper you can do the following:
```
larsson_iccv19::Solver<2, 0, true> estimator;
RansacEstimator<larsson_iccv19::Solver<2, 0, true>>  solver(x, X, estimator);

Camera best_model;
ransac_lib::RansacStatistics ransac_stats;

ransac_lib::LocallyOptimizedMSAC<Camera,std::vector<Camera>,RansacEstimator<larsson_iccv19::Solver<2, 0, true>>> lomsac;
int inliers =  lomsac.EstimateModel(options, solver, &best_model, &ransac_stats);
```

## License
radialpose is licensed under the BSD 3-Clause license. Please see [License](https://github.com/vlarsson/radialpose/blob/master/LICENSE) for details.

## Citing
If you are using the library for (scientific) publications, please cite the following source:
```
@inproceedings{larsson2019revisiting,
  title = {{Revisiting Radial Distortion Absolute Pose}},
  author = {Larsson, Viktor and Sattler, Torsten and Kukelova, Zuzana and Pollefeys, Marc},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year = {2019}
}
```
If you use any of the re-implementations of the other methods please cite the corresponding papers as well.
```
@inproceedings{bujnak2010new,
  title={New efficient solution to the absolute pose problem for camera with unknown focal length and radial distortion},
  author={Bujnak, Martin and Kukelova, Zuzana and Pajdla, Tomas},
  booktitle={Asian Conference on Computer Vision (ACCV)},
  year={2010}
}
@inproceedings{kukelova2013real,
  title={Real-time solution to the absolute pose problem with unknown radial distortion and focal length},
  author={Kukelova, Zuzana and Bujnak, Martin and Pajdla, Tomas},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year={2013}
}
@inproceedings{larsson2017making,
  title={Making minimal solvers for absolute pose estimation compact and robust},
  author={Larsson, Viktor and Kukelova, Zuzana and Zheng, Yinqiang},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year={2017}
}
@article{oskarsson2018fast,
  title={A fast minimal solver for absolute camera pose with unknown focal length and radial distortion from four planar points},
  author={Oskarsson, Magnus},
  journal={arXiv preprint arXiv:1805.10705},
  year={2018}
}
```



