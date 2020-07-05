clear all
clear mex
clc
RANSACLIB_DIR = '../RansacLib/';
sources = {'../solvers/bujnak_accv10.cc', ...
    '../solvers/kukelova_iccv13.cc','../solvers/larsson_iccv17.cc', ...
    '../solvers/larsson_iccv19.cc','../solvers/larsson_iccv19_impl.cc', ...
    '../solvers/oskarsson_arxiv18.cc','../misc/distortion.cc', ...
    '../misc/univariate.cc','../misc/refinement.cc'};


if ispc
    comp_flags = "COMPFLAGS=$COMPFLAGS /bigobj";
    EIGEN_DIR = 'c:/work/thirdparty/eigen3/';
else
    comp_flags = 'CXXFLAGS="$CXXFLAGS -std=c++14"';
    EIGEN_DIR = '/usr/include/eigen3/';    
end
    
%%
tic
mex(comp_flags,'-v','-DEIGEN_NO_DEBUG',['-I' EIGEN_DIR],'radialpose_mex.cpp',sources{:})
toc
%%
tic
mex(comp_flags,'-v','-D_USE_MATH_DEFINES','-DEIGEN_NO_DEBUG',['-I' RANSACLIB_DIR],['-I' EIGEN_DIR],'ransac_radialpose_mex.cc',sources{:})
toc

%% Test radialpose_mex

% GT parameters
R_gt = orth(randn(3,3));
R_gt = R_gt * det(R_gt);
t_gt = randn(3,1);
dist_gt = [-0.1];
focal_gt = 2000;

% Setup instance
N = 8;
X = [randn(2,N); 2+10*rand(1,N)];
xu = X ./ X([3;3;3],:);
X = R_gt' *(X - t_gt * ones(1,N));
d2 = sum(xu(1:2,:).^2);
xd = focal_gt * xu(1:2,:) .* ([1;1] * (1+ dist_gt(1) * d2));


[R,t,f,dist] = radialpose_mex(xd,X,1)

%% Test ransac_radialpose_mex

% GT parameters
R_gt = orth(randn(3,3));
R_gt = R_gt * det(R_gt);
t_gt = randn(3,1);
dist_gt = [-0.1];
focal_gt = 2000;
tol = 2.0;

% Setup instance
N = 100;
X = [randn(2,N); 2+10*rand(1,N)];
xu = X ./ X([3;3;3],:);
X = R_gt' *(X - t_gt * ones(1,N));
d2 = sum(xu(1:2,:).^2);
xd = focal_gt * xu(1:2,:) .* ([1;1] * (1+ dist_gt(1) * d2));

ind_outlier = randperm(N,round(N*0.1));
xd(:,ind_outlier) = randn(2,length(ind_outlier)) * std(xd(:));
xd = xd + 0.1 * randn(size(xd));

[R,t,f,dist] = ransac_radialpose_mex(xd,X,1,tol)

