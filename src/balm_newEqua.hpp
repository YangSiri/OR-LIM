//
// Created by cyz on 2022/8/21.
//
// MVR：Multi-View cloud Registration
// inspired from https://arxiv.org/pdf/2108.02976.pdf
//
//

#ifndef BALM_BALM_NEWEQUA_HPP
#define BALM_BALM_NEWEQUA_HPP

#include <ros/ros.h>
#include <pcl/common/transforms.h>
#include <unordered_map>
#include <opencv/cv.h>
#include <thread>
#include <mutex>
#include <fstream>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include "ceresfactors.hpp"

typedef std::vector<Eigen::Vector3d> PL_VEC;
typedef pcl::PointXYZINormal PointType;
#define MIN_PS 7
using namespace std;
const string rootFolder = "/home/cyz/workspace/catkin_localizeInHDmap/src/FAST_LIO_LC-master/FAST-LIO/PCD/directions/";

// Key of hash table
class VOXEL_LOC
{
public:
    int64_t x, y, z;

    VOXEL_LOC(int64_t vx=0, int64_t vy=0, int64_t vz=0): x(vx), y(vy), z(vz){}

    bool operator== (const VOXEL_LOC &other) const
    {
        return (x==other.x && y==other.y && z==other.z);
    }
};

// Hash value
namespace std
{
    template<>
    struct hash<VOXEL_LOC>
    {
        size_t operator() (const VOXEL_LOC &s) const
        {
            using std::size_t; using std::hash;
            return ((hash<int64_t>()(s.x) ^ (hash<int64_t>()(s.y) << 1)) >> 1) ^ (hash<int64_t>()(s.z) << 1);
        }
    };
}

struct M_POINT
{
    float xyz[3];
    int count = 0;
};

// Similar with PCL voxelgrid filter
void down_sampling_voxel(pcl::PointCloud<PointType> &pl_feat, double voxel_size)
{
    if(voxel_size < 0.01)
    {
        return;
    }

    unordered_map<VOXEL_LOC, M_POINT> feat_map;
    uint plsize = pl_feat.size();

    for(uint i=0; i<plsize; i++)
    {
        PointType &p_c = pl_feat[i];
        float loc_xyz[3];
        for(int j=0; j<3; j++)
        {
            loc_xyz[j] = p_c.data[j] / voxel_size;
            if(loc_xyz[j] < 0)
            {
                loc_xyz[j] -= 1.0;
            }
        }

        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        auto iter = feat_map.find(position);
        if(iter != feat_map.end())
        {
            iter->second.xyz[0] += p_c.x;
            iter->second.xyz[1] += p_c.y;
            iter->second.xyz[2] += p_c.z;
            iter->second.count++;
        }
        else
        {
            M_POINT anp;
            anp.xyz[0] = p_c.x;
            anp.xyz[1] = p_c.y;
            anp.xyz[2] = p_c.z;
            anp.count = 1;
            feat_map[position] = anp;
        }
    }

    plsize = feat_map.size();
    pl_feat.clear();
    pl_feat.resize(plsize);

    uint i = 0;
    for(auto iter=feat_map.begin(); iter!=feat_map.end(); ++iter)
    {
        pl_feat[i].x = iter->second.xyz[0]/iter->second.count;
        pl_feat[i].y = iter->second.xyz[1]/iter->second.count;
        pl_feat[i].z = iter->second.xyz[2]/iter->second.count;
        i++;
    }

}

void down_sampling_voxel(PL_VEC &pl_feat, double voxel_size)
{
    unordered_map<VOXEL_LOC, M_POINT> feat_map;
    uint plsize = pl_feat.size();

    for(uint i=0; i<plsize; i++)
    {
        Eigen::Vector3d &p_c = pl_feat[i];
        double loc_xyz[3];
        for(int j=0; j<3; j++)
        {
            loc_xyz[j] = p_c[j] / voxel_size;
            if(loc_xyz[j] < 0)
            {
                loc_xyz[j] -= 1.0;
            }
        }

        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        auto iter = feat_map.find(position);
        if(iter != feat_map.end())
        {
            iter->second.xyz[0] += p_c[0];
            iter->second.xyz[1] += p_c[1];
            iter->second.xyz[2] += p_c[2];
            iter->second.count++;
        }
        else
        {
            M_POINT anp;
            anp.xyz[0] = p_c[0];
            anp.xyz[1] = p_c[1];
            anp.xyz[2] = p_c[2];
            anp.count = 1;
            feat_map[position] = anp;
        }

    }

    plsize = feat_map.size();
    pl_feat.resize(plsize);

    uint i = 0;
    for(auto iter=feat_map.begin(); iter!=feat_map.end(); ++iter)
    {
        pl_feat[i][0] = iter->second.xyz[0]/iter->second.count;
        pl_feat[i][1] = iter->second.xyz[1]/iter->second.count;
        pl_feat[i][2] = iter->second.xyz[2]/iter->second.count;
        i++;
    }


}

void plvec_trans_func(vector<Eigen::Vector3d> &orig, vector<Eigen::Vector3d> &tran,
                      Eigen::Matrix3d R, Eigen::Vector3d t)
{
    uint orig_size = orig.size();
    tran.resize(orig_size);

    for(uint i=0; i<orig_size; i++)
    {
        tran[i] = R*orig[i] + t;
    }
}

template <typename T>
void pub_func(T &pl, ros::Publisher &pub, const ros::Time &current_time)
{
    pl.height = 1; pl.width = pl.size();
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(pl, output);
    output.header.frame_id = "map";
    output.header.stamp = current_time;
    pub.publish(output);
}

// Convert PointCloud2 to PointType
void rosmsg2ptype(const sensor_msgs::PointCloud2 &pl_msg, pcl::PointCloud<PointType> &plt)
{
    pcl::PointCloud<pcl::PointXYZI> pl;
    pcl::fromROSMsg(pl_msg, pl);

    uint asize = pl.size();
    plt.resize(asize);
//    plt.header.stamp = pl_msg.header.stamp.toNSec();

    for(uint i=0; i<asize; i++)
    {
        plt[i].x = pl[i].x;
        plt[i].y = pl[i].y;
        plt[i].z = pl[i].z;
        plt[i].intensity = pl[i].intensity;
    }
}

// P_fix in the paper
// Summation of P_fix
//class SIG_VEC_CLASS
//{
//public:
//    Eigen::Matrix3d sigma_vTv;
//    Eigen::Vector3d sigma_vi;
//    int sigma_size;
//
//    SIG_VEC_CLASS()
//    {
//        sigma_vTv.setZero();
//        sigma_vi.setZero();
//        sigma_size = 0;
//    }
//
//    void tozero()
//    {
//        sigma_vTv.setZero();
//        sigma_vi.setZero();
//        sigma_size = 0;
//    }
//
//};

const double one_three = (1.0 / 3.0);
double feat_eigen_limit[2] = {3*3, 2*2};
double opt_feat_eigen_limit[2] = {4*4, 3*3};


// LM optimizer for map-refine
class LM_SLWD_VOXEL
{
public:
    int slwd_size;  // the size of sliding window
    int filternum;  //
    int thd_num, jac_leng;
    int iter_max = 20;

    double corn_less;  //

    vector<SO3> so3_poses, so3_poses_temp;
    vector<Eigen::Vector3d> t_poses, t_poses_temp;

    vector<Gaussian3D > gauss3d_opt;

    vector<int> lam_types; // 0 surf, 1 line
    vector<vector<SIG_VEC_CLASS> > sig_vecs;
    vector<vector<Eigen::Vector3d>* > plvec_voxels;
    vector<vector<int>* > slwd_nums;
    vector<bool> voxel_opt_flag;
    vector<std::pair<int, int> > voxel_No_size;
    vector<Eigen::Vector3d > voxel_normals;

    vector<vector<int> > vox_No_perFrame;
    vector<bool> frame_opt_flag;

    int map_refine_flag;
    mutex my_mutex;
    pcl::PointCloud<PointType> cloud_per_frame;

    LM_SLWD_VOXEL(int ss, int fn, int thnum): slwd_size(ss), filternum(fn), thd_num(thnum){

        so3_poses.resize(ss); t_poses.resize(ss);
        so3_poses_temp.resize(ss); t_poses_temp.resize(ss);

        frame_opt_flag.resize(ss, true);
        vox_No_perFrame.resize(ss);

        jac_leng = 6*ss;
        corn_less = 0.1;
        map_refine_flag = 0;
    }

    // Decide the number of points used for optimization per frame
    void downsample(vector<Eigen::Vector3d> &plvec_orig, int cur_frame,
                    vector<Eigen::Vector3d> &plvec_voxel,
                    vector<int> &slwd_num, int filternum2use){

        uint plsize = plvec_orig.size();
        if(plsize <= (uint)filternum2use){

            for(uint i=0; i<plsize; i++){
                plvec_voxel.push_back(plvec_orig[i]);
                slwd_num.push_back(cur_frame);
            }
            return;
        }

        Eigen::Vector3d center;
        double part = 1.0 * plsize / filternum2use;
//        cout<< "[Debug] plsize/filternum2use: " << plsize << " / " << filternum2use << endl;

        for(int i=0; i<filternum2use; i++)
        {
            uint np = part*i;
            uint nn = part*(i+1);
            center.setZero();
            for(uint j=np; j<nn; j++)
                center += plvec_orig[j];

            center = center / (nn-np);
            plvec_voxel.push_back(center);
            slwd_num.push_back(cur_frame);
        }
    }

    // Push voxel into optimizer
    void push_voxel(vector<vector<Eigen::Vector3d>* > &plvec_orig, SIG_VEC_CLASS &marg_info,
                    int lam_type, const Eigen::Vector3d& normal){

        int process_frame_size = 0;
        for(int i=0; i<slwd_size; i++)
            if(!plvec_orig[i]->empty())
                process_frame_size++;
        // Only one scan
        if(process_frame_size < 2)
            return;

//        int filternum2use = filternum;
//        if(filternum * process_frame_size < MIN_PS)
//            filternum2use = MIN_PS / process_frame_size + 1;

//        vector<SIG_VEC_CLASS> sig_vec (slwd_size);
//        vector<Eigen::Vector3d> *plvec_voxel = new vector<Eigen::Vector3d>();
//        // Frame num in sliding window for each point in "plvec_voxel"
//        vector<int> *slwd_num = new vector<int>();
//        plvec_voxel->reserve(filternum2use*slwd_size);
//        slwd_num->reserve(filternum2use*slwd_size);
//        sig_vec->resize(slwd_size);
        Gaussian3D gaussian3D(slwd_size);

        int curVoxNo = plvec_voxels.size(), cnt = 0;
        // retain one point for one scan (you can modify)
        for(int i = 0; i < slwd_size; i++) {

            gaussian3D.sig_vecs_local[i].sigma_size = plvec_orig[i]->size();
            cnt += plvec_orig[i]->size();

            if (!plvec_orig[i]->empty()) {

                Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
                Eigen::Vector3d centor =  Eigen::Vector3d::Zero();
                for (auto pt : *plvec_orig[i]) {
                    centor += pt;
                    covMat += pt * pt.transpose();
                }
                covMat = covMat - centor * centor.transpose() / plvec_orig[i]->size();
                covMat = covMat / plvec_orig[i]->size();
                centor = centor / plvec_orig[i]->size();

                gaussian3D.sig_vecs_local[i].sigma_vi = centor;
                gaussian3D.sig_vecs_local[i].sigma_vTv = covMat;

                vox_No_perFrame[i].emplace_back(curVoxNo);
//                downsample(*plvec_orig[i], i, *plvec_voxel, *slwd_num, filternum2use);
            }
        }
        gaussian3D.ptSize = cnt;
        gaussian3D.margin = marg_info;
        gauss3d_opt.emplace_back(gaussian3D);

        // for(int i=0; i<slwd_size; i++)
        // {
        //   for(uint j=0; j<plvec_orig[i]->size(); j++)
        //   {
        //     plvec_voxel->push_back(plvec_orig[i]->at(j));
        //     slwd_num->push_back(i);
        //   }
        // }

//        plvec_voxels.push_back(plvec_voxel); // Push a voxel into optimizer
//        slwd_nums.push_back(slwd_num);
//        lam_types.push_back(lam_type);
//        sig_vecs.push_back(sig_vec);     // history points out of sliding window

        voxel_normals.emplace_back(normal.normalized()); // point number and normal direction for every voxel
//        voxel_No_size.emplace_back(std::make_pair(curVoxNo, plvec_voxel->size() + sig_vec.sigma_size));
    }


    void directionAdjust(Eigen::Vector3d& v1){

        const Eigen::Vector3d axis_x(1,0,0);
        const Eigen::Vector3d axis_y(0,1,0);
        const Eigen::Vector3d axis_z(0,0,1);

        const double thre = M_PI/6 ;
        double ang_x = atan2(v1.cross(axis_x).norm(), v1.transpose()*axis_x);
        double ang_y = atan2(v1.cross(axis_y).norm(), v1.transpose()*axis_y);
        double ang_z = atan2(v1.cross(axis_z).norm(), v1.transpose()*axis_z);
        if ( ang_x < thre || M_PI - ang_x < thre){  // [0, PI]
            if (v1(0) < 0) v1 *= -1;
        }else if (ang_y < thre || M_PI - ang_y < thre){
            if (v1(1) < 0) v1 *= -1;
        }else if (v1(2) < 0)
            v1 *= -1;
    }
    void evaluateNormalperFrame(double ratio){

        string rootFolder = "/home/cyz/workspace/catkin_localizeInHDmap/src/FAST_LIO_LC-master/FAST-LIO/PCD/directions/";
        std::vector<ofstream > txtors(slwd_size);
        for (int i = 0; i < slwd_size; ++i)
            txtors[i].open(rootFolder+to_string(i)+".txt", ios::out);
        ofstream scoreTxt(rootFolder+"scores.txt", ios::out);

        voxel_opt_flag.resize(plvec_voxels.size(), false);

        int voxNum = int(ratio * voxel_No_size.back().first);
        cout << "[ EVA ] Number of voxels: " << voxNum << " / " << plvec_voxels.size() << endl;
        std::sort(voxel_No_size.begin(), voxel_No_size.end(),
                  [](std::pair<int, int> a, std::pair<int, int> b)
                  { return a.second > b.second; });  // decreasing order

        std::vector<Eigen::Matrix3d > cov_perFrame(slwd_size, Eigen::Matrix3d::Zero());
        int voxNo, frameNo, cnt, ptr, cntAll;

        for (int i = 0; i < voxNum; ++i) {

            voxNo = voxel_No_size[i].first;
            vector<int> &slwd_num = *slwd_nums[voxNo];
            cntAll = slwd_num.size();

            vector<std::pair<int, int> > frame_size;  // point number for each frame
            ptr = 0;
            while (ptr < cntAll) {
                frameNo = slwd_num[ptr];
                cnt = std::count(slwd_num.begin()+ptr, slwd_num.end(), frameNo);
                frame_size.emplace_back(std::make_pair(frameNo, cnt));
                ptr += cnt;
            }
            cnt = frame_size.size();
            for (int j = 0; j < cnt ; ++j) {
                directionAdjust(voxel_normals[voxNo]);

                // cov_perFrame[frame_size[j].first] += frame_size[j].second *
                //                                      (voxel_normals[voxNo] * voxel_normals[voxNo].transpose());
                cov_perFrame[frame_size[j].first] += (voxel_normals[voxNo] * voxel_normals[voxNo].transpose());

                txtors[frame_size[j].first] << voxel_normals[voxNo].x() << " " << voxel_normals[voxNo].y() << " "
                                            << voxel_normals[voxNo].z() << " " << frame_size[j].second << endl;
            }
            voxel_opt_flag[voxNo] = true;
        }
//        cout << "[ EVA ] Number of voxels: " << std::count(voxel_opt_flag.begin(), voxel_opt_flag.end(), true) << endl;
        for (int i = 0; i < slwd_size; ++i)
            txtors[i].close();
        txtors.clear();

        Eigen::Vector3d eigen_value;
        for (int i = 0; i < slwd_size; ++i) {

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(2 * cov_perFrame[i]);
            eigen_value = saes.eigenvalues();  // increasing
            double scoreI = sqrt(eigen_value(0)) / sqrt(eigen_value(2));
            cout << "[ EVA ] Frame " << i << " sphericity " << scoreI << endl;
            scoreTxt << scoreI << endl;

            if (scoreI < 0.3)  // sphericity
                frame_opt_flag[i] = false;
        }
        scoreTxt.close();

    }

    // Calculate residual
//    void evaluate_only_residual(vector<SO3> &so3_ps, vector<Eigen::Vector3d> &t_ps, double &residual){
//
//        residual = 0;
//        uint gps_size = plvec_voxels.size();
//        Eigen::Vector3d vec_tran;
//
//        for(uint a=0; a<gps_size; a++){
//
//            if (!voxel_opt_flag[a]) continue;
//
//            uint k = lam_types[a];
//            SIG_VEC_CLASS &sig_vec = sig_vecs[a];
//            vector<Eigen::Vector3d> &plvec_voxel = *plvec_voxels[a];
//            vector<int> &slwd_num = *slwd_nums[a];
//            uint backnum = plvec_voxel.size();
//
//            Eigen::Vector3d centor(Eigen::Vector3d::Zero());
//            Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());
//
//            for(uint i=0; i<backnum; i++)
//            {
//                vec_tran = so3_ps[slwd_num[i]].matrix()*plvec_voxel[i] + t_ps[slwd_num[i]];
//                centor += vec_tran;
//                covMat += vec_tran * vec_tran.transpose();
//
//                {
//                    PointType pt;
//                    pt.x = vec_tran(0);
//                    pt.y = vec_tran(1);
//                    pt.z = vec_tran(2);
//                    pt.intensity = slwd_num[i];
//                    pt.curvature = a;
//                    cloud_per_frame.points.emplace_back(pt);
//                }
//            }
//
//            double N_points = backnum + sig_vec.sigma_size;
//            centor += sig_vec.sigma_vi;
//            covMat += sig_vec.sigma_vTv;
//
//            covMat = covMat - centor*centor.transpose()/N_points;
//            covMat = covMat / N_points;
//
//            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
//            Eigen::Vector3d eigen_value = saes.eigenvalues();
//
//            if(k == 1)
//                residual += corn_less*eigen_value[k];
//            else
//                residual += eigen_value[k];
//        }
//
//        getVoxelCloud();
//    }

    // ceres numeric jacob
    bool damping_iter(){

        bool ifconverged = false;
        my_mutex.lock();
        map_refine_flag = 1;
        my_mutex.unlock();

        int meaN = gauss3d_opt.size();
        cout << "gauss3d_opt.size(): " << meaN << endl;

//        int meaN = plvec_voxels.size();
//        if( meaN!= slwd_nums.size() || meaN != lam_types.size() || meaN != sig_vecs.size()){
//
//            printf("size is not equal\n");
//            exit(0);
//        }

        double residual1, residual2, q;

        ceres::LocalParameterization *q_parameterization = new ParamSO3();
        ceres::Problem::Options problem_options;
        std::unique_ptr<ceres::Problem> problem;
        std::vector<ceres::ResidualBlockId > resiIDs;
        auto evalCallBack = new SurfRegXXEvalCallBack;

        ceres::Solver::Options options;
//        options.minimizer_type = ceres::LEVENBERG_MARQUARDT;
        //  options.gradient_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
        //  options.function_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.use_nonmonotonic_steps = true;
        options.max_consecutive_nonmonotonic_steps = 5;
        options.update_state_every_iteration = true;
        //    options.trust_region_strategy_type = ceres::DOGLEG;
        //    options.dogleg_type = ceres::SUBSPACE_DOGLEG;
        //    options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = thd_num;
        options.max_num_iterations = 999;
        options.evaluation_callback = evalCallBack;

        vector<Eigen::Vector3d > rot_params(slwd_size);
        vector<Eigen::Vector3d > trans_params(slwd_size);
        for (int j = 0; j < slwd_size; ++j) {
            rot_params[j] = so3_poses[j].log();
            trans_params[j] = t_poses[j];
            evalCallBack->addFrame(rot_params[j].data(), trans_params[j].data());
        }

        iter_max = 1;  // max iteration
        for(int i=0; i < iter_max; i++){

            problem.reset(new ceres::Problem(problem_options));
            for (int j = 0; j < slwd_size; ++j){
                problem->AddParameterBlock(rot_params[j].data(), 3, q_parameterization);
                problem->AddParameterBlock(trans_params[j].data(), 3);
            }

            for(auto &gauss_cur : gauss3d_opt){

                gauss_cur.initialize(evalCallBack->param_rot_, evalCallBack->param_trans_);
//                cout<< "lamdas_global: " << gauss_cur.lamdas.transpose() << endl;

                Matrix3d lamda_vecs_local;
                Vector3d lamdas_local;
                bool isvalid = false;
                for (int j = 0; j < slwd_size; ++j) {
                    auto& info_local = gauss_cur.sig_vecs_local[j];
                    if(info_local.sigma_size < 5) continue;
                    isvalid = true;

                    double scalar = sqrt(info_local.sigma_size*1.0 );
                    ceres::CostFunction *cost_functor =
                            new EigenFactorTrans2(&gauss_cur, info_local.sigma_vi, scalar);
                    problem->AddResidualBlock(cost_functor, nullptr,
                                              rot_params[j].data(), trans_params[j].data());

                    /// local SVD
                    auto eigen_solver = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(info_local.sigma_vTv);
                    Matrix3d axis_ = eigen_solver.eigenvectors();
                    axis_.col(2) = axis_.col(0).cross(axis_.col(1));
                    SO3 rot = SO3(axis_);
                    lamda_vecs_local = rot.matrix();
                    lamdas_local = eigen_solver.eigenvalues();
//                    cout<< "lamdas_local: " << lamdas_local.transpose() << endl;  // 0?

                    double rs0 = sqrt(lamdas_local.y() * info_local.sigma_size );
                    double rs1 = sqrt(lamdas_local.z() * info_local.sigma_size );
                    if (isnan(rs0) || isnan(rs1)) {
                        cout<< "[ X ] scalar: " << scalar << ", rs0: " << rs0 << ", rs1: " << rs1 << endl;
                        continue;
                    }

                    ceres::CostFunction *fr0 =
                            new EigenFactorRotAxis(&gauss_cur, lamda_vecs_local.col(1), rs0);
                    ceres::CostFunction *fr1 =
                            new EigenFactorRotAxis(&gauss_cur, lamda_vecs_local.col(2), rs1);
                    // ceres::CostFunction *fr3 =
                    //     new EigenFactorRotAxis(&info_map, info_local.rot_mat.col(0),
                    //     rs2);
                    problem->AddResidualBlock(fr0, nullptr,
                            //  new ceres::HuberLoss(0.1 * rs0),
                                              rot_params[j].data());
                    problem->AddResidualBlock(fr1, nullptr,
                            //  new ceres::HuberLoss(0.1 * rs1),
                                              rot_params[j].data());
                }
                if (isvalid)   evalCallBack->addPlane(&gauss_cur);
            }

            ceres::Solver::Summary summary;
            ceres::Solve(options, problem.get(), &summary);
            cout << summary.BriefReport() << endl;

//            evaluate_only_residual(so3_poses, t_poses, residual2);
//            cout<< "After optimization residual2: " << residual2 << endl;

            q = (residual1-residual2);

            if(fabs(q) < 1e-9){
                cout << "[ LM ] CONVERGED " << endl;
                ifconverged = true;
                break;
            }
        }

        my_mutex.lock();
        map_refine_flag = 2;
        my_mutex.unlock();
        return ifconverged;
//        return true;
    }

    int read_refine_state()
    {
        int tem_flag;
        my_mutex.lock();
        tem_flag = map_refine_flag;
        my_mutex.unlock();
        return tem_flag;
    }

    void set_refine_state(int tem)
    {
        my_mutex.lock();
        map_refine_flag = tem;
        my_mutex.unlock();
    }

    void free_voxel()
    {
        gauss3d_opt.clear();

        uint a_size = plvec_voxels.size();
        for(uint i=0; i<a_size; i++)
        {
            delete (plvec_voxels[i]);
            delete (slwd_nums[i]);
        }
        for(uint i=0; i<slwd_size; i++)
            vox_No_perFrame[i].clear();
        vox_No_perFrame.clear();

        plvec_voxels.clear();
        slwd_nums.clear();
        sig_vecs.clear();
        lam_types.clear();
        voxel_normals.clear();
        voxel_No_size.clear();
    }

    void getVoxelCloud(){

        if (!cloud_per_frame.empty())
            pcl::io::savePCDFileBinaryCompressed("/home/cyz/workspace/catkin_localizeInHDmap/"
                                                 "src/FAST_LIO_LC-master/FAST-LIO/PCD/submapWithLable.pcd", cloud_per_frame);
        cloud_per_frame.points.resize(0);
    }

};

class OCTO_TREE
{
public:
    static int voxel_windowsize;
    vector<PL_VEC*> plvec_orig;
    vector<PL_VEC*> plvec_tran;
    int octo_state; // 0 is end of tree, 1 is not
    PL_VEC sig_vec_points;
    SIG_VEC_CLASS sig_vec;
    int ftype;
    int points_size, sw_points_size;
    double feat_eigen_ratio, feat_eigen_ratio_test;
    PointType ap_centor_direct;
    double voxel_center[3]; // x, y, z
    double quater_length;
    OCTO_TREE* leaves[8];
    bool is2opt;
    int capacity;  // num of frames
    pcl::PointCloud<PointType> root_centors;
    Eigen::Vector3d normal_dir;
    vector<Eigen::Vector3d> frame_cento_local;
    vector<Eigen::Matrix3d> frame_cov_local;

    OCTO_TREE(int ft, int capa): ftype(ft), capacity(capa)
    {
        octo_state = 0;
        for(int i=0; i<8; i++)
        {
            leaves[i] = nullptr;
        }

        for(int i=0; i<capacity; i++)
        {
            plvec_orig.push_back(new PL_VEC());
            plvec_tran.push_back(new PL_VEC());
        }

        frame_cento_local.resize(capacity, Eigen::Vector3d::Zero());
        frame_cov_local.resize(capacity, Eigen::Matrix3d::Zero());

        is2opt = true;
    }

    // Used by "recut"
    void calc_eigen()
    {
        Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());
        Eigen::Vector3d center(0, 0, 0);

        uint asize;
        for(int i=0; i<OCTO_TREE::voxel_windowsize; i++)
        {
            asize = plvec_tran[i]->size();
            for(uint j=0; j<asize; j++)
            {
                covMat += (*plvec_tran[i])[j] * (*plvec_tran[i])[j].transpose();
                center += (*plvec_tran[i])[j];
            }
        }

        covMat += sig_vec.sigma_vTv;
        center += sig_vec.sigma_vi;
        center /= points_size;

        covMat = covMat/points_size - center*center.transpose();

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
        feat_eigen_ratio = saes.eigenvalues()[2] / saes.eigenvalues()[ftype];
        Eigen::Vector3d direct_vec = saes.eigenvectors().col(2*ftype);

        ap_centor_direct.x = center.x();
        ap_centor_direct.y = center.y();
        ap_centor_direct.z = center.z();
        ap_centor_direct.normal_x = direct_vec.x();
        ap_centor_direct.normal_y = direct_vec.y();
        ap_centor_direct.normal_z = direct_vec.z();
    }

    /// Cut root voxel into small pieces
    /// frame_head: Position of newest scan in sliding window
    void recut(int layer, uint frame_head, pcl::PointCloud<PointType> &pl_feat_map){

        if(octo_state == 0){

            points_size = 0;
            for(int i=0; i<OCTO_TREE::voxel_windowsize; i++)
                points_size += plvec_orig[i]->size();


            points_size += sig_vec.sigma_size;
            if(points_size < MIN_PS){
                feat_eigen_ratio = -1;
                return;
            }

            calc_eigen(); // calculate eigenvalue ratio

            if(isnan(feat_eigen_ratio)){
                feat_eigen_ratio = -1;
                return;
            }

            if(feat_eigen_ratio >= feat_eigen_limit[ftype]){
                pl_feat_map.push_back(ap_centor_direct);
                return;
            }

            // if(layer == 3)
            if(layer == 4)
                return;

            octo_state = 1;
            // All points in sliding window should be put into subvoxel
            frame_head = 0;
        }

        int leafnum;
        uint a_size;
        for(int i=frame_head; i<OCTO_TREE::voxel_windowsize; i++){

            a_size = plvec_tran[i]->size();
            for(uint j=0; j<a_size; j++)
            {
                int xyz[3] = {0, 0, 0};
                for(uint k=0; k<3; k++)
                    if((*plvec_tran[i])[j][k] > voxel_center[k])
                        xyz[k] = 1;

                leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];  // fixme : 4x+2y+z? <8
                if(leaves[leafnum] == nullptr){
                    leaves[leafnum] = new OCTO_TREE(ftype, capacity);
                    leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
                    leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
                    leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
                    leaves[leafnum]->quater_length = quater_length / 2;
                }
                leaves[leafnum]->plvec_orig[i]->push_back((*plvec_orig[i])[j]);
                leaves[leafnum]->plvec_tran[i]->push_back((*plvec_tran[i])[j]);
            }
        }

        if(layer != 0){
            for(int i=frame_head; i<OCTO_TREE::voxel_windowsize; i++)
                if(plvec_orig[i]->size() != 0){  // clear

                    vector<Eigen::Vector3d>().swap(*plvec_orig[i]);
                    vector<Eigen::Vector3d>().swap(*plvec_tran[i]);
                }
        }

        layer++;
        for(uint i=0; i<8; i++)  // recursive
            if(leaves[i] != nullptr)
                leaves[i]->recut(layer, frame_head, pl_feat_map);

    }

    /// marginalize scans in sliding-window (assume margi_size is 5)
    /// \input layer : recursive iterate
    /// \input margi_size
    /// \input q_poses
    /// \input t_poses
    /// \input window_base
    /// \output pl_feat_map
    void marginalize(int layer, int margi_size, vector<Eigen::Quaterniond> &q_poses,
                     vector<Eigen::Vector3d> &t_poses, int window_base,
                     pcl::PointCloud<PointType> &pl_feat_map)
    {
        if(octo_state!=1 || layer==0){

            if(octo_state != 1)
                for(int i=0; i<OCTO_TREE::voxel_windowsize; i++)
                    // Update points by new poses
                    plvec_trans_func(*plvec_orig[i], *plvec_tran[i],
                                     q_poses[i+window_base].matrix(), t_poses[i+window_base]);


            // Push front scans into P_fix
            uint a_size;
            if(feat_eigen_ratio > feat_eigen_limit[ftype]){

                for(int i=0; i<margi_size; i++)
                    sig_vec_points.insert(sig_vec_points.end(), plvec_tran[i]->begin(), plvec_tran[i]->end());

                down_sampling_voxel(sig_vec_points, quater_length);

                a_size = sig_vec_points.size();
                sig_vec.tozero();
                sig_vec.sigma_size = a_size;
                for(uint i=0; i<a_size; i++){
                    sig_vec.sigma_vTv += sig_vec_points[i] * sig_vec_points[i].transpose();
                    sig_vec.sigma_vi  += sig_vec_points[i];
                }
            }

            // Clear front scans
            for(int i=0; i<margi_size; i++)
            {
                PL_VEC().swap(*plvec_orig[i]);
                PL_VEC().swap(*plvec_tran[i]);
                // plvec_orig[i].clear(); plvec_orig[i].shrink_to_fit();
            }

            if(layer == 0){
                a_size = 0;
                for(int i=margi_size; i<OCTO_TREE::voxel_windowsize; i++)
                    a_size += plvec_orig[i]->size();

                if(a_size == 0)
                    // Voxel has no points left in sliding-window
                    is2opt = false;
            }

            for(int i=margi_size; i<OCTO_TREE::voxel_windowsize; i++)
            {
                plvec_orig[i]->swap(*plvec_orig[i-margi_size]);
                plvec_tran[i]->swap(*plvec_tran[i-margi_size]);
            }

            if(octo_state != 1)
            {
                points_size = 0;
                for(int i=0; i<OCTO_TREE::voxel_windowsize-margi_size; i++)
                    points_size += plvec_orig[i]->size();

                points_size += sig_vec.sigma_size;
                if(points_size < MIN_PS){
                    feat_eigen_ratio = -1;
                    return;
                }

                calc_eigen();

                if(isnan(feat_eigen_ratio)){
                    feat_eigen_ratio = -1;
                    return;
                }

                if(feat_eigen_ratio >= feat_eigen_limit[ftype])
                    pl_feat_map.push_back(ap_centor_direct);

            }
        }

        if(octo_state == 1)
        {
            layer++;
            for(int i=0; i<8; i++)
                if(leaves[i] != nullptr)
                    leaves[i]->marginalize(layer, margi_size, q_poses, t_poses, window_base, pl_feat_map);
        }

    }


    void marginalize(int layer, int margi_size, vector<Eigen::Quaterniond> &q_poses,
                     vector<Eigen::Vector3d> &t_poses, int window_base,
                     pcl::PointCloud<PointType> &pl_feat_map,
                     bool ifmargin, Eigen::Vector3d plvec_tran_){

        if(octo_state != 1 || layer==0){

            if(octo_state != 1 && !ifmargin)
                for(int i = 0; i < OCTO_TREE::voxel_windowsize; i++)
                    // Update points by new poses
                    plvec_trans_func(*plvec_orig[i], *plvec_tran[i],
                                     q_poses[i+window_base].matrix(), t_poses[i+window_base]);

            // Push front scans into P_fix
            uint a_size;
            if (ifmargin){
                sig_vec_points.emplace_back(plvec_tran_);

                a_size = sig_vec_points.size();

                down_sampling_voxel(sig_vec_points, quater_length);
                sig_vec.tozero();
                sig_vec.sigma_size = a_size;

                for (uint i = 0; i < a_size; i++) {
                    sig_vec.sigma_vTv += sig_vec_points[i] * sig_vec_points[i].transpose();
                    sig_vec.sigma_vi += sig_vec_points[i];
                }

            }else {
//            else if(feat_eigen_ratio > feat_eigen_limit[ftype])

                // Clear front scans
                for (int i = 0; i < margi_size; i++) {
                    PL_VEC().swap(*plvec_orig[i]);
                    PL_VEC().swap(*plvec_tran[i]);
                    // plvec_orig[i].clear(); plvec_orig[i].shrink_to_fit();
                }

                if (layer == 0) {
                    a_size = 0;
                    for (int i = margi_size; i < OCTO_TREE::voxel_windowsize; i++)
                        a_size += plvec_orig[i]->size();
                    if (a_size == 0)
                        // Voxel has no points left in sliding-window
                        is2opt = false;
                }

                for (int i = margi_size; i < OCTO_TREE::voxel_windowsize; i++) {
                    plvec_orig[i]->swap(*plvec_orig[i - margi_size]);
                    plvec_tran[i]->swap(*plvec_tran[i - margi_size]);
                }
            }

            if(octo_state != 1){

                points_size = 0;
                for(int i = 0; i<OCTO_TREE::voxel_windowsize - margi_size; i++)
                    points_size += plvec_orig[i]->size();

                points_size += sig_vec.sigma_size;
                if(points_size < MIN_PS){
                    feat_eigen_ratio = -1;
                    return;
                }

                calc_eigen();

                if(isnan(feat_eigen_ratio)){
                    feat_eigen_ratio = -1;
                    return;
                }

                if(feat_eigen_ratio >= feat_eigen_limit[ftype])
                    pl_feat_map.push_back(ap_centor_direct);
            }
        }

        if(octo_state == 1){
            layer++;
            for(int i=0; i<8; i++)
                if(leaves[i] != nullptr)
                    leaves[i]->marginalize(layer, margi_size, q_poses, t_poses, window_base, pl_feat_map,
                                           ifmargin, plvec_tran_);
        }

    }

    // Used by "traversal_opt"
    void traversal_opt_calc_eigen()
    {
        Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());
        Eigen::Vector3d center(0, 0, 0);

        uint asize;
        for(int i=0; i<OCTO_TREE::voxel_windowsize; i++)
        {
            asize = plvec_tran[i]->size();
            for(uint j=0; j<asize; j++)
            {
                covMat += (*plvec_tran[i])[j] * (*plvec_tran[i])[j].transpose();
                center += (*plvec_tran[i])[j];
            }
        }

        covMat -= center*center.transpose()/sw_points_size;
        covMat /= sw_points_size;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
        feat_eigen_ratio_test = (saes.eigenvalues()[2] / saes.eigenvalues()[ftype]);

        normal_dir = saes.eigenvectors().col(ftype);
    }

    // Push voxel into "opt_lsv" (LM optimizer)
    void traversal_opt(LM_SLWD_VOXEL &opt_lsv)
    {
        if(octo_state != 1){

            sw_points_size = 0;
            for(int i=0; i<OCTO_TREE::voxel_windowsize; i++)
                sw_points_size += plvec_orig[i]->size();

            if(sw_points_size < MIN_PS)
                return;

            traversal_opt_calc_eigen();

            if(isnan(feat_eigen_ratio_test))
                return;

            if(feat_eigen_ratio_test > opt_feat_eigen_limit[ftype])
                opt_lsv.push_voxel(plvec_orig, sig_vec, ftype, normal_dir);
        }
        else{

            for(int i=0; i<8; i++)
                if(leaves[i] != nullptr)
                    leaves[i]->traversal_opt(opt_lsv);
        }
    }

};

int OCTO_TREE::voxel_windowsize = 0;

#endif //BALM_BALM_NEWEQUA_HPP
