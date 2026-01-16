///
/// Created by cyz on 2022/6/12.
///
/// Linear Interpolation Pose Optimization with AutoDiff Derivatives in Ceres
///

#ifndef BALM_BALMCLASS_CT_HPP
#define BALM_BALMCLASS_CT_HPP

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

typedef std::vector<Eigen::Vector4d> PL_VEC;  // (x, y, z, time)
typedef pcl::PointXYZINormal PointType;
#define MIN_PS 7
using namespace std;

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
    float xyz[4];
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
            iter->second.xyz[3] += p_c.curvature;
            iter->second.count++;
        }
        else
        {
            M_POINT anp;
            anp.xyz[0] = p_c.x;
            anp.xyz[1] = p_c.y;
            anp.xyz[2] = p_c.z;
            anp.xyz[3] = p_c.curvature;
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

void down_sampling_voxel(PL_VEC &pl_feat, double voxel_size){

    unordered_map<VOXEL_LOC, M_POINT> feat_map;
    uint plsize = pl_feat.size();

    for(uint i=0; i<plsize; i++){

        Eigen::Vector4d &p_c = pl_feat[i];
        double loc_xyz[3];
        for(int j=0; j<3; j++){
            loc_xyz[j] = p_c[j] / voxel_size;
            if(loc_xyz[j] < 0)
                loc_xyz[j] -= 1.0;
        }

        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        auto iter = feat_map.find(position);
        if(iter != feat_map.end()){

            iter->second.xyz[0] += p_c[0];
            iter->second.xyz[1] += p_c[1];
            iter->second.xyz[2] += p_c[2];
            iter->second.xyz[3] += p_c[3];
            iter->second.count++;
        }
        else{
            M_POINT anp;
            anp.xyz[0] = p_c[0];
            anp.xyz[1] = p_c[1];
            anp.xyz[2] = p_c[2];
            anp.xyz[3] = p_c[3];
            anp.count = 1;
            feat_map[position] = anp;
        }
    }

    plsize = feat_map.size();
    pl_feat.resize(plsize);

    uint i = 0; // sampled center point
    for(auto iter=feat_map.begin(); iter!=feat_map.end(); ++iter){
        pl_feat[i][0] = iter->second.xyz[0] / iter->second.count;
        pl_feat[i][1] = iter->second.xyz[1] / iter->second.count;
        pl_feat[i][2] = iter->second.xyz[2] / iter->second.count;
        pl_feat[i][3] = iter->second.xyz[3] / iter->second.count;
        i++;
    }

}

void plvec_trans_func(const vector<Eigen::Vector4d> &orig, vector<Eigen::Vector4d> &tran,
                      const Eigen::Quaterniond &R1, const Eigen::Vector3d &t1,
                      const Eigen::Quaterniond &R2, const Eigen::Vector3d &t2){

    uint orig_size = orig.size();

    tran.resize(orig_size);
    Eigen::Vector3d pt;
    Eigen::Quaterniond rot;

    for(uint i=0; i<orig_size; i++){

        rot = R1.slerp(orig[i](3), R2);
        pt = rot * orig[i].topRows(3);
        pt(0) += (1 - orig[i](3)) * t1(0) + orig[i](3) * t2(0);
        pt(1) += (1 - orig[i](3)) * t1(1) + orig[i](3) * t2(1);
        pt(2) += (1 - orig[i](3)) * t1(2) + orig[i](3) * t2(2);

        tran[i](0) = pt(0);
        tran[i](1) = pt(1);
        tran[i](2) = pt(2);
        tran[i](3) = orig[i](3);
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
void rosmsg2ptype(const sensor_msgs::PointCloud2 &pl_msg, pcl::PointCloud<PointType> &plt){

    pcl::fromROSMsg(pl_msg, plt);
    return;
}


const double one_three = (1.0 / 3.0);
double feat_eigen_limit[2] = {3*3, 2*2};      // marginalized feature threshold
double opt_feat_eigen_limit[2] = {4*4, 3*3};  // lamda_3/lamda_1

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

    vector<int> lam_types; // 0 surf, 1 line
    vector<SIG_VEC_CLASS> sig_vecs;
    vector<vector<Eigen::Vector4d>* > plvec_voxels;
    vector<vector<int>*> slwd_nums;
    int map_refine_flag;
    mutex my_mutex;

    LM_SLWD_VOXEL(int ss, int fn, int thnum): slwd_size(ss), filternum(fn), thd_num(thnum)
    {
        so3_poses.resize(ss); t_poses.resize(ss);
        so3_poses_temp.resize(ss); t_poses_temp.resize(ss);
        jac_leng = 6*ss;
        corn_less = 0.1;
        map_refine_flag = 0;
    }

    // Decide the number of points used for optimization per frame
    void downsample(vector<Eigen::Vector4d> &plvec_orig, int cur_frame,
                    vector<Eigen::Vector4d> &plvec_voxel,
                    vector<int> &slwd_num, int filternum2use){

        uint plsize = plvec_orig.size();
//        cout<< "[Debug] plsize / filternum2use: " << plsize << " / " << filternum2use << endl;

        if(plsize <= (uint)filternum2use){
            for(uint i=0; i<plsize; i++){
                plvec_voxel.push_back(plvec_orig[i]);
                slwd_num.push_back(cur_frame);
            }
            return;
        }

        Eigen::Vector4d center;
        double part = 1.0 * plsize / filternum2use;

        for(int i=0; i<filternum2use; i++){

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
    void push_voxel(const vector<vector<Eigen::Vector4d>*> &plvec_orig, SIG_VEC_CLASS &sig_vec, int lam_type)
    {
        int process_frame_size = 0;
        for(int i=0; i < slwd_size; i++)
            if(!plvec_orig[i]->empty())
                process_frame_size++;

        // Only one scan
        if(process_frame_size <= 1)
            return;

        int filternum2use = filternum;
        if(filternum * process_frame_size < MIN_PS)
            filternum2use = MIN_PS / process_frame_size + 1;

        vector<Eigen::Vector4d> *plvec_voxel = new vector<Eigen::Vector4d>();
        // Frame num in sliding window for each point in "plvec_voxel"
        vector<int> *slwd_num = new vector<int>();
        plvec_voxel->reserve(filternum2use*slwd_size);
        slwd_num->reserve(filternum2use*slwd_size);

        // retain one point for one scan (you can modify)
        for(int i=0; i < slwd_size; i++)
            if(!plvec_orig[i]->empty()){
                downsample(*plvec_orig[i], i, *plvec_voxel, *slwd_num, filternum2use);
            }


        // for(int i=0; i<slwd_size; i++)
        // {
        //   for(uint j=0; j<plvec_orig[i]->size(); j++)
        //   {
        //     plvec_voxel->push_back(plvec_orig[i]->at(j));
        //     slwd_num->push_back(i);
        //   }
        // }

        plvec_voxels.push_back(plvec_voxel); // Push a voxel into optimizer
        slwd_nums.push_back(slwd_num);
        lam_types.push_back(lam_type);
        sig_vecs.push_back(sig_vec); // history points out of sliding window
    }

    // Calculate Hessian, Jacobian, residual
    void acc_t_evaluate(std::unique_ptr<ceres::Problem> &problem, vector<double*> &posesVec,
                        int head, int end, double &residual){

        residual = 0;
        // In program, lambda_0 < lambda_1 < lambda_2
        // For plane, the residual is lambda_0
        // For line, the residual is lambda_0+lambda_1
        // We only calculate lambda_1 here
        for(int a=head; a<end; a++){

            uint k = lam_types[a]; // 0 is surf, 1 is line
            SIG_VEC_CLASS &sig_vec = sig_vecs[a];
            vector<Eigen::Vector4d> &plvec_voxel = *plvec_voxels[a];
            // Position in sliding window for each point in "plvec_voxel"
            vector<int> &slwd_num = *slwd_nums[a];

            auto* factor = new GeometricFeatureFactor(plvec_voxel, slwd_num, sig_vec, k, slwd_size,1);
//            auto* cost_functor = new ceres::DynamicAutoDiffCostFunction<GeometricFeatureFactor>(factor);
            auto* cost_functor = new ceres::DynamicNumericDiffCostFunction<GeometricFeatureFactor>(factor);
            for (int i = 0; i < slwd_size; ++i)
                cost_functor->AddParameterBlock(4);
            for (int i = 0; i < slwd_size; ++i)
                cost_functor->AddParameterBlock(3);

            cost_functor->SetNumResiduals(1);
            problem->AddResidualBlock(cost_functor, new ceres::CauchyLoss(0.3), posesVec);

        }

    }

    // Multithread for "acc_t_evaluate"
    void divide_thread(std::unique_ptr<ceres::Problem> &problem, vector<double*> &posesVec, double &residual){

        residual = 0;

        vector<double> resis(thd_num, 0);

        uint gps_size = plvec_voxels.size();
        if(gps_size < (uint)thd_num){

            acc_t_evaluate(problem, posesVec, 0, gps_size, residual);
            residual = resis[0];
            return;
        }

        vector<thread*> mthreads(thd_num);

        double part = 1.0*(gps_size)/thd_num;
        for(int i=0; i<thd_num; i++){

            int np = part*i;
            int nn = part*(i+1);

            mthreads[i] = new thread(&LM_SLWD_VOXEL::acc_t_evaluate, this,
                                     ref(problem), ref(posesVec), np, nn, ref(resis[i]));
        }

        for(int i=0; i<thd_num; i++){
            mthreads[i]->join();
            residual += resis[i];
            delete mthreads[i];
        }

    }

    // Calculate residual
    void evaluate_only_residual(vector<SO3> &so3_ps, vector<Eigen::Vector3d> &t_ps, double &residual){

        residual = 0;
        uint gps_size = plvec_voxels.size();
        Eigen::Vector3d vec_tran;

        for(uint a=0; a < gps_size; a++){

            uint k = lam_types[a];
            SIG_VEC_CLASS &sig_vec = sig_vecs[a];
            vector<Eigen::Vector4d> &plvec_voxel = *plvec_voxels[a];
            vector<int> &slwd_num = *slwd_nums[a];
            uint backnum = plvec_voxel.size();

            Eigen::Vector3d centor(Eigen::Vector3d::Zero());
            Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());
            Eigen::Quaterniond R1, R2;

            for(uint i=0; i<backnum; i++)
            {
                if (slwd_num[i] == 0) continue;
                R1 = so3_ps[slwd_num[i]-1].unit_quaternion();
                R2 = so3_ps[slwd_num[i]].unit_quaternion();
                double ratio = plvec_voxel[i][3];

                vec_tran = (R1.slerp(ratio, R2)) * plvec_voxel[i].topRows(3);
                vec_tran(0) += (1-ratio) * t_ps[slwd_num[i]-1](0) + ratio * t_ps[slwd_num[i]](0);
                vec_tran(1) += (1-ratio) * t_ps[slwd_num[i]-1](1) + ratio * t_ps[slwd_num[i]](1);
                vec_tran(2) += (1-ratio) * t_ps[slwd_num[i]-1](2) + ratio * t_ps[slwd_num[i]](2);

                centor += vec_tran;
                covMat += vec_tran * vec_tran.transpose();
            }

            double N_points = backnum + sig_vec.sigma_size;
            centor += sig_vec.sigma_vi;
            covMat += sig_vec.sigma_vTv;

            covMat = covMat - centor*centor.transpose()/N_points;
            covMat = covMat / N_points;

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
            Eigen::Vector3d eigen_value = saes.eigenvalues();

            if(k == 1)
                residual += corn_less*eigen_value[k];
            else
                residual += eigen_value[k];
        }
    }

    // ceres numeric jacob
    bool damping_iter(){

        bool ifconverged = false;
        my_mutex.lock();
        map_refine_flag = 1;
        my_mutex.unlock();

        int meaN = plvec_voxels.size();
        if( meaN!= slwd_nums.size() || meaN != lam_types.size() || meaN != sig_vecs.size()){

            printf("size is not equal\n");
            exit(0);
        }

        double residual1, residual2, q;

//        double** posesVec = new double *[slwd_size];  // pose parameters for optimization
        vector<double* > posesVec;
        for (int j = 0; j < slwd_size; ++j) {
//            posesVec[j] = new double[7];
//            posesVec[j][0] = so3_poses[j].x();
//            posesVec[j][1] = so3_poses[j].y();
//            posesVec[j][2] = so3_poses[j].z();
//            posesVec[j][3] = so3_poses[j].w();
//
//            posesVec[j][4] = t_poses[j].x();
//            posesVec[j][5] = t_poses[j].y();
//            posesVec[j][6] = t_poses[j].z();
            posesVec.emplace_back(so3_poses[j].unit_quaternion().coeffs().data());
        }
        for (int j = 0; j < slwd_size; ++j) {
//            posesVec[j] = new double[3];
//            posesVec[j][0] = t_poses[j].x();
//            posesVec[j][1] = t_poses[j].y();
//            posesVec[j][2] = t_poses[j].z();
            posesVec.emplace_back(t_poses[j].data());
        }

        ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
        ceres::Problem::Options problem_options;
        std::unique_ptr<ceres::Problem> problem;
        std::vector<ceres::ResidualBlockId > resiIDs;

        ceres::Solver::Options options;
        options.minimizer_type = ceres::TRUST_REGION;
        //  options.gradient_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
        //  options.function_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        //    options.trust_region_strategy_type = ceres::DOGLEG;
        //    options.dogleg_type = ceres::SUBSPACE_DOGLEG;
        //    options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = thd_num;
        options.max_num_iterations = 25;

        iter_max = 1;  // max iteration
        for(int i=0; i < iter_max; i++){

            problem.reset(new ceres::Problem(problem_options));
            for (int j = 0; j < slwd_size; ++j)
                problem->AddParameterBlock(posesVec[j], 4, q_parameterization);
            for (int j = slwd_size; j < 2*slwd_size; ++j)
                problem->AddParameterBlock(posesVec[j], 3);

//            divide_thread(problem, posesVec, residual1);
            for(int a=0; a < meaN; a++){

                uint k = lam_types[a]; // 0 is surf, 1 is line
                SIG_VEC_CLASS &sig_vec = sig_vecs[a];
                vector<Eigen::Vector4d> &plvec_voxel = *plvec_voxels[a];
                // Position in sliding window for each point in "plvec_voxel"
                vector<int> &slwd_num = *slwd_nums[a];

                auto* factor = new GeometricFeatureFactor(plvec_voxel, slwd_num, sig_vec, k, slwd_size,1);
//                auto* cost_functor = new ceres::DynamicAutoDiffCostFunction<GeometricFeatureFactor>(factor);
                auto* cost_functor = new ceres::DynamicNumericDiffCostFunction<GeometricFeatureFactor>(factor);
                for (int i = 0; i < slwd_size; ++i)
                    cost_functor->AddParameterBlock(4);
                for (int i = 0; i < slwd_size; ++i)
                    cost_functor->AddParameterBlock(3);
//                cost_functor->AddParameterBlock(slwd_size);
                cost_functor->SetNumResiduals(1);

                problem->AddResidualBlock(cost_functor, NULL, posesVec);
            }

            ceres::Solver::Summary summary;
            ceres::Solve(options, problem.get(), &summary);

            evaluate_only_residual(so3_poses, t_poses, residual2);
            cout<< "After optimization residual2: " << residual2 << endl;

//            for (int j = 0; j < slwd_size; ++j) {
//                so3_poses_temp[j].x() = posesVec[j][0];
//                so3_poses_temp[j].y() = posesVec[j][1];
//                so3_poses_temp[j].z() = posesVec[j][2];
//                so3_poses_temp[j].w() = posesVec[j][3];
//            }
//            for (int j = 0; j < slwd_size; ++j) {
//                t_poses_temp[j].x() = posesVec[j+slwd_size][0];
//                t_poses_temp[j].y() = posesVec[j+slwd_size][1];
//                t_poses_temp[j].z() = posesVec[j+slwd_size][2];
//            }
//            evaluate_only_residual(so3_poses_temp, t_poses_temp, residual2);

            q = (residual1-residual2);

            if(fabs(q) < 1e-9){
                cout << "[ LM ] CONVERGED " << endl;
                ifconverged = true;
                break;
            }
        }
//        for (int i = 0; i < 2 * slwd_size; ++i)
//            delete[] posesVec[i];
//        delete[] posesVec;

        my_mutex.lock();
        map_refine_flag = 2;
        my_mutex.unlock();
        return ifconverged;
//        return true;
    }

    // ceres analytical jacob
    bool damping_iter_Analytical(){

        bool ifconverged = false;
        my_mutex.lock();
        map_refine_flag = 1;
        my_mutex.unlock();

        int meaN = plvec_voxels.size();
        if( meaN!= slwd_nums.size() || meaN != lam_types.size() || meaN != sig_vecs.size()){

            printf("size is not equal\n");
            exit(0);
        }

        double residual1, residual2, q;

//        double** posesVec = new double *[slwd_size];  // pose parameters for optimization
        vector<double* > posesVec;
        for (int j = 0; j < slwd_size; ++j){
            posesVec.emplace_back(so3_poses[j].log().data());
            posesVec.emplace_back(t_poses[j].data());
        }
//        for (int j = 0; j < slwd_size; ++j)
//            posesVec.emplace_back(t_poses[j].data());

//        ceres::LocalParameterization *q_parameterization = new EigenQuaternionParameterizationTangent();
        ceres::LocalParameterization *q_parameterization = new ParamSO3();
        ceres::Problem::Options problem_options;
        std::unique_ptr<ceres::Problem> problem;
        std::vector<ceres::ResidualBlockId > resiIDs;

        ceres::Solver::Options options;
        options.minimizer_type = ceres::TRUST_REGION;
        //  options.gradient_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
        //  options.function_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        //    options.trust_region_strategy_type = ceres::DOGLEG;
        //    options.dogleg_type = ceres::SUBSPACE_DOGLEG;
        //    options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.minimizer_progress_to_stdout = true;
        options.num_threads = thd_num;
        options.max_num_iterations = 25;

        iter_max = 1;  // max iteration
        for(int i=0; i<iter_max; i++){

            problem.reset(new ceres::Problem(problem_options));
            for (int j = 0; j < slwd_size; ++j){
                problem->AddParameterBlock(so3_poses[j].log().data(), 3, q_parameterization);
                problem->AddParameterBlock(t_poses[j].data(), 3);
            }
//            for (int j = slwd_size; j < 2*slwd_size; ++j)
//                problem->AddParameterBlock(posesVec[j], 3);

            for(int a=0; a < meaN; a++){

                uint k = lam_types[a]; // 0 is surf, 1 is line
                SIG_VEC_CLASS &sig_vec = sig_vecs[a];
                vector<Eigen::Vector4d> &plvec_voxel = *plvec_voxels[a];
                // Position in sliding window for each point in "plvec_voxel"
                vector<int> &slwd_num = *slwd_nums[a];
                cout<< "[ Building ] voxel " << a << " with "<< plvec_voxel.size() << " points." << endl;

                auto* cost_functor = new GeometricFeatureFactorAnalytical(plvec_voxel, slwd_num, sig_vec, k, slwd_size,1);
                problem->AddResidualBlock(cost_functor, NULL, posesVec);  // FixMe: Duplicate parameter blocks?
            }

            ceres::Solver::Summary summary;
            ceres::Solve(options, problem.get(), &summary);

            evaluate_only_residual(so3_poses, t_poses, residual2);
            cout<< "After optimization residual2: " << residual2 << endl;

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
        uint a_size = plvec_voxels.size();
        for(uint i=0; i<a_size; i++)
        {
            delete (plvec_voxels[i]);
            delete (slwd_nums[i]);
        }

        plvec_voxels.clear();
        slwd_nums.clear();
        sig_vecs.clear();
        lam_types.clear();
    }

};


class OCTO_TREE
{
public:
    static int voxel_windowsize;
    vector<PL_VEC*> plvec_orig;
    vector<PL_VEC*> plvec_tran;
    int octo_state; // 0 is end of tree, 1 is not
    PL_VEC sig_vec_points;  // not used?
    SIG_VEC_CLASS sig_vec;
    int ftype;
    int points_size, sw_points_size;
    double feat_eigen_ratio, feat_eigen_ratio_test;
    PointType ap_centor_direct;
    double voxel_center[3]; // x, y, z
    double quater_length;
    OCTO_TREE* leaves[8];
    bool is2opt;
    int capacity;
    pcl::PointCloud<PointType> root_centors;

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
            for(uint j=0; j<asize; j++){
                covMat += (*plvec_tran[i])[j].topRows(3) * (*plvec_tran[i])[j].topRows(3).transpose();
                center += (*plvec_tran[i])[j].topRows(3);
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
//                cout << "current size: " << leaves[leafnum]->plvec_orig[i] << endl;
            }
        }

        if(layer != 0){
            for(int i=frame_head; i<OCTO_TREE::voxel_windowsize; i++)
                if(plvec_orig[i]->size() != 0){  // clear

                    vector<Eigen::Vector4d>().swap(*plvec_orig[i]);
                    vector<Eigen::Vector4d>().swap(*plvec_tran[i]);
                }
        }

        layer++;
        for(uint i=0; i<8; i++)  // recursive
            if(leaves[i] != nullptr)
                leaves[i]->recut(layer, frame_head, pl_feat_map);

    }

    /// marginalize scans in sliding-window
    /// \input layer : recursive iterate
    /// \input margi_size
    /// \input q_poses
    /// \input t_poses
    /// \input window_base
    /// \output pl_feat_map
    void marginalize(int layer, int margi_size, vector<Eigen::Quaterniond> &q_poses,
                     vector<Eigen::Vector3d> &t_poses, int window_base,
                     pcl::PointCloud<PointType> &pl_feat_map){

        if(octo_state!=1 || layer==0){

            if(octo_state != 1){
                int i = 0;
                if (window_base == 0) i=1;
                for( ; i<OCTO_TREE::voxel_windowsize; i++)
                    // Update points by new poses
                    plvec_trans_func(*plvec_orig[i], *plvec_tran[i],
                                     q_poses[i+window_base-1], t_poses[i+window_base-1],
                                     q_poses[i+window_base], t_poses[i+window_base]);
            }

            // Push front scans into P_fix
            uint a_size;
            if(feat_eigen_ratio > feat_eigen_limit[ftype])
            {
                for(int i=0; i<margi_size; i++)
                    sig_vec_points.insert(sig_vec_points.end(), plvec_tran[i]->begin(), plvec_tran[i]->end());

                down_sampling_voxel(sig_vec_points, quater_length);

                a_size = sig_vec_points.size();
                sig_vec.tozero();
                sig_vec.sigma_size = a_size;
                for(uint i=0; i<a_size; i++){
                    sig_vec.sigma_vTv += sig_vec_points[i].topRows(3) * sig_vec_points[i].topRows(3).transpose();
                    sig_vec.sigma_vi  += sig_vec_points[i].topRows(3);
                }
            }

            // Clear front scans
            for(int i=0; i<margi_size; i++)
            {
                PL_VEC().swap(*plvec_orig[i]);
                PL_VEC().swap(*plvec_tran[i]);  // FixMe SIG ?!
                // plvec_orig[i].clear(); plvec_orig[i].shrink_to_fit();
            }

            if(layer == 0){
                a_size = 0;
                for(int i=margi_size; i<OCTO_TREE::voxel_windowsize; i++)
                    a_size += plvec_orig[i]->size();

                if(a_size == 0)
                    // Voxel has no points in slidingwindow
                    is2opt = false;
            }

            for(int i=margi_size; i<OCTO_TREE::voxel_windowsize; i++){
                plvec_orig[i]->swap(*plvec_orig[i-margi_size]);
                plvec_tran[i]->swap(*plvec_tran[i-margi_size]);
            }

            if(octo_state != 1){

                points_size = 0;
                for(int i=0; i<OCTO_TREE::voxel_windowsize-margi_size; i++)
                    points_size += plvec_orig[i]->size();

                points_size += sig_vec.sigma_size;
                if(points_size < MIN_PS)
                {
                    feat_eigen_ratio = -1;
                    return;
                }

                calc_eigen();

                if(isnan(feat_eigen_ratio))
                {
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
                    leaves[i]->marginalize(layer, margi_size, q_poses, t_poses, window_base, pl_feat_map);
        }

    }

    // Used by "traversal_opt"
    void traversal_opt_calc_eigen()
    {
        Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());
        Eigen::Vector3d center(0, 0, 0);

        uint asize;
        for(int i=0; i<OCTO_TREE::voxel_windowsize; i++){

            asize = plvec_tran[i]->size();
            for(uint j=0; j<asize; j++){
                covMat += (*plvec_tran[i])[j].topRows(3) * (*plvec_tran[i])[j].topRows(3).transpose();
                center += (*plvec_tran[i])[j].topRows(3);
//                cout<< "[Debug] (*plvec_tran[i])[j]: " << (*plvec_tran[i])[j][3] << endl;
            }
        }

        covMat -= center*center.transpose()/sw_points_size;
        covMat /= sw_points_size;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
        feat_eigen_ratio_test = (saes.eigenvalues()[2] / saes.eigenvalues()[ftype]);
    }

    // Push voxel into "opt_lsv" (LM optimizer)
    void traversal_opt(LM_SLWD_VOXEL &opt_lsv){

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
                opt_lsv.push_voxel(plvec_orig, sig_vec, ftype);
        }
        else{
            for(int i=0; i<8; i++)
                if(leaves[i] != nullptr)
                    leaves[i]->traversal_opt(opt_lsv);
        }
    }

};

int OCTO_TREE::voxel_windowsize = 0;

#endif //BALM_BALMCLASS_CT_HPP
