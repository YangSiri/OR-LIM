//
/// Created by cyz on 2022/6/17.
//
/// Linear Interpolation Pose Optimization with Analytical Derivatives
//

#ifndef BALM_BALMCLASS_CT_JACOB_HPP
#define BALM_BALMCLASS_CT_JACOB_HPP

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

// Similar with PCL voxel grid filter
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
    if (orig_size > 100000)
        cout << "[Debug] plvec_trans_func->orig_size: " << orig_size << endl;
    tran.resize(orig_size);
    Eigen::Vector3d pt;
    Eigen::Quaterniond rot;

    for(uint i=0; i<orig_size; i++){

//        transformPoint_CT(R1, R2, t1, t2, orig[i].topRows(3), pt, orig[i](3));
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
    pcl::PointCloud<PointType> cloud_per_frame;

    vector<int> lam_types; // 0 surf, 1 line
    vector<double> feaScores; // feature scores
    double* feaRes_perFrame;  // residuals for per frame
    int* num_of_pts_per_frame;  //
    bool* converged_perFrame;  //
    vector<SIG_VEC_CLASS> sig_vecs;
    vector<vector<Eigen::Vector4d>* > plvec_voxels;
    vector<vector<int>* > slwd_nums;
    int map_refine_flag;
    mutex my_mutex;
    int maxResPos;

    bool ifdebug = true;

    LM_SLWD_VOXEL(int ss, int fn, int thnum): slwd_size(ss), filternum(fn), thd_num(thnum)
    {
        so3_poses.resize(ss); t_poses.resize(ss);
        so3_poses_temp.resize(ss); t_poses_temp.resize(ss);
        jac_leng = 6*ss;
        corn_less = 0.1;
        map_refine_flag = 0;

        feaRes_perFrame = new double[ss];
        converged_perFrame = new bool[ss];
        num_of_pts_per_frame = new int[ss];

    }

    // Decide the number of points used for optimization per frame
    void downsample(vector<Eigen::Vector4d> &plvec_orig, int cur_frame,
                    vector<Eigen::Vector4d> &plvec_voxel,
                    vector<int> &slwd_num, int filternum2use)
    {
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

        for(int i=0; i < filternum2use; i++){

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
    void push_voxel(vector<vector<Eigen::Vector4d>*> &plvec_orig, SIG_VEC_CLASS &sig_vec, int lam_type, double feaWeight)
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
        for(int i=0; i<slwd_size; i++)
            if(!plvec_orig[i]->empty())
                downsample(*plvec_orig[i], i, *plvec_voxel, *slwd_num, filternum2use);

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
        feaScores.push_back(feaWeight);
        sig_vecs.push_back(sig_vec); // history points out of sliding window
    }

    // Calculate Hessian, Jacobian, residual
    void acc_t_evaluate(vector<SO3> &so3_ps, vector<Eigen::Vector3d> &t_ps, int head, int end,
                        Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual){

        Hess.setZero(); JacT.setZero(); residual = 0;
        Eigen::MatrixXd _hess(Hess);
        Eigen::MatrixXd _jact(JacT);

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
            uint backnum = plvec_voxel.size();

            Eigen::Vector3d vec_tran;
            vector<Eigen::Vector3d> plvec_back(backnum);
            // derivative point to T (R, t)
            vector<Eigen::Matrix3d> point_xis(backnum);  // SO3 jacob with reference to the (back) pose R2
            vector<Eigen::Matrix3d> point_xis_bef(backnum);  // SO3 jacob with reference to the front pose R1
            float* ratios = new float[backnum];

            Eigen::Vector3d centor(Eigen::Vector3d::Zero());
            Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());
            Eigen::Quaterniond R1, R2, R;
            double **jacobians = new double *[3];
            for (int i = 0; i < 2; ++i)
                jacobians[i] = new double[12];
            jacobians[2] = new double[3];

            for(uint i=0; i<backnum; i++){

                if (slwd_num[i] == 0 ) continue;
                R1 = so3_ps[slwd_num[i]-1].unit_quaternion();
                R2 = so3_ps[slwd_num[i]].unit_quaternion();
                ratios[i] = plvec_voxel[i][3];

                QuaternionCostFunctor::evaluateJacobAnalytically(R1, R2, R, ratios[i], jacobians);
                Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_0(jacobians[0]);
                Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_1(jacobians[1]);
//                vec_tran = (R1.slerp(ratios[i], R2)) * plvec_voxel[i].topRows(3);
                vec_tran = R * plvec_voxel[i].topRows(3);
                // left multiplication instead of right multiplication in paper
                point_xis[i] = -SO3::hat(vec_tran);  // eq. (10)
//                plvec_back[i] = vec_tran + t_ps[slwd_num[i]]; // after trans

                point_xis_bef[i] = point_xis[i] * jacobian_0.block<3,3>(0,0);
                point_xis[i]= (point_xis[i]*jacobian_1.block<3,3>(0,0)).eval();

                if (isnan(point_xis_bef[i](0,0))){  // FixMe: nan in Jacob!?
                    point_xis_bef[i] = Eigen::Matrix3d::Identity();
                    cout << "[ WARN ] Nan point " << i << " in frame " << slwd_num[i] << " with ratio " << ratios[i]
                         << " point_xis_bef\n" << jacobian_0 << endl;
                }
                if (isnan(point_xis[i](0,0))){  // FixMe: nan in Jacob!?
                    point_xis[i] = Eigen::Matrix3d::Identity();
                    cout << "[ WARN ] Nan point " << i << " in frame " << slwd_num[i] << " with ratio " << ratios[i]
                         << "point_xis\n" << jacobian_1 << endl;
                }

                plvec_back[i](0) = vec_tran(0) + (1-ratios[i]) * t_ps[slwd_num[i]-1](0) + ratios[i] * t_ps[slwd_num[i]](0);
                plvec_back[i](1) = vec_tran(1) + (1-ratios[i]) * t_ps[slwd_num[i]-1](1) + ratios[i] * t_ps[slwd_num[i]](1);
                plvec_back[i](2) = vec_tran(2) + (1-ratios[i]) * t_ps[slwd_num[i]-1](2) + ratios[i] * t_ps[slwd_num[i]](2);

                centor += plvec_back[i];
                covMat += plvec_back[i] * plvec_back[i].transpose();

            }

            double N_points = backnum + sig_vec.sigma_size;
            centor += sig_vec.sigma_vi;
            covMat += sig_vec.sigma_vTv;

            covMat = covMat - centor*centor.transpose()/N_points;
            covMat = covMat / N_points;
            centor = centor / N_points;

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
            Eigen::Vector3d eigen_value = saes.eigenvalues();

            Eigen::Matrix3d U = saes.eigenvectors();
            Eigen::Vector3d u[3]; // eigenvectors
            for(int j=0; j<3; j++)
                u[j] = U.block<3, 1>(0, j);

            // Jacobian matrix
            Eigen::Matrix3d ukukT = u[k] * u[k].transpose();
            Eigen::Vector3d vec_Jt;
            for(uint i=0; i<backnum; i++){
                if (slwd_num[i] == 0 ) continue;

                plvec_back[i] = plvec_back[i] - centor;
                vec_Jt = 2.0/N_points * ukukT * plvec_back[i];  // eq. (6)
                // T1
                _jact.block<3, 1>(6*(slwd_num[i]-1)+3, 0) += (1-ratios[i]) * vec_Jt;  // translation
                _jact.block<3, 1>(6*(slwd_num[i]-1), 0) += point_xis_bef[i].transpose() * vec_Jt;  // rot
                // T2
                _jact.block<3, 1>(6*slwd_num[i]+3, 0) += ratios[i] * vec_Jt;
                _jact.block<3, 1>(6*slwd_num[i], 0) += point_xis[i].transpose() * vec_Jt;

//                if (isnan(_jact.block<6, 1>(6*slwd_num[i]-3, 0)(0,0)) || isnan(_jact.block<6, 1>(6*slwd_num[i]-3, 0)(0,3)))
//                cout << "_jact.block<"<< 6*slwd_num[i]-3 << "," << 0 << ">(6 X 1) \n"
//                     << _jact.block<6, 1>(6*slwd_num[i]-3, 0).transpose()  << endl;
            }

            // Hessian matrix
            Eigen::Matrix3d Hessian33;
            Eigen::Matrix3d C_k;
            vector<Eigen::Matrix3d> C_k_np(3);
            for(uint i=0; i<3; i++){
                if(i == k){
                    C_k_np[i].setZero();
                    continue;
                }
                Hessian33 = u[i]*u[k].transpose();
                // part of F matrix in paper
                C_k_np[i] = -1.0/N_points/(eigen_value[i]-eigen_value[k]) * (Hessian33 + Hessian33.transpose()); // F_(m,n)
            }

            Eigen::Matrix3d h33;
            Eigen::Matrix<double, 6, 6> h66;
            uint rownum, colnum;
            for(uint j=0; j<backnum; j++){
                if (slwd_num[j] == 0 ) continue;
                colnum = 6*slwd_num[j];

                for(int f=0; f<3; f++)
                    C_k.block<1, 3>(f, 0) = plvec_back[j].transpose() * C_k_np[f];  // F matrix
                C_k = U * C_k;  // UF
                // block matrix operation, half Hessian matrix
                for(uint i=j; i<backnum; i++){
                    if (slwd_num[i] == 0 ) continue;
//                    if (slwd_num[i] == 0 || (maxResPos>0 && slwd_num[i]==maxResPos)) continue;
                    rownum = 6*slwd_num[i];

                    Hessian33 = u[k]*(plvec_back[i]).transpose()*C_k + u[k].dot(plvec_back[i])*C_k;
                    if(i == j)
                        Hessian33 += (N_points-1)/N_points * ukukT;
                    else
                        Hessian33 -= 1.0/N_points * ukukT;

                    Hessian33 = 2.0/N_points * Hessian33; // Hessian matrix of lambda and point Eq. (7)

                    // TODO Three kinds of situations here:
                    /// 1. different point of same frame;
                    /// 2. same point of same frame;
                    /// 3. different point of different frame;
                    if(rownum == colnum && i!=j)  /// case 1
                    {
                        /// matrix in block
                        //    |A B|
                        //    |C D|6X6

                        /// T1
                        _hess.block<3, 3>(rownum-3, colnum-3) += (1-ratios[i])*(1-ratios[j])*
                                                                 (Hessian33 + Hessian33.transpose());

                        h33 = (point_xis_bef[i].transpose()) * Hessian33;
                        _hess.block<3, 3>(rownum-6, colnum-3) += (1-ratios[i])*h33;
                        _hess.block<3, 3>(rownum-3, colnum-6) += (1-ratios[i])*h33.transpose();

                        h33 = Hessian33 * point_xis_bef[j];
                        _hess.block<3, 3>(rownum-3, colnum-6) += (1-ratios[j])*h33;
                        _hess.block<3, 3>(rownum-6, colnum-3) += (1-ratios[j])*h33.transpose();
                        h33 = (point_xis_bef[i].transpose()) * h33;
                        _hess.block<3, 3>(rownum-6, colnum-6) += h33 + h33.transpose();

                        /// T2
                        _hess.block<3, 3>(rownum+3, colnum+3) += ratios[i]*ratios[j]*(Hessian33 + Hessian33.transpose());

                        h33 = (point_xis[i].transpose())*Hessian33;
                        _hess.block<3, 3>(rownum, colnum+3) +=  ratios[i]*h33;
                        _hess.block<3, 3>(rownum+3, colnum) += ratios[i]*h33.transpose();

                        h33 = Hessian33 * point_xis[j];
                        _hess.block<3, 3>(rownum+3, colnum) += ratios[j]*h33;
                        _hess.block<3, 3>(rownum, colnum+3) += ratios[j]*h33.transpose();
                        h33 = (point_xis[i].transpose()) * h33;
                        _hess.block<3, 3>(rownum, colnum) += h33 + h33.transpose();

                        /// downLeft col-6
                        // M_(m,n-1)
                        h66.block<3,3>(0,0) = (point_xis[i].transpose()) * Hessian33 * point_xis_bef[j];
                        h66.block<3,3>(0,3) = (point_xis[i].transpose()) * Hessian33 * (1-ratios[j]);
                        h66.block<3,3>(3,0) = ratios[i] * Hessian33 * point_xis_bef[j];
                        h66.block<3,3>(3,3) = ratios[i] * Hessian33 * (1-ratios[j]);
                        _hess.block<6, 6>(rownum, colnum-6) += h66;
                        // M_(n,m-1)
                        h66.block<3,3>(0,0) += (point_xis[j].transpose()) * (Hessian33.transpose()) * point_xis_bef[i];
                        h66.block<3,3>(0,3) += (point_xis[j].transpose()) * (Hessian33.transpose()) * (1-ratios[i]);
                        h66.block<3,3>(3,0) += ratios[j] * (Hessian33.transpose()) * point_xis_bef[i];
                        h66.block<3,3>(3,3) += ratios[j] * (Hessian33.transpose()) * (1-ratios[i]);
                        _hess.block<6, 6>(rownum, colnum-6) += h66;

//                        _hess.block<3, 3>(rownum+3, colnum-3) += ratios[i]*(1-ratios[j])*Hessian33 +
//                                                                 ratios[j]*(1-ratios[i])*Hessian33.transpose();
//                        // C
//                        h33 = (Hessian33.transpose()) * point_xis_bef[i];
//                        _hess.block<3, 3>(rownum+3, colnum-6) += ratios[j]*h33;
//                        h33 = Hessian33 * point_xis_bef[j];
//                        _hess.block<3, 3>(rownum+3, colnum-6) += ratios[i]*h33;
//                        // B
//                        h33 = (point_xis[j].transpose()) * (Hessian33.transpose());
//                        _hess.block<3, 3>(rownum, colnum-3) += (1-ratios[i])*h33;
//                        h33 = (point_xis[i].transpose()) * Hessian33;
//                        _hess.block<3, 3>(rownum, colnum-3) += (1-ratios[j])*h33;
//                        // A
//                        h33 = (h33 * point_xis_bef[j]).eval();
//                        _hess.block<3, 3>(rownum, colnum-6) += h33;
//                        h33 = (point_xis[j].transpose()) * (Hessian33.transpose()) * point_xis_bef[i];
//                        _hess.block<3, 3>(rownum, colnum-6) += h33;

                        /// topRight row-6
//                        _hess.block<3, 3>(rownum-3, colnum+3) += (1-ratios[i])*ratios[j]*Hessian33 +
//                                                                 (1-ratios[j])*ratios[i]*Hessian33.transpose();
//                        // C
//                        h33 = (Hessian33.transpose()) * point_xis[i];
//                        _hess.block<3, 3>(rownum-3, colnum) += (1-ratios[j])*h33;
//                        h33 = Hessian33 * point_xis[j];
//                        _hess.block<3, 3>(rownum-3, colnum) += (1-ratios[i])*h33;
//                        // B
//                        h33 = (point_xis_bef[j].transpose()) * (Hessian33.transpose());
//                        _hess.block<3, 3>(rownum-6, colnum+3) += ratios[i]*h33;
//                        h33 = (point_xis_bef[i].transpose())*Hessian33;
//                        _hess.block<3, 3>(rownum-6, colnum+3) += ratios[j]*h33;
//                        // A
//                        h33 = (h33 * point_xis[j]).eval();
//                        _hess.block<3, 3>(rownum-6, colnum) += h33;
//                        h33 = (point_xis_bef[j].transpose()) * (Hessian33.transpose()) * point_xis[i];
//                        _hess.block<3, 3>(rownum-6, colnum) += h33;

                        /// symmetric?
//                        _hess.block<6, 6>(rownum, colnum-6) += _hess.block<6, 6>(rownum-6, colnum).transpose();

                    }else{
                        /// (rownum == colnum): case 2
                        // T1
                        _hess.block<3, 3>(rownum-3, colnum-3) += (1-ratios[i]) * (1-ratios[j])*Hessian33;
                        h33 = Hessian33 * point_xis_bef[j];
                        _hess.block<3, 3>(rownum-3, colnum-6) += (1-ratios[j]) *h33;
                        _hess.block<3, 3>(rownum-6, colnum-3) += (1-ratios[i]) *(point_xis_bef[i].transpose())*Hessian33;
                        _hess.block<3, 3>(rownum-6, colnum-6) += (point_xis_bef[i].transpose())*h33;

                        // T2
                        _hess.block<3, 3>(rownum+3, colnum+3) += ratios[i]*ratios[j]*Hessian33;
                        h33 = Hessian33 * point_xis[j];
                        _hess.block<3, 3>(rownum+3, colnum) += ratios[j]*h33;
                        _hess.block<3, 3>(rownum, colnum+3) += ratios[i]*(point_xis[i].transpose())*Hessian33;
                        _hess.block<3, 3>(rownum, colnum) += (point_xis[i].transpose())*h33;

//                        if (rownum != colnum){ /// case 3
                        /// topRight, row-6
//                            _hess.block<3, 3>(rownum-3, colnum+3) += (1-ratios[i])*ratios[j]*Hessian33;
//                            h33 = Hessian33 * point_xis[j];
//                            _hess.block<3, 3>(rownum-3, colnum) += (1-ratios[i])*h33;
//                            _hess.block<3, 3>(rownum-6, colnum+3) += ratios[j]*(point_xis_bef[i].transpose())*Hessian33;
//                            _hess.block<3, 3>(rownum-6, colnum) += (point_xis_bef[i].transpose())*h33;

                        // M_(m-1,n)
                        h66.block<3,3>(0,0) = (point_xis_bef[i].transpose()) * Hessian33 * point_xis[j];
                        h66.block<3,3>(0,3) = (point_xis_bef[i].transpose()) * Hessian33 * ratios[j];
                        h66.block<3,3>(3,0) = (1-ratios[i]) * Hessian33 * point_xis[j];
                        h66.block<3,3>(3,3) = (1-ratios[i]) * Hessian33 * ratios[j];
                        _hess.block<6, 6>(rownum-6, colnum) += h66;

                        /// downLeft, col-6
                        // M_(m,n-1)
                        h66.block<3,3>(0,0) = (point_xis[i].transpose()) * Hessian33 * point_xis_bef[j];
                        h66.block<3,3>(0,3) = (point_xis[i].transpose()) * Hessian33 * (1-ratios[j]);
                        h66.block<3,3>(3,0) = ratios[i] * Hessian33 * point_xis_bef[j];
                        h66.block<3,3>(3,3) = ratios[i] * Hessian33 * (1-ratios[j]);
                        _hess.block<6, 6>(rownum, colnum-6) += h66;

//                            _hess.block<3, 3>(rownum+3, colnum-3) += ratios[i]*(1-ratios[j])*Hessian33;
//                            h33 = Hessian33 * point_xis_bef[j];
//                            _hess.block<3, 3>(rownum+3, colnum-6) += ratios[i]*h33;
//                            _hess.block<3, 3>(rownum, colnum-3) += (1-ratios[j])*(point_xis[i].transpose())*Hessian33;
//                            _hess.block<3, 3>(rownum, colnum-6) += (point_xis[i].transpose())*h33;
//                        }
                    }
//                    cout << "_hess.block< "<< rownum << "," << colnum << " >(12 X 12) \n" <<
//                         _hess.block<12, 12>(rownum-6, colnum-6)  << endl;
                }
            }

            if(k == 1){
                // add weight for line feature
                residual += corn_less*eigen_value[k];
                Hess += corn_less*_hess; JacT += corn_less*_jact;
            }else{
                residual += eigen_value[k];
                Hess += _hess; JacT += _jact;
            }
            for(auto id : slwd_num){
                if (id == 0) continue;
                feaRes_perFrame[id] += feaScores[a] * eigen_value[k];
                num_of_pts_per_frame[id]++;
            }

            _hess.setZero(); _jact.setZero();
        }

        // Hessian is symmetric, copy to save time
        for(int j=0; j<jac_leng; j+=6)
            for(int i=j+6; i<jac_leng; i+=6)
                Hess.block<6, 6>(j, i) = Hess.block<6, 6>(i, j).transpose();
    }

    // Multithread for "acc_t_evaluate"
    void divide_thread(vector<SO3> &so3_ps, vector<Eigen::Vector3d> &t_ps,
                       Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual){

        Hess.setZero(); JacT.setZero(); residual = 0;
        for (int i = 0; i < slwd_size; ++i){
            num_of_pts_per_frame[i] = 0;
            feaRes_perFrame[i] = 0;
        }

        vector<Eigen::MatrixXd> hessians(thd_num, Hess);
        vector<Eigen::VectorXd> jacobians(thd_num, JacT);
        vector<double> resis(thd_num, 0);

        uint gps_size = plvec_voxels.size();
        if(gps_size < (uint)thd_num)
        {
            acc_t_evaluate(so3_ps, t_ps, 0, gps_size, Hess, JacT, residual);
            Hess = hessians[0];
            JacT = jacobians[0];
            residual = resis[0];
            return;
        }

        vector<thread*> mthreads(thd_num);

        double part = 1.0*(gps_size)/thd_num;
        for(int i=0; i<thd_num; i++)
        {
            int np = part*i;
            int nn = part*(i+1);

            mthreads[i] = new thread(&LM_SLWD_VOXEL::acc_t_evaluate, this, ref(so3_ps), ref(t_ps), np, nn,
                                     ref(hessians[i]), ref(jacobians[i]), ref(resis[i]));
        }

        for(int i=0; i<thd_num; i++)
        {
            mthreads[i]->join();
            Hess += hessians[i];
            JacT += jacobians[i];
            residual += resis[i];
            delete mthreads[i];
        }

    }

    // Calculate residual
    void evaluate_only_residual(vector<SO3> &so3_ps, vector<Eigen::Vector3d> &t_ps, double &residual){

        residual = 0;
        uint gps_size = plvec_voxels.size();
        Eigen::Vector3d vec_tran;

        for(uint a=0; a<gps_size; a++){

            uint k = lam_types[a];
            SIG_VEC_CLASS &sig_vec = sig_vecs[a];
            vector<Eigen::Vector4d> &plvec_voxel = *plvec_voxels[a];
            vector<int> &slwd_num = *slwd_nums[a];
            uint backnum = plvec_voxel.size();

            Eigen::Vector3d centor(Eigen::Vector3d::Zero());
            Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());
            Eigen::Quaterniond R1, R2;

            for(uint i=0; i<backnum; i++){

                if (slwd_num[i] == 0) continue;
                R1 = so3_ps[slwd_num[i]-1].unit_quaternion();
                R2 = so3_ps[slwd_num[i]].unit_quaternion();
                double ratio = plvec_voxel[i][3];

//                transformPoint_CT(R1, R2, t_ps[slwd_num[i]-1], t_ps[slwd_num[i]],
//                                  plvec_voxel[i].topRows(3), vec_tran, ratio);

                vec_tran = (R1.slerp(ratio, R2)) * plvec_voxel[i].topRows(3);
                vec_tran(0) += (1-ratio) * t_ps[slwd_num[i]-1](0) + ratio * t_ps[slwd_num[i]](0);
                vec_tran(1) += (1-ratio) * t_ps[slwd_num[i]-1](1) + ratio * t_ps[slwd_num[i]](1);
                vec_tran(2) += (1-ratio) * t_ps[slwd_num[i]-1](2) + ratio * t_ps[slwd_num[i]](2);

                centor += vec_tran;
                covMat += vec_tran * vec_tran.transpose();

                if (ifdebug){
                    PointType pt;
                    pt.x = vec_tran(0);
                    pt.y = vec_tran(1);
                    pt.z = vec_tran(2);
                    pt.intensity = slwd_num[i];  // frame No.
                    pt.curvature = a;  // voxel No.
                    cloud_per_frame.points.emplace_back(pt);
                }
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

        getVoxelCloud();
    }

    // LM process
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

        double u = 0.01, v = 2;
        Eigen::MatrixXd D(jac_leng, jac_leng), Hess(jac_leng, jac_leng);
        Eigen::VectorXd JacT(jac_leng), dxi(jac_leng);

        Eigen::MatrixXd Hess2(jac_leng, jac_leng);
        Eigen::VectorXd JacT2(jac_leng);

        D.setIdentity();
        double residual1, residual2, q;
        bool is_calc_hess = true;

        cv::Mat matA(jac_leng, jac_leng, CV_64F, cv::Scalar::all(0));
        cv::Mat matB(jac_leng, 1, CV_64F, cv::Scalar::all(0));
        cv::Mat matX(jac_leng, 1, CV_64F, cv::Scalar::all(0));

        maxResPos = 0;
        iter_max = 50;  // max iteration
        for(int i=0; i<iter_max; i++){

            if(is_calc_hess){
                // calculate Hessian, Jacobian, residual
                divide_thread(so3_poses, t_poses, Hess, JacT, residual1);
//                for (int f = 0; f < slwd_size; ++f){
////                    cout << "[Debug] pt number per frame: " << num_of_pts_per_frame[f];
//                    feaRes_perFrame[f] /= num_of_pts_per_frame[f];
////                    cout << ", avg Residuals per frame: " << feaRes_perFrame[f] << endl;
//                }
//                feaRes_perFrame[0] = 0;
//                maxResPos = max_element(feaRes_perFrame, feaRes_perFrame+slwd_size) - feaRes_perFrame;
//                cout << "[Debug] max Residuals: " << feaRes_perFrame[maxResPos] << endl;
            }

            D = Hess.diagonal().asDiagonal();
            Hess2 = Hess + u*D;  // Eq.14

            for(int j=0; j<jac_leng; j++){
                matB.at<double>(j, 0) = -JacT(j, 0);
                for(int f=0; f<jac_leng; f++)
                    matA.at<double>(j, f) = Hess2(j, f);
            }

            cv::solve(matA, matB, matX, cv::DECOMP_QR);
            for(int j=0; j<jac_leng; j++)
                dxi(j, 0) = matX.at<double>(j, 0);
            if (isnan(dxi(0, 0))) dxi.setZero();

            for(int j=0; j<slwd_size; j++){
                // left multiplication
                so3_poses_temp[j] = SO3::exp(dxi.block<3, 1>(6*(j), 0)) * so3_poses[j];
                t_poses_temp[j] = t_poses[j] + dxi.block<3, 1>(6*(j)+3, 0);
            }

            // LM
            double q1 = 0.5*(dxi.transpose() * (u*D*dxi-JacT))[0];
            // double q1 = 0.5*dxi.dot(u*D*dxi-JacT);
            evaluate_only_residual(so3_poses_temp, t_poses_temp, residual2);

            q = (residual1-residual2);
            printf("residual_%d: %lf u: %lf v: %lf q: %lf %lf %lf\n", i, residual1, u, v, q/q1, q1, q);

            if(q > 0){

                so3_poses = so3_poses_temp;
                t_poses = t_poses_temp;
                q = q / q1;
                v = 2;
                q = 1 - pow(2*q-1, 3);
                u *= (q<one_three ? one_three:q);
                is_calc_hess = true;
            }else{

                u = u * v;
                v = 2 * v;
                is_calc_hess = false;
            }

            if(fabs(q) < 1e-6){
                cout << "[ LM ] CONVERGED " << endl;
                ifconverged = true;
                break;
            }
            if (isnan(u)) break;
        }

        my_mutex.lock();
        map_refine_flag = 2;
        my_mutex.unlock();
        return ifconverged;
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
        feaScores.clear();

    }

    void getVoxelCloud(){

        if (!cloud_per_frame.empty())
            pcl::io::savePCDFileBinaryCompressed("/home/cyz/workspace/catkin_localizeInHDmap/src/"
                                                 "FAST_LIO_LC-master/FAST-LIO/PCD/submapWithLable.pcd", cloud_per_frame);
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

    OCTO_TREE(int ft, int capa): ftype(ft), capacity(capa){

        octo_state = 0;
        for(int i=0; i<8; i++)
            leaves[i] = nullptr;

        for(int i=0; i<capacity; i++){
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
        for(int i=1; i<OCTO_TREE::voxel_windowsize; i++)
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

            octo_state = 1;  //
            // All points in sliding window should be put into sub-voxel
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

        if(octo_state != 1 || layer == 0 ){

            if(octo_state != 1){
                int i = 0;
                if (window_base == 0) i=1;
                for( ; i<OCTO_TREE::voxel_windowsize; i++)
                    // Update points by new poses
                    plvec_trans_func(*plvec_orig[i], *plvec_tran[i],
                                     q_poses[i+window_base-1], t_poses[i+window_base-1],
                                     q_poses[i+window_base], t_poses[i+window_base]);
                calc_eigen();  // update feature value?
            }

            // Push front scans into P_fix
            uint a_size;
            if(feat_eigen_ratio > feat_eigen_limit[ftype])
            {
                for(int i=1; i<margi_size; i++)
                    sig_vec_points.insert(sig_vec_points.end(), plvec_tran[i]->begin(), plvec_tran[i]->end());

                down_sampling_voxel(sig_vec_points, quater_length);

                a_size = sig_vec_points.size();
                sig_vec.tozero();
                sig_vec.sigma_size = a_size;
                for(uint i=0; i<a_size; i++)
                {
                    sig_vec.sigma_vTv += sig_vec_points[i].topRows(3) * sig_vec_points[i].topRows(3).transpose();
                    sig_vec.sigma_vi  += sig_vec_points[i].topRows(3);
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
                    // Voxel has no points in sliding window
                    is2opt = false;
            }

            for(int i=margi_size; i<OCTO_TREE::voxel_windowsize; i++)
            {
                plvec_orig[i]->swap(*plvec_orig[i-margi_size]);
                plvec_tran[i]->swap(*plvec_tran[i-margi_size]);
            }

            if(octo_state != 1){  // what for?
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
        for(int i=1; i<OCTO_TREE::voxel_windowsize; i++){

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
                opt_lsv.push_voxel(plvec_orig, sig_vec, ftype, feat_eigen_ratio_test);
        }
        else{
            for(int i=0; i<8; i++)
                if(leaves[i] != nullptr)
                    leaves[i]->traversal_opt(opt_lsv);
        }
    }

};

int OCTO_TREE::voxel_windowsize = 0;

#endif //BALM_BALMCLASS_CT_JACOB_HPP
