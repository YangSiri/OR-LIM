//
// Created by cyz on 2022/6/12.
//

#ifndef BALM_CERESFACTORS_HPP
#define BALM_CERESFACTORS_HPP

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/SVD>

#include "myso3/myso3.hpp"

using namespace std;

///////////////////// Rotation Slerp with Jacobians ////////////////////
/// From https://www.cnblogs.com/JingeTU/p/13526943.html //////////////
class QuaternionCostFunctor {
public:
    QuaternionCostFunctor(const Eigen::Quaterniond &_ci_q_w) : ci_q_w_{_ci_q_w} {}

    template <typename T>
    bool operator()(const T *const _c0_q_w, const T *const _c1_q_w,
                    const T *const _alpha, T *_e) const {

        const Eigen::Quaternion<T> c0_q_w(_c0_q_w);
        const Eigen::Quaternion<T> c1_q_w(_c1_q_w);

        const Eigen::Quaternion<T> delta_qua = c1_q_w * c0_q_w.inverse();
        Eigen::AngleAxis<T> delta_aa(delta_qua);
        delta_aa.angle() *= _alpha[0];

        const Eigen::Quaternion<T> ci_q_w_p =
                Eigen::Quaternion<T>(delta_aa) * c0_q_w;

        const Eigen::Quaternion<T> ci_q_w(
                static_cast<T>(ci_q_w_.w()), static_cast<T>(ci_q_w_.x()),
                static_cast<T>(ci_q_w_.y()), static_cast<T>(ci_q_w_.z()));

        const Eigen::Quaternion<T> e_q = ci_q_w_p * ci_q_w.inverse();

        const Eigen::AngleAxis<T> e_aa(e_q);

        Eigen::Map<Eigen::Matrix<T, 3, 1>> e(_e);

        e = e_aa.axis() * e_aa.angle();

        return true;
    }

    void evaluateAnalytically(const double *const _c0_q_w,
                              const double *const _c1_q_w,
                              const double *const _alpha, double *_e,
                              double **_jacobians) const {

        const Eigen::Quaternion<double> c0_q_w(_c0_q_w);
        const Eigen::Quaternion<double> c1_q_w(_c1_q_w);

        const Eigen::Quaternion<double> delta_qua = c1_q_w * c0_q_w.inverse();
        Eigen::AngleAxis<double> delta_aa(delta_qua);
        delta_aa.angle() *= _alpha[0];

        const Eigen::Quaternion<double> ci_q_w_p =
                Eigen::Quaternion<double>(delta_aa) * c0_q_w;

        const Eigen::Quaternion<double> e_q = ci_q_w_p * ci_q_w_.inverse();

        const Eigen::AngleAxis<double> e_aa(e_q);

        Eigen::Map<Eigen::Matrix<double, 3, 1>> e(_e);

        e = e_aa.axis() * e_aa.angle();

        if (_jacobians != nullptr) {

            const Eigen::Vector3d tau_axis = delta_aa.axis();
            const double tau_angle = delta_aa.angle() / _alpha[0];
            if (_jacobians[0] != nullptr) { /// c0_q_w
                Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(
                        _jacobians[0]);
                J.setZero();
                J.block(0, 0, 3, 3) = -(_alpha[0] - 1) *
                                      J_l((_alpha[0] - 1) * tau_angle, tau_axis) *
                                      J_r_inv(tau_angle, tau_axis);
            }
            if (_jacobians[1] != nullptr) { /// c1_q_w
                Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(
                        _jacobians[1]);
                J.setZero();
                J.block(0, 0, 3, 3) = _alpha[0] *
                                      J_l(_alpha[0] * tau_angle, tau_axis) *
                                      J_l_inv(tau_angle, tau_axis);
            }
            if (_jacobians[2] != nullptr) { /// alpha
                Eigen::Map<Eigen::Matrix<double, 3, 1>> J(_jacobians[2]);
                J = J_l(_alpha[0] * tau_angle, tau_axis) * tau_angle * tau_axis;
            }
        }
    }

    static void evaluateJacobAnalytically(const Eigen::Quaternion<double> &c0_q_w,
                                          const Eigen::Quaternion<double> &c1_q_w,
                                          Eigen::Quaternion<double> &ci_q_w_p,
                                          const double _alpha,
                                          double **_jacobians) {

//        const Eigen::Quaternion<double> c0_q_w(_c0_q_w);
//        const Eigen::Quaternion<double> c1_q_w(_c1_q_w);

        const Eigen::Quaternion<double> delta_qua = c1_q_w * c0_q_w.inverse();
        Eigen::AngleAxis<double> delta_aa(delta_qua);
        delta_aa.angle() *= _alpha;

        ci_q_w_p = Eigen::Quaternion<double>(delta_aa) * c0_q_w;

        if (_jacobians != nullptr) {

            const Eigen::Vector3d tau_axis = delta_aa.axis();
            const double tau_angle = delta_aa.angle() / _alpha;
            if (_jacobians[0] != nullptr) { /// c0_q_w
                Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(
                        _jacobians[0]);
                J.setZero();
                J.block(0, 0, 3, 3) = -(_alpha - 1) *
                                      J_l((_alpha - 1) * tau_angle, tau_axis) *
                                      J_r_inv(tau_angle, tau_axis);
            }
            if (_jacobians[1] != nullptr) { /// c1_q_w
                Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(
                        _jacobians[1]);
                J.setZero();
                J.block(0, 0, 3, 3) = _alpha *
                                      J_l(_alpha * tau_angle, tau_axis) *
                                      J_l_inv(tau_angle, tau_axis);
            }
            if (_jacobians[2] != nullptr) { /// alpha
                Eigen::Map<Eigen::Matrix<double, 3, 1>> J(_jacobians[2]);
                J = J_l(_alpha * tau_angle, tau_axis) * tau_angle * tau_axis;
            }
        }
    }

    static bool GlobalToLocal(const double *x, double *jacobian) {

        const double qw = x[3];
        const double qx = x[0];
        const double qy = x[1];
        const double qz = x[2];
        jacobian[0] = qw,  jacobian[1] = qz,  jacobian[2] = -qy;
        jacobian[3] = -qz, jacobian[4] = qw,  jacobian[5] = qx;
        jacobian[6] = qy,  jacobian[7] = -qx, jacobian[8] = qw;
        jacobian[9] = -qx, jacobian[10] = -qy, jacobian[11] = -qz;
        return true;
    }

    static inline Eigen::Matrix3d skew(const Eigen::Vector3d &_v) {
        Eigen::Matrix3d res;
        res.setZero();
        res(0, 1) = -_v[2], res(0, 2) = _v[1], res(1, 2) = -_v[0];
        res(1, 0) = _v[2], res(2, 0) = -_v[1], res(2, 1) = _v[0];
        return res;
    }

    static Eigen::Matrix3d J_l(const double _angle,
                               const Eigen::Vector3d &_axis) {
        /// (145)
        Eigen::Matrix3d res =
                Eigen::Matrix3d::Identity() +
                (1 - std::cos(_angle)) / (_angle * _angle) * skew(_angle * _axis) +
                (_angle - std::sin(_angle)) / (_angle * _angle * _angle) *
                skew(_angle * _axis) * skew(_angle * _axis);
        if (isnan(res(0,0))) cout << " Nan in J_l " << endl;
        return res;
    }

    static Eigen::Matrix3d J_l_inv(const double _angle,
                                   const Eigen::Vector3d &_axis) {
        /// (146)
        Eigen::Matrix3d res =
                Eigen::Matrix3d::Identity() - 0.5 * skew(_angle * _axis) +
                (1 / (_angle * _angle) -
                 (1 + std::cos(_angle)) / (2 * _angle * std::sin(_angle))) *
                skew(_angle * _axis) * skew(_angle * _axis);
        if (isnan(res(0,0))) cout << " Nan in J_l_inv " << endl;
        return res;
    }

    static Eigen::Matrix3d J_r(const double _angle,
                               const Eigen::Vector3d &_axis) {
        return J_l(_angle, _axis).transpose();
        /// (143)
        Eigen::Matrix3d res =
                Eigen::Matrix3d::Identity() -
                (1 - std::cos(_angle)) / (_angle * _angle) * skew(_angle * _axis) +
                (_angle - std::sin(_angle)) / (_angle * _angle * _angle) *
                skew(_angle * _axis) * skew(_angle * _axis);
        return res;
    }

    static Eigen::Matrix3d J_r_inv(const double _angle,
                                   const Eigen::Vector3d &_axis) {
        return J_l_inv(_angle, _axis).transpose();
        /// (144)
        Eigen::Matrix3d res =
                Eigen::Matrix3d::Identity() + 0.5 * skew(_angle * _axis) +
                (1 / (_angle * _angle) -
                 (1 + std::cos(_angle)) / (2 * _angle * std::sin(_angle))) *
                skew(_angle * _axis) * skew(_angle * _axis);
        return res;
    }

private:
    const Eigen::Quaterniond ci_q_w_;
};


///////////// Pose Linear Interpolation without Jacobians ////////////////////
class GeometricFeatureFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static constexpr int NumResiduals() { return 1; }

    GeometricFeatureFactor(const vector<Eigen::Vector4d>& pc,
                           const vector<int>& slwd_num,
                           const SIG_VEC_CLASS& sig_vec,
                           int feaType, int winN, double weight)
            : measurement_(pc), slwd_Nos_(slwd_num),
              info_fix_(sig_vec), feaType_(feaType),
              winSize(winN), weight_(weight) {

        backnum = measurement_.size();
    }

    template <class T>
    bool operator()(T const* const* parameters, T* sResiduals) const {

        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using QuaT = Eigen::Quaternion<T>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;
        using MatXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

        Vec3T centor(Vec3T::Zero());
        Mat3T covMat(Mat3T::Zero());
        vector<Vec3T> plvec_back;
        plvec_back.resize(backnum);

        for(uint i = 0; i < backnum; i++){

            int frameNo = slwd_Nos_[i];
            if (frameNo == 0) continue;
//            cout << "[Debug] ratio: " << measurement_[i][3] << endl;
//            ceres::AngleAxisRotatePoint();

            Eigen::Map<QuaT >  R_1(const_cast<T*>(parameters[frameNo-1]));
//            Eigen::Map<Vec3T const>  p_1 (parameters[frameNo-1+4]);
            Eigen::Map<Vec3T const>  p_1 (parameters[winSize + frameNo-1]);
            Eigen::Map<QuaT const>  R_2(parameters[frameNo]);
//            Eigen::Map<Vec3T const>  p_2 (parameters[frameNo+4]);
            Eigen::Map<Vec3T const>  p_2 (parameters[winSize + frameNo]);

//            QuaT R = R_1.slerp(T(measurement_[i][3]), R_2);
            // for numericDiffCostFunc
            QuaT R1_norm = R_1; R1_norm.normalize();
            QuaT R2_norm = R_2; R2_norm.normalize();
            QuaT R = R1_norm.slerp(T(measurement_[i][3]), R2_norm);

            R.normalize();
            plvec_back[i] = R * measurement_[i].topRows(3).template cast<T>();
            plvec_back[i](0) += T(1-measurement_[i][3]) * p_1(0) + T(measurement_[i][3]) * p_2(0);
            plvec_back[i](1) += T(1-measurement_[i][3]) * p_1(1) + T(measurement_[i][3]) * p_2(1);
            plvec_back[i](2) += T(1-measurement_[i][3]) * p_1(2) + T(measurement_[i][3]) * p_2(2);

            centor += plvec_back[i];
            covMat += plvec_back[i] * plvec_back[i].transpose();
        }

        T N_points = T(backnum) + T(info_fix_.sigma_size);
        centor += info_fix_.sigma_vi.cast<T>();
        covMat += info_fix_.sigma_vTv.cast<T>();

        covMat = covMat - centor*centor.transpose()/N_points;
        covMat = covMat / N_points;
        centor = centor / N_points;

        Eigen::JacobiSVD<MatXT> svd(covMat, Eigen::ComputeThinU|Eigen::ComputeThinV);
        Vec3T eigen_value = svd.singularValues();  // decreasing

        Eigen::Map<Eigen::Matrix<T, 1, 1>> residuals(sResiduals);
        residuals.template block<1, 1>(0, 0) = Eigen::Matrix<T, 1, 1>(T(eigen_value(2-feaType_)));
//        residuals.template block<1, 1>(0, 0) = Eigen::Matrix<T, 1, 1>(T(eigen_value(feaType_, feaType_)));

        residuals = T(weight_) * residuals;

        return true;
    }

private:
    vector<Eigen::Vector4d> measurement_;
    vector<int> slwd_Nos_;
    SIG_VEC_CLASS info_fix_;
    int feaType_, backnum, winSize;
    double weight_;

};


// adopted for quaternion param
class EigenQuaternionParameterizationTangent : public ceres::EigenQuaternionParameterization{
public:
    virtual ~EigenQuaternionParameterizationTangent() {}

    virtual bool ComputeJacobian(const double* x, double* jacobian) const{

        Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> J (jacobian);
        J.setZero();
        J.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
    };
};

class GeometricFeatureFactorAnalytical : public ceres::CostFunction{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    GeometricFeatureFactorAnalytical(const vector<Eigen::Vector4d>& pc,
                                     const vector<int>& slwd_num,
                                     const SIG_VEC_CLASS& sig_vec,
                                     int feaType, int winN, double weight)
            : measurement_(pc), slwd_Nos_(slwd_num),
              info_fix_(sig_vec), feaType_(feaType),
              winSize(winN), weight_(weight) {

        backnum = measurement_.size();

        set_num_residuals(1);

        auto* ptr = mutable_parameter_block_sizes();
        ptr->clear();
        for (int i = 0; i < winN; ++i)
            ptr->emplace_back(3);
        for (int i = 0; i < winN; ++i)
            ptr->emplace_back(3);
    }
    virtual ~GeometricFeatureFactorAnalytical() {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const{

        using Vec3T = Eigen::Matrix<double, 3, 1>;
        using QuaT = Eigen::Quaternion<double>;
        using Mat3T = Eigen::Matrix<double, 3, 3>;
        using MatXT = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

        Vec3T centor(Vec3T::Zero());
        Mat3T covMat(Mat3T::Zero());
        vector<Vec3T> plvec_back;
        plvec_back.resize(backnum);
        // derivative point to T (R, t)
        vector<Eigen::Matrix3d> point_xis(backnum);  // SO3 jacob with reference to the (back) pose R2
        vector<Eigen::Matrix3d> point_xis_bef(backnum);  // SO3 jacob with reference to the front pose R1
        double* ratios = new double[backnum];

        double** jacob_R = new double *[3];
        for (int i = 0; i < 2; ++i)
            jacob_R[i] = new double[12];
        jacob_R[2] = new double[3];

        for(uint i = 0; i < backnum; i++){

            int frameNo = slwd_Nos_[i];
            if (frameNo == 0) continue;
//            cout << "[Debug] ratio: " << measurement_[i][3] << endl;
//            ceres::AngleAxisRotatePoint();

//            Eigen::Map<QuaT >  R_1(const_cast<double*>(parameters[frameNo-1]));
//            Eigen::Map<QuaT const>  R_2(parameters[frameNo]);

            SO3 R_1(SO3::exp(Eigen::Map<Vec3T const>(parameters[2*frameNo-2])));
            SO3 R_2(SO3::exp(Eigen::Map<Vec3T const>(parameters[2*frameNo])));
            Eigen::Map<Vec3T const>  p_1 (parameters[2*frameNo-1]);
            Eigen::Map<Vec3T const>  p_2 (parameters[2*frameNo+1]);

//            SO3 R_1(SO3::exp(Eigen::Map<Vec3T const>(parameters[frameNo-1])));
//            SO3 R_2(SO3::exp(Eigen::Map<Vec3T const>(parameters[frameNo])));
//            Eigen::Map<Vec3T const>  p_1 (parameters[winSize + frameNo-1]);
//            Eigen::Map<Vec3T const>  p_2 (parameters[winSize + frameNo]);
            ratios[i] = measurement_[i][3];

//            QuaT R = R_1.slerp(double(ratios[i]), R_2);
//            QuaternionCostFunctor::evaluateJacobAnalytically(R_1, R_2, R, ratios[i], jacob_R);
//            R.normalize();

            QuaT R_eva;
            SO3 R = R_1*SO3::exp(double(ratios[i]) * SO3::log(R_1.inverse()*R_2));
            QuaternionCostFunctor::evaluateJacobAnalytically(R_1.unit_quaternion(), R_2.unit_quaternion(), R_eva, ratios[i], jacob_R);
            cout << "[Debug] R_eva: " << R_eva.coeffs() << ", R: " << R.unit_quaternion().coeffs() << endl;

            Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_0(jacob_R[0]);
            Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_1(jacob_R[1]);

            plvec_back[i] = R * measurement_[i].topRows(3);
            point_xis[i] = -SO3::hat(plvec_back[i]);  // eq. (10)
            point_xis_bef[i] = point_xis[i] * jacobian_0.block<3,3>(0,0);
            point_xis[i]= (point_xis[i]*jacobian_1.block<3,3>(0,0)).eval();

            plvec_back[i](0) += 1-measurement_[i][3] * p_1(0) + measurement_[i][3] * p_2(0);
            plvec_back[i](1) += 1-measurement_[i][3] * p_1(1) + measurement_[i][3] * p_2(1);
            plvec_back[i](2) += 1-measurement_[i][3] * p_1(2) + measurement_[i][3] * p_2(2);

            centor += plvec_back[i];
            covMat += plvec_back[i] * plvec_back[i].transpose();
        }

        double N_points = double(backnum) + double(info_fix_.sigma_size);
        centor += info_fix_.sigma_vi;
        covMat += info_fix_.sigma_vTv;

        covMat = covMat - centor*centor.transpose()/N_points;
        covMat = covMat / N_points;
        centor = centor / N_points;

        Eigen::SelfAdjointEigenSolver<Mat3T > saes;
        saes.compute(covMat);
        Vec3T eigen_value = saes.eigenvalues();

        Eigen::Matrix3d U = saes.eigenvectors();
        Eigen::Vector3d u[3]; // eigenvectors
        for(int j=0; j<3; j++)
            u[j] = U.block<3, 1>(0, j);

        Eigen::Map<Eigen::Matrix<double, 1, 1>> res(residuals);
        res.block<1, 1>(0, 0) = weight_ * Eigen::Matrix<double, 1, 1>(double(eigen_value(feaType_)));

        if (jacobians!= nullptr){
            // initialize
            for (int i = 0; i < winSize; ++i) {
                if (jacobians[i] != nullptr){
                    Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor> > J_1(jacobians[i]);
                    J_1.setZero();
                }
            }
            for (int i = 0; i < winSize; ++i) {
                if (jacobians[winSize + i] != nullptr){
                    Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor> > J_1(jacobians[winSize + i]);
                    J_1.setZero();
                }
            }
            // Jacobian matrix
            Eigen::Matrix3d ukukT = u[feaType_] * u[feaType_].transpose();
            Eigen::Vector3d vec_Jt;
            for(uint i=0; i<backnum; i++){
                if (slwd_Nos_[i] == 0 ) continue;

                plvec_back[i] = plvec_back[i] - centor;
                vec_Jt = 2.0/N_points * ukukT * plvec_back[i];  // eq. (6)
                // T1
                if (jacobians[slwd_Nos_[i]-1] != nullptr) {
                    Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor> > J_1(jacobians[slwd_Nos_[i] - 1]);
                    J_1.block<1, 3>(0, 0) += vec_Jt.transpose() * point_xis_bef[i];  // rot
                }
                if (jacobians[winSize + slwd_Nos_[i]-1] != nullptr) {
                    Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor> > J_2(jacobians[winSize + slwd_Nos_[i]-1]);
                    J_2.block<1, 3>(0, 0) += (1-ratios[i]) * vec_Jt.transpose();  // translation
                }
                // T2
                if (jacobians[slwd_Nos_[i]] != nullptr) {
                    Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor> > J_1(jacobians[slwd_Nos_[i]]);
                    J_1.block<1, 3>(0, 0) += vec_Jt.transpose() * point_xis[i];  // rot
                }
                if (jacobians[winSize + slwd_Nos_[i]] != nullptr) {
                    Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor> > J_2(jacobians[winSize + slwd_Nos_[i]]);
                    J_2.block<1, 3>(0, 0) += ratios[i] * vec_Jt.transpose();  // translation

//                    if (isinf(J_1(0,0))){  // FixMe: nan in Jacob!?
//                        cout << "[ WARN ] Nan point " << i << " in frame " << slwd_Nos_[i] << " with ratio " << ratios[i]
//                             << "rot\n" << J_1 << endl;
//                    }
//                    if (isinf(J_2(0,0))){  // FixMe: nan in Jacob!?
//                        cout << "[ WARN ] Nan point " << i << " in frame " << slwd_Nos_[i] << " with ratio " << ratios[i]
//                             << "trans\n" << J_2 << endl;
//                    }
                }


//                // T1
//                if (jacobians[6*slwd_Nos_[i]-6] != NULL){
//                    Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J_1(jacobians[6*slwd_Nos_[i]-6]);
//                    J_1.block<1, 3>(4, 0) += (1-ratios[i]) * vec_Jt.transpose();  // translation
//                    J_1.block<1, 3>(0, 0) += vec_Jt.transpose() * point_xis_bef[i];  // rot
//                }
//                // T2
//                if (jacobians[6*slwd_Nos_[i]] != NULL){
//                    Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J_2(jacobians[6*slwd_Nos_[i]]);
//                    J_2.block<1, 3>(4, 0) += ratios[i] * vec_Jt.transpose();
//                    J_2.block<1, 3>(0, 0) += vec_Jt.transpose() * point_xis[i];
//                }
//                if (isnan(_jact.block<6, 1>(6*slwd_num[i]-3, 0)(0,0)) || isnan(_jact.block<6, 1>(6*slwd_num[i]-3, 0)(0,3)))
//                cout << "_jact.block<"<< 6*slwd_num[i]-3 << "," << 0 << ">(6 X 1) \n"
//                     << _jact.block<6, 1>(6*slwd_num[i]-3, 0).transpose()  << endl;
            }
        }

        return true;
    }

private:
    vector<Eigen::Vector4d> measurement_;
    vector<int> slwd_Nos_;
    SIG_VEC_CLASS info_fix_;
    int feaType_, backnum, winSize;
    double weight_;
};


// ref: https://arxiv.org/pdf/2108.02976.pdf
struct ParamSO3 : public ceres::LocalParameterization {
    virtual bool Plus(const double *x, const double *delta,
                      double *x_plus_delta) const {
        Eigen::Map<const Vector3d> tangent(x);
        Eigen::Map<const Vector3d> drot(delta);
        Eigen::Map<Vector3d> out(x_plus_delta);

        out = (SO3::exp(tangent) * SO3::exp(drot)).log();

        return true;
    }

    virtual bool ComputeJacobian(const double *x, double *jacobian) const {
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> j(jacobian);
        j.setIdentity();
        return true;
    }

    virtual int GlobalSize() const { return 3; };
    virtual int LocalSize() const { return 3; };
};

class SurfRegXXEvalCallBack : public ceres::EvaluationCallback {
public:
    SurfRegXXEvalCallBack() = default;

    virtual ~SurfRegXXEvalCallBack() = default;

    virtual void PrepareForEvaluation(bool evaluate_jacobians,
                                      bool new_evaluation_point) override{
        if (new_evaluation_point){
//            cout << "cells: " << cells_.size() << endl;
            for(auto cell : cells_){
                cell->updateMeanCov(param_rot_, param_trans_);
                cell->decompose();
            }
        }
    }

    void addPlane(Gaussian3D *plane){
        cells_.emplace_back(plane);
    }

    void addFrame(double *rot, double *trans){
        param_rot_.emplace_back(rot);
        param_trans_.emplace_back(trans);
    }

    std::vector<Gaussian3D* > cells_;
    std::vector<double *> param_rot_, param_trans_;
};

class EigenFactorRotAxis : public ceres::SizedCostFunction<1, 3> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    EigenFactorRotAxis(const Gaussian3D *info_map, const Vector3d &axis,
                       const double s): scalar_(s), info_map_(info_map), axis_(axis) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals,
                          double **jacobians) const override{

        Eigen::Map<const Vector3d> rot_log(parameters[0]);
        const auto &rot = SO3::exp(rot_log);

        const RowVector3d &nhat = info_map_->lamda_vecs.col(0).transpose();

        if (residuals) {
            residuals[0] = nhat.dot(rot * axis_) * scalar_;
            if (isnan(residuals[0])) {
                cout << "[WARN] fuck EigenFactorRotAxis" << endl;
                cout << axis_.transpose() << endl;
                cout << scalar_ << endl;
                cout << nhat << endl;
                exit(0);
            }
        }

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<RowVector3d> jacob(jacobians[0]);
                jacob = -scalar_ * (nhat * rot.matrix() * QuaternionCostFunctor::skew(axis_));
            }
        }

        return true;
    }

public:
    const double scalar_ = 1.0;
    const Gaussian3D *info_map_ = nullptr;
    const Vector3d axis_;
};

class EigenFactorTrans2 : public ceres::SizedCostFunction<1, 3, 3> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    EigenFactorTrans2(const Gaussian3D *info_, const Vector3d &mu,
                      const double s):  scalar_(s), info_map_(info_), mu_(mu) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals,
                          double **jacobians) const override{

        Eigen::Map<const Vector3d> rot_log(parameters[0]);
        Eigen::Map<const Vector3d> t(parameters[1]);
        const auto &rot = SO3::exp(rot_log);

        const RowVector3d &nhat = info_map_->lamda_vecs.col(0).transpose();
        const Vector3d &mu_tgt = info_map_->mean;

        if (residuals) {
            residuals[0] = scalar_ * nhat.dot((rot * mu_ + t) - mu_tgt);
            if (isnan(residuals[0])) {
                cout << "[WARN] fuck EigenFactorTrans2" << endl;
                exit(0);
            }
        }

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<RowVector3d> jacob(jacobians[0]);
                jacob = -nhat * rot.matrix() * QuaternionCostFunctor::skew(mu_);
                jacob *= scalar_;
            }

            if (jacobians[1]) {
                Eigen::Map<RowVector3d> jacob(jacobians[1]);
                jacob = nhat * scalar_;
            }
        }

        return true;
    }

public:
    const double scalar_ = 1.0;

    const Gaussian3D *info_map_ = nullptr;
    const Vector3d mu_;
};

#endif //BALM_CERESFACTORS_HPP
