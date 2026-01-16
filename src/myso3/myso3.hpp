#ifndef MYSO3
#define MYSO3

// Extracted from "https://github.com/strasdat/Sophus"

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Dense>

using namespace Eigen;

const double SMALL_EPS = 1e-10;

class SO3
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SO3                        ();

    SO3                        (const SO3 & other);

    explicit
    SO3                        (const Matrix3d & _R);

    explicit
    SO3                        (const Quaterniond & unit_quaternion);

    SO3                        (double rot_x,
                                double rot_y,
                                double rot_z);
    void
    operator=                  (const SO3 & so3);

    SO3
    operator*                  (const SO3 & so3) const;

    void
    operator*=                 (const SO3 & so3);

    Vector3d
    operator*                  (const Vector3d & xyz) const;

    SO3
    inverse                    () const;

    Matrix3d
    matrix                     () const;

    Matrix3d
    Adj                        () const;

    Matrix3d
    generator                  (int i);

    Vector3d
    log                        () const;

    static SO3
    exp                        (const Vector3d & omega);

    static SO3
    expAndTheta                (const Vector3d & omega,
                                double * theta);
    static Vector3d
    log                        (const SO3 & so3);

    static Vector3d
    logAndTheta                (const SO3 & so3,
                                double * theta);

    static Matrix3d
    hat                        (const Vector3d & omega);

    static Vector3d
    vee                        (const Matrix3d & Omega);

    static Vector3d
    lieBracket                 (const Vector3d & omega1,
                                const Vector3d & omega2);

    static Matrix3d
    d_lieBracketab_by_d_a      (const Vector3d & b);

    void
    setQuaternion              (const Quaterniond& quaternion);

    Quaterniond& unit_quaternion(){
        return unit_quaternion_;
    }

    static const int DoF = 3;

protected:
    Quaterniond unit_quaternion_;
};

inline std::ostream& operator <<(std::ostream & out_str,
                                 const SO3 & so3)
{

    out_str << so3.log().transpose() << std::endl;
    return out_str;
}

// P_fix in the paper
// Summation of P_fix
class SIG_VEC_CLASS
{
public:
    Eigen::Matrix3d sigma_vTv;
    Eigen::Vector3d sigma_vi;
    int sigma_size;

    SIG_VEC_CLASS()
    {
        sigma_vTv.setZero();
        sigma_vi.setZero();
        sigma_size = 0;
    }

    void tozero()
    {
        sigma_vTv.setZero();
        sigma_vi.setZero();
        sigma_size = 0;
    }
};

using namespace std;
class Gaussian3D{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
    int winSize, ptSize = 0;
    vector<SIG_VEC_CLASS> sig_vecs_local;  // local mean and cov per frame
    bool inited = false;

    Eigen::Vector3d lamdas;  // eigenvalues for this voxel
    Eigen::Matrix3d lamda_vecs;  // eigenvectors for this voxel
    // in map frame
    Eigen::Vector3d mean;
    Eigen::Matrix3d cov;
    SIG_VEC_CLASS margin;  // marginalized mean and cov

    Gaussian3D(int N): winSize(N) {
        sig_vecs_local.resize(N);
        inited = false;
    }

    void initialize(const vector<double* > &rots, const vector<double* > &trans){

        if (!inited){
            updateMeanCov(rots, trans);
            decompose();
            inited = true;
        }
    }

    void decompose(){

        auto eigen_solver = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(cov);

        Matrix3d axis_ = eigen_solver.eigenvectors();

        axis_.col(2) = axis_.col(0).cross(axis_.col(1));
        SO3 rot = SO3(axis_);
        lamda_vecs = rot.matrix();
        lamdas = eigen_solver.eigenvalues();
    }

    void updateMeanCov(const vector<double* > &rots, const vector<double* > &trans){

        assert(rots.size()  == sig_vecs_local.size());
        assert(trans.size() == sig_vecs_local.size());

        mean = margin.sigma_vi;
        cov = margin.sigma_vTv;
        vector<SIG_VEC_CLASS> sig_vecs_global(winSize);
        for (int i = 0; i < winSize; ++i) {
            if (sig_vecs_local[i].sigma_size == 0) continue;

            SO3 rot = SO3::exp(Eigen::Map<Eigen::Vector3d>(rots[i]));
            sig_vecs_global[i].sigma_vi = rot * sig_vecs_local[i].sigma_vi +
                                          Eigen::Map<Eigen::Vector3d>(trans[i]);
            sig_vecs_global[i].sigma_vTv = rot.matrix() * sig_vecs_local[i].sigma_vTv * rot.matrix().transpose();

            mean += sig_vecs_local[i].sigma_size * sig_vecs_global[i].sigma_vi;
            cov += sig_vecs_local[i].sigma_size * sig_vecs_global[i].sigma_vTv;
        }
        mean /= (ptSize+margin.sigma_size);
        for (int i = 0; i < winSize; ++i) {
            if (sig_vecs_local[i].sigma_size == 0) continue;
            cov += sig_vecs_local[i].sigma_size * (sig_vecs_global[i].sigma_vi - mean) *
                   (sig_vecs_global[i].sigma_vi - mean).transpose();
        }
        cov /= (ptSize+margin.sigma_size);
    }
};
#endif
