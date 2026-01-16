//
// Created by cyz on 2022/7/31.
//
// inspired from ROLL and LIO-SAM
//

#ifndef BALM_GRAPH_OPT_HPP
#define BALM_GRAPH_OPT_HPP

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

#pragma clang diagnostic push
#pragma ide diagnostic ignored "EndlessLoop"
using namespace gtsam;

class GraphOptimization{

public:
    GraphOptimization(int maxNo, int margN);
    ~GraphOptimization();

    void setTgl(Eigen::Matrix4d mat);
    void getTgl(Eigen::Matrix4d &tgl);
    void affine2qt(const Eigen::Matrix4d aff, Eigen::Vector3d &locP, Eigen::Quaterniond &locQ);
    // void inputGPS(double t, double latitude, double longitude, double altitude, double posAccuracy);
    void feedIMU_msg(sensor_msgs::Imu imuMsg);

    void inputOdom(double t, Eigen::Quaterniond OdomQ, Eigen::Vector3d OdomP, double q_err, double t_err);
    void inputOdom(double t, Eigen::Quaterniond OdomQ, Eigen::Vector3d OdomP,
                   double q_err, double t_err,
                   Eigen::Quaterniond globalQ, Eigen::Vector3d globalP,
                   Eigen::Vector3d vel = Eigen::Vector3d::Zero(),
                   Eigen::Vector3d ba = Eigen::Vector3d::Zero(),
                   Eigen::Vector3d bg = Eigen::Vector3d::Zero());
    void inputGlobalLocPose(double t, Eigen::Quaterniond locQ, Eigen::Vector3d locP,
                            double t_error, double q_error);
    void getGlobalOdom(Eigen::Vector3d &odomP, Eigen::Quaterniond &odomQ);
    void getGlobalAffine(Eigen::Affine3f &Tml);

    void resetOptimization(Eigen::Matrix4d Tgl);
    void optimize();
    void optimizeWithIMU();
    bool updatePosesGlobal(vector<Eigen::Quaterniond> &q_poses, vector<Eigen::Vector3d> &t_poses, int win_base);

    bool isInitialized;
    bool reInitialize;

private:
    // void GPS2XYZ(double latitude, double longitude, double altitude, double* xyz);
    void updateGlobalPath();

    // format t, tx,ty,tz,qw,qx,qy,qz
    map<double, vector<double> > localPoseMap;
    map<double, vector<double> > globalPoseMap; // only synchronized gps is used
    map<double, vector<double> > GPSPositionMap;
    map<double, vector<double> > globalLocPoseMap;
    bool initGPS;
    bool newGPS;
    bool newGlobalLocPose;
    // GeographicLib::LocalCartesian geoConverter;
    std::mutex mPoseMap;
    Eigen::Matrix4d WGlobal_T_WLocal;
    Eigen::Vector3d lastP;
    Eigen::Quaterniond lastQ;
    int maxFrameNum, margFrameNum;

//    std::thread threadOpt;

    // for acc jump identification
    vector<Eigen::Matrix4d> poseVec;
    Eigen::Matrix4d Tacc;
    bool resetGlobalPoses;
    Eigen::Matrix4d backupTgl;
    // double stdx,meanx,stdy,meany,stdz,meanz;

    std::deque<sensor_msgs::Imu> imuQueOpt;
    std::deque<sensor_msgs::Imu> imuQueImu;
    double lastImuT_opt = -1;

    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;
    gtsam::Vector noiseModelBetweenBias;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
};

// keep quaternion input to adapt ceres implementation
GraphOptimization::GraphOptimization(int maxNo, int margN): maxFrameNum(maxNo), margFrameNum(margN){

    initGPS = false;
    newGPS = false;
    newGlobalLocPose = false;
    WGlobal_T_WLocal = Eigen::Matrix4d::Identity();
//    threadOpt = std::thread(&GraphOptimization::optimize, this);

    reInitialize = false;
    Tacc = Eigen::Matrix4d::Identity();
    backupTgl = Eigen::Matrix4d::Identity();

    // IMU
    priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); // 1e4 m/s
    priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-2); // 1e-2 ~ 1e-3 seems to be good

    double imuAccBiasN = 0.000043, imuGyrBiasN = 0.000001;
    boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(9.80511);
    p->accelerometerCovariance = gtsam::Matrix33::Identity(3,3) * pow(0.08,2); // acc white noise in continuous
    p->gyroscopeCovariance     = gtsam::Matrix33::Identity(3,3) * pow(0.02,2); // gyro white noise in continuous
    p->integrationCovariance   = gtsam::Matrix33::Identity(3,3) * 1e-8 ; // error committed in integrating position from velocities
    gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias
    noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();

    imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
    imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization
}

void GraphOptimization::setTgl(Eigen::Matrix4d mat){

    WGlobal_T_WLocal = mat;
    backupTgl = mat;
}

GraphOptimization::~GraphOptimization()
{
//    threadOpt.detach();
}

void GraphOptimization::affine2qt(const Eigen::Matrix4d aff, Eigen::Vector3d &p, Eigen::Quaterniond &q)
{
    q = Eigen::Quaterniond(aff.block<3,3>(0,0));
    p = aff.block<3,1>(0,3);
    // cout<<"q: "<<q.w()<<" "<<q.x()<<" "<<q.y()<<" "<<q.z()<<endl;
    return;
}

// here we update globally aligned odom pose asap
void GraphOptimization::inputOdom(double t, Eigen::Quaterniond OdomQ, Eigen::Vector3d OdomP,
                                  double q_err, double t_err){

    mPoseMap.lock();
    vector<double> localPose{OdomP.x(), OdomP.y(), OdomP.z(),
                             OdomQ.w(), OdomQ.x(), OdomQ.y(), OdomQ.z(), q_err, t_err};
    localPoseMap[t] = localPose;
//    globalPoseMap[t] = localPose;

    // cout << setiosflags(ios::fixed) << setprecision(6) << "lio: "<<t<<endl;

    Eigen::Quaterniond globalQ;
    globalQ = WGlobal_T_WLocal.block<3, 3>(0, 0) * OdomQ;
    Eigen::Vector3d globalP = WGlobal_T_WLocal.block<3, 3>(0, 0) * OdomP + WGlobal_T_WLocal.block<3, 1>(0, 3);
    vector<double> globalPose{globalP.x(), globalP.y(), globalP.z(),
                              globalQ.w(), globalQ.x(), globalQ.y(), globalQ.z()};
    globalPoseMap[t] = globalPose;
    lastP = globalP;
    lastQ = globalQ;

    // sliding-window
    if ((int)localPoseMap.size() > maxFrameNum){

        // special note: erase function of map is not the same as vector
        localPoseMap.erase((localPoseMap.begin())->first);
        globalPoseMap.erase((globalPoseMap.begin())->first);
    }

    mPoseMap.unlock();
}

void GraphOptimization::inputOdom(double t, Eigen::Quaterniond OdomQ, Eigen::Vector3d OdomP,
                                  double q_err, double t_err,
                                  Eigen::Quaterniond globalQ, Eigen::Vector3d globalP,
                                  Eigen::Vector3d vel, Eigen::Vector3d ba, Eigen::Vector3d bg){

    mPoseMap.lock();
    vector<double> localPose{OdomP.x(), OdomP.y(), OdomP.z(),
                             OdomQ.w(), OdomQ.x(), OdomQ.y(), OdomQ.z(), q_err, t_err,
                             vel.x(), vel.y(), vel.z(),
                             ba.x(),  ba.y(), ba.z(),
                             bg.x(),  bg.y(), bg.z() };
    localPoseMap[t] = localPose;
//    globalPoseMap[t] = localPose;

    // cout << setiosflags(ios::fixed) << setprecision(6) << "lio: "<<t<<endl;

    vector<double> globalPose{globalP.x(), globalP.y(), globalP.z(),
                              globalQ.w(), globalQ.x(), globalQ.y(), globalQ.z()};
    globalPoseMap[t] = globalPose;
    lastP = globalP;
    lastQ = globalQ;

    // sliding-window
    if ((int)localPoseMap.size() > maxFrameNum){

        // special note: erase function of map is not the same as vector
        localPoseMap.erase((localPoseMap.begin())->first);
        globalPoseMap.erase((globalPoseMap.begin())->first);
    }

    mPoseMap.unlock();
}

void GraphOptimization::feedIMU_msg(sensor_msgs::Imu imuMsg){

    imuQueOpt.emplace_back(imuMsg);
    imuQueImu.emplace_back(imuMsg);
}

void GraphOptimization::getGlobalOdom(Eigen::Vector3d &odomP, Eigen::Quaterniond &odomQ)
{
    odomP = lastP;
    odomQ = lastQ;
}

void GraphOptimization::getGlobalAffine(Eigen::Affine3f &Tml)
{
    // need to wait for the 1st opt. completion
    Eigen::Matrix4d tmp;
    tmp.block<3,3>(0,0) = lastQ.toRotationMatrix();
    tmp.block<3,1>(0,3) = lastP;
    Tml = tmp.cast<float>();
}

void GraphOptimization::getTgl(Eigen::Matrix4d &tgl)
{
    tgl = WGlobal_T_WLocal;
}


void GraphOptimization::inputGlobalLocPose(double t, Eigen::Quaterniond locQ, Eigen::Vector3d locP,
                                           double t_error, double q_error){

    mPoseMap.lock();
    vector<double> globalLocPose{locP.x(), locP.y(), locP.z(),
                                 locQ.w(), locQ.x(), locQ.y(), locQ.z(), t_error, q_error};
    globalLocPoseMap[t] = globalLocPose;
    // vector<double> globalLocPose{locP.x(), locP.y(), locP.z(),t_error};
    // globalLocPoseMap[t] = globalLocPose;

    // cout<<setiosflags(ios::fixed)<<setprecision(6)<<"global matching: "<<t<<endl;

    if (reInitialize && (int)globalLocPoseMap.size() < 10){
        mPoseMap.unlock();
        return;
    }
    newGlobalLocPose = true;
    mPoseMap.unlock();
}

void GraphOptimization::resetOptimization(Eigen::Matrix4d Tgl)
{
    // cout<<"too much jump in Tgl change, forfeit estimate:  Tgl change "<<deltaTransGL<<endl;
    WGlobal_T_WLocal = Tgl;

    globalLocPoseMap.clear();
    globalPoseMap.clear();
    localPoseMap.clear();
    reInitialize = true;
}

gtsam::Pose3 QT2gtsamPose(vector<double> qt)
{
    Eigen::Quaterniond q(qt[3],qt[4],qt[5],qt[6]);
    Eigen::Matrix3d qtMatrix = q.matrix();
    // here it returns (thetaZ,thetaY,thetaX)
    Eigen::Vector3d ypr = qtMatrix.eulerAngles(2,1,0);
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(ypr[2], ypr[1], ypr[0]),
                        gtsam::Point3(qt[0], qt[1], qt[2]));
}

void gtsamPose2Matrix4d(gtsam::Pose3 pose, Eigen::Matrix4d & mat)
{
    Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(pose.rotation().roll(),Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(pose.rotation().pitch(),Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(pose.rotation().yaw(),Eigen::Vector3d::UnitZ()));
    Eigen::Matrix3d rotation_matrix;
    rotation_matrix = yawAngle*pitchAngle*rollAngle;
    mat.block<3,3>(0,0) = rotation_matrix;
    mat.block<3,1>(0,3) = Eigen::Vector3d(pose.translation().x(),pose.translation().y(),pose.translation().z());

}
void gtsamPose2Vector(gtsam::Pose3 pose, vector<double> &qt)
{
    Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(pose.rotation().roll(),Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(pose.rotation().pitch(),Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(pose.rotation().yaw(),Eigen::Vector3d::UnitZ()));
    Eigen::Matrix3d rotation_matrix = (yawAngle*pitchAngle*rollAngle).matrix();
    Eigen::Quaterniond qd =  Eigen::Quaterniond(rotation_matrix);
    qt[0] = pose.translation().x();
    qt[1] = pose.translation().y();
    qt[2] = pose.translation().z();
    qt[3] = qd.w(); // are you sure?
    qt[4] = qd.x();
    qt[5] = qd.y();
    qt[6] = qd.z();
}

void GraphOptimization::optimize(){

    if(newGlobalLocPose){

        if (reInitialize) reInitialize = false;

//            TicToc opt_time;
        newGlobalLocPose = false;
        // printf("global optimization using global localization and odometry\n");
//            TicToc globalOptimizationTime;

        // gtsam
        NonlinearFactorGraph gtSAMgraphTM;
        Values initialEstimateTM;
        ISAM2 *isamTM;
        Values isamCurrentEstimateTM;
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isamTM = new ISAM2(parameters);

        //add param
        mPoseMap.lock();

        int length = localPoseMap.size();
        // cout<<" pose no. before opt "<< length<<endl;
        map<double, vector<double> >::iterator iterIni,iterLIO, iterLIOnext, iterGlobalLoc;
        iterIni = globalPoseMap.begin();  //using odomTOmap value for initial guess
        int i = 0;
        int found = 0;
        // int outlierNO = 0;
        for (iterLIO = localPoseMap.begin(); iterLIO != localPoseMap.end(); iterLIO++, i++,iterIni++){

            gtsam::Pose3 poseFrom = QT2gtsamPose(iterLIO->second);
            // vio factor
            // cout<<"i lio pose: " <<i<<endl;
            iterLIOnext = iterLIO;
            iterLIOnext++;
            if(iterLIOnext != localPoseMap.end()){

                /// odometry factor
                noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances(
                        (Vector(6) << iterLIOnext->second[7], iterLIOnext->second[7], iterLIOnext->second[7],
                                iterLIOnext->second[8], iterLIOnext->second[8], iterLIOnext->second[8]).finished());
                gtsam::Pose3 poseTo   = QT2gtsamPose(iterLIOnext->second);
                gtSAMgraphTM.add(BetweenFactor<Pose3>(X(i),X(i+1), poseFrom.between(poseTo),
                                                      odometryNoise));
            }

            double t = iterLIO->first;
            iterGlobalLoc = globalLocPoseMap.find(t); // synchronized global loc pose
            if (iterGlobalLoc != globalLocPoseMap.end()){
                /// BA prior factor
                gtsam::Pose3 poseGlobal = QT2gtsamPose(iterGlobalLoc->second);
                // seems to be overconfident
                // noiseModel::Diagonal::shared_ptr corrNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1).finished()); // rad*rad, meter*meter
                double tE = iterGlobalLoc->second[7] / 2;
                double tQ = iterGlobalLoc->second[8] / 2;
                noiseModel::Diagonal::shared_ptr corrNoise = noiseModel::Diagonal::Variances(
                        (Vector(6) << tQ,tQ,tQ,tE,tE,tE).finished()); // rad*rad, meter*meter
                gtSAMgraphTM.add(PriorFactor<Pose3>(X(i), poseGlobal, corrNoise));
                found++;

                // for CC
                Eigen::Matrix4d local = Eigen::Matrix4d::Identity();
                Eigen::Matrix4d global = Eigen::Matrix4d::Identity();
                global.block<3, 3>(0, 0) = Eigen::Quaterniond(iterGlobalLoc->second[3], iterGlobalLoc->second[4],
                                                              iterGlobalLoc->second[5], iterGlobalLoc->second[6]).toRotationMatrix();
                global.block<3, 1>(0, 3) = Eigen::Vector3d(iterGlobalLoc->second[0], iterGlobalLoc->second[1], iterGlobalLoc->second[2]);
                local.block<3, 3>(0, 0) = Eigen::Quaterniond(iterLIO->second[3], iterLIO->second[4],
                                                             iterLIO->second[5], iterLIO->second[6]).toRotationMatrix();
                local.block<3, 1>(0, 3) = Eigen::Vector3d(iterLIO->second[0], iterLIO->second[1], iterLIO->second[2]);
                backupTgl = global * local.inverse(); // get the newest Tgl as backup
            }

            gtsam::Pose3 poseGuess = QT2gtsamPose(iterIni->second);
            initialEstimateTM.insert(X(i), poseGuess);
        }
        cout <<"[ Graph ] found: " << found;

        if (found == 0){
            mPoseMap.unlock();
            return;
        }

        isamTM->update(gtSAMgraphTM, initialEstimateTM);
        isamTM->update();
        isamTM->update();
        gtSAMgraphTM.resize(0);
        initialEstimateTM.clear();

        isamCurrentEstimateTM = isamTM->calculateEstimate();
//        gtsam::LevenbergMarquardtParams lm_params;
//        lm_params.setLinearSolverType("MULTIFRONTAL_QR");
//        lm_params.setMaxIterations(30);
//        gtsam::LevenbergMarquardtOptimizer lm_optimizer(gtSAMgraphTM, initialEstimateTM, lm_params);
//        isamCurrentEstimateTM = lm_optimizer.optimize();
//        isamCurrentEstimateTM.print("current estimate:\n");

        // cout<<"Estimate size: "<<length<<endl;
        // update global pose
        iterIni = globalPoseMap.begin();

        Eigen::Matrix4d start;
        Eigen::Matrix4d end;
        Eigen::Matrix4d WVIO_T_body = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d WGPS_T_body = Eigen::Matrix4d::Identity();

        for (int i = 0; i < length; i++, iterIni++){
            // cout<<"i "<<i<<"length :"<<length<<endl;
            vector<double> globalPose(7,0);
            gtsamPose2Vector(isamCurrentEstimateTM.at<Pose3>(X(i)),globalPose);
            iterIni->second = globalPose;

            double t = iterIni->first;
            WVIO_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(localPoseMap[t][3], localPoseMap[t][4],
                                                               localPoseMap[t][5], localPoseMap[t][6]).toRotationMatrix();
            WVIO_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(localPoseMap[t][0], localPoseMap[t][1], localPoseMap[t][2]);
            WGPS_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(globalPose[3], globalPose[4],
                                                               globalPose[5], globalPose[6]).toRotationMatrix();
            WGPS_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(globalPose[0], globalPose[1], globalPose[2]);

            WGlobal_T_WLocal = WGPS_T_body * WVIO_T_body.inverse();

            if (i == 0 ) start = WGlobal_T_WLocal;
            if (i == length - 1 ) end = WGlobal_T_WLocal;
        }

        // update noise model
//        priorVelNoise  = gtsam::noiseModel::Gaussian::Covariance(isamTM->marginalCovariance(V(length-1)));
//        priorBiasNoise = gtsam::noiseModel::Gaussian::Covariance(isamTM->marginalCovariance(B(length-1)));

        // cout<<"Tgl change "<<deltaTransGL<<endl;
        // w. consistency check: GTSAM implementation needs CC as well
        // OR the process will just crash without warning (the opt. time takes more than 500 ms)
        // Tgl change too much, forfeit this optimization
        Eigen::Matrix4d TglDelta = start.inverse() * end;
        // cout<<"x y z: "<<TglDelta(0,3)<<" "<<TglDelta(1,3)<<" "<<TglDelta(2,3)<<endl;
        double deltaTransGL = sqrt(TglDelta(0,3)*TglDelta(0,3) +
                                   TglDelta(1,3)*TglDelta(1,3) +
                                   TglDelta(2,3)*TglDelta(2,3) );
        if ( deltaTransGL > 1){
            cout<<"[ WARN ] Reset when deltaTgl = " << deltaTransGL << endl;
//                resetOptimization(backupTgl);
            setTgl(Eigen::Matrix4d::Identity());
            isInitialized = false;
        }
//            else setTgl(end);

//            double opt_t = opt_time.toc();
//             cout<<"[ Graph ] optimization takes: "<<opt_t<<" ms"<<endl; // gtsam implementation usually takes less than 10 ms
        cout << "... optimization done. " << endl; // gtsam implementation usually takes less than 10 ms

//            if(opt_t> 100) cout<<"gtsam opt. takes more than 100 ms"<<endl;

        mPoseMap.unlock();
    }
    // waiting for poses to be accumulated
//        std::chrono::milliseconds dura(500);
//        std::this_thread::sleep_for(dura);

}

/* add imu factors between prior factors */
void GraphOptimization::optimizeWithIMU(){

    if(newGlobalLocPose){

        if (reInitialize) reInitialize = false;

//            TicToc opt_time;
        newGlobalLocPose = false;
        // printf("global optimization using global localization and odometry\n");
//            TicToc globalOptimizationTime;

        gtsam::NavState prevState_, propState_;
        gtsam::imuBias::ConstantBias prevBias_;
        gtsam::Vector3 preVel_;

        // gtsam
        NonlinearFactorGraph gtSAMgraphTM;
        Values initialEstimateTM;
        ISAM2 *isamTM;
        Values isamCurrentEstimateTM;
        ISAM2Params parameters;
        parameters.factorization = ISAM2Params::QR;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isamTM = new ISAM2(parameters);

        //add param
        mPoseMap.lock();
        bool optimized = false, founded = false, addimu = false;
        const double preIntegr_thre = 0.25, hz = 600.0;  // TODO: big integration difference between poses would crash?
        int length = localPoseMap.size();
        // cout<<" pose no. before opt "<< length<<endl;
        map<double, vector<double> >::iterator iterIni, iterLIO, iterLIOnext, iterGlobalLoc;
        iterIni = globalPoseMap.begin();  //using odomTOmap value for initial guess
        int i = 0;
        int found = 0;
        // int outlierNO = 0;
        for (iterLIO = localPoseMap.begin(); iterLIO != localPoseMap.end(); iterLIO++, i++, iterIni++){

            gtsam::Pose3 poseGuess = QT2gtsamPose(iterIni->second);
            initialEstimateTM.insert(X(i), poseGuess);

            founded = false;
            double t = iterLIO->first;
            iterGlobalLoc = globalLocPoseMap.find(t); // finding global loc pose
            if (iterGlobalLoc != globalLocPoseMap.end()){
                /// BA prior factor
                gtsam::Pose3 poseGlobal = QT2gtsamPose(iterGlobalLoc->second);
                // seems to be overconfident
                // noiseModel::Diagonal::shared_ptr corrNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1).finished()); // rad*rad, meter*meter
                double tE = iterGlobalLoc->second[7] / 2;
                double tQ = iterGlobalLoc->second[8] / 2;
                noiseModel::Diagonal::shared_ptr corrNoise = noiseModel::Diagonal::Variances(
                        (Vector(6) << tQ,tQ,tQ,tE,tE,tE).finished()); // rad*rad, meter*meter
                gtSAMgraphTM.add(PriorFactor<Pose3>(X(i), poseGlobal, corrNoise));
                found ++;
                founded = true;

                // for CC
                Eigen::Matrix4d local = Eigen::Matrix4d::Identity();
                Eigen::Matrix4d global = Eigen::Matrix4d::Identity();
                global.block<3, 3>(0, 0) = Eigen::Quaterniond(iterGlobalLoc->second[3], iterGlobalLoc->second[4],
                                                              iterGlobalLoc->second[5], iterGlobalLoc->second[6]).toRotationMatrix();
                global.block<3, 1>(0, 3) = Eigen::Vector3d(iterGlobalLoc->second[0], iterGlobalLoc->second[1], iterGlobalLoc->second[2]);
                local.block<3, 3>(0, 0) = Eigen::Quaterniond(iterLIO->second[3], iterLIO->second[4],
                                                             iterLIO->second[5], iterLIO->second[6]).toRotationMatrix();
                local.block<3, 1>(0, 3) = Eigen::Vector3d(iterLIO->second[0], iterLIO->second[1], iterLIO->second[2]);
                backupTgl = global * local.inverse(); // get the newest Tgl as backup

                if (addimu) {
                    iterLIOnext = iterLIO;
                    iterLIOnext--;
                    gtsam::Pose3 poseFrom = QT2gtsamPose(iterLIOnext->second);
                    prevBias_ = gtsam::imuBias::ConstantBias((gtsam::Vector(6) << 0,0,0,
                            iterLIOnext->second[15], iterLIOnext->second[16], iterLIOnext->second[17]).finished());
                    prevState_ = gtsam::NavState(poseFrom, (gtsam::Vector(3) << iterLIOnext->second[9],
                            iterLIOnext->second[10], iterLIOnext->second[11]).finished());

                    /// IMU preintegration
                    imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
                    int imuCnt=0;
                    while (!imuQueOpt.empty()){
                        // pop and integrate imu data that is between two optimizations
                        sensor_msgs::Imu *thisImu = &imuQueOpt.front();
                        double imuTime = thisImu->header.stamp.toSec();
                        if (imuTime < iterLIOnext->first)
                            imuQueOpt.pop_front();
                        else if (imuTime < iterLIO->first){
//                        cout << "imuTime: " << to_string(imuTime) << endl;
                            double dt = (lastImuT_opt < 0) ? (1.0 / hz) : (imuTime - lastImuT_opt);
                            if (dt != 0){
                                imuIntegratorOpt_->integrateMeasurement(
                                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                                imuCnt++;
                                lastImuT_opt = imuTime;
                            }
                            imuQueOpt.pop_front();
                        }else
                            break;
                    }
//                    cout << "[ Graph ] integrated IMU " << imuCnt << " between " << to_string(iterLIOnext->first) << " & " << to_string(iterLIO->first) << endl;
                    cout << "[ Graph ] integrated IMU " << imuCnt << " between " << i-1 << " & " << i << endl;

                    propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);

                    gtsam::Pose3 poseTo = QT2gtsamPose(iterLIO->second);
                    gtsam::Pose3 poseBetw  = poseFrom.between(poseTo);
                    gtsam::Pose3 poseBetw_imu = poseFrom.between(propState_.pose());
                    poseBetw = poseBetw.inverse() * poseBetw_imu;
                    Eigen::AngleAxisd angleAxisd(poseBetw.rotation().matrix());
                    cout << "[ Debug ] integration errors: " << angleAxisd.angle() << " / " << poseBetw.translation().norm() << endl;
                    if (poseBetw.translation().norm() < preIntegr_thre){
                        /// IMU factor
                        const auto &preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements &>(*imuIntegratorOpt_);
                        gtsam::ImuFactor imu_factor(X(i-1), V(i-1), X(i), V(i), B(i-1), preint_imu);
                        gtSAMgraphTM.add(imu_factor);
                        // add imu bias between factor
                        gtSAMgraphTM.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>
                                                 (B(i-1), B(i),
                                                  gtsam::imuBias::ConstantBias(),
                                                  gtsam::noiseModel::Diagonal::Sigmas(
                                                          sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));

//                    initialEstimateTM.insert(X(i), propState_.pose());
                        initialEstimateTM.insert(V(i), propState_.v());
                        initialEstimateTM.insert(B(i), prevBias_);
                        addimu = false;
                    }
                }
            }

            if (!founded && i > 0){
                iterLIOnext = iterLIO;
                iterLIOnext--;
                gtsam::Pose3 poseFrom = QT2gtsamPose(iterLIOnext->second);
//                prevBias_ = gtsam::imuBias::ConstantBias((gtsam::Vector(6) << iterLIOnext->second[12],
//                        iterLIOnext->second[13], iterLIOnext->second[14], iterLIOnext->second[15],
//                        iterLIOnext->second[16], iterLIOnext->second[17]).finished());
                prevBias_ = gtsam::imuBias::ConstantBias((gtsam::Vector(6) << 0, 0, 0,  // ba~0
                        iterLIOnext->second[15], iterLIOnext->second[16], iterLIOnext->second[17]).finished());
                prevState_ = gtsam::NavState(poseFrom, (gtsam::Vector(3) << iterLIOnext->second[9],
                        iterLIOnext->second[10], iterLIOnext->second[11]).finished());

                if (!addimu && !initialEstimateTM.exists(V(i-1))){
                    /// velocity prior
                    gtsam::PriorFactor<gtsam::Vector3> priorVel(V(i - 1), prevState_.v(), priorVelNoise);
                    gtSAMgraphTM.add(priorVel);
                    /// bias prior
                    gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(i - 1), prevBias_, priorBiasNoise);
                    gtSAMgraphTM.add(priorBias);
                    initialEstimateTM.insert(V(i - 1), prevState_.v());
                    initialEstimateTM.insert(B(i - 1), prevBias_);
                }

                /// IMU preintegration
                imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
                int imuCnt=0;
                while (!imuQueOpt.empty()){
                    // pop and integrate imu data that is between two optimizations
                    sensor_msgs::Imu *thisImu = &imuQueOpt.front();
                    double imuTime = thisImu->header.stamp.toSec();
                    if (imuTime < iterLIOnext->first)
                        imuQueOpt.pop_front();
                    else if (imuTime < iterLIO->first){
//                        cout << "imuTime: " << to_string(imuTime) << endl;
                        double dt = (lastImuT_opt < 0) ? (1.0 / hz) : (imuTime - lastImuT_opt);
                        if (dt != 0){
                            imuIntegratorOpt_->integrateMeasurement(
                                    gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                    gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                            imuCnt++;
                            lastImuT_opt = imuTime;
                        }
                        imuQueOpt.pop_front();
                    }else
                        break;
                }
                cout << "[ Graph ] integrated IMU " << imuCnt << " between " << i-1 << " & " << i << endl;
                propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);

                gtsam::Pose3 poseTo = QT2gtsamPose(iterLIO->second);
                gtsam::Pose3 poseBetw  = poseFrom.between(poseTo);
                gtsam::Pose3 poseBetw_imu = poseFrom.between(propState_.pose());
                poseBetw = poseBetw.inverse() * poseBetw_imu;
                Eigen::AngleAxisd angleAxisd(poseBetw.rotation().matrix());
                cout << "[ Debug ] integration errors: " << angleAxisd.angle() << " / " << poseBetw.translation().norm() << endl;
                if (poseBetw.translation().norm() < preIntegr_thre) {
                    /// IMU factor
                    const auto &preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements &>(*imuIntegratorOpt_);
                    gtsam::ImuFactor imu_factor(X(i - 1), V(i - 1), X(i), V(i), B(i - 1), preint_imu);
                    gtSAMgraphTM.add(imu_factor);
                    // add imu bias between factor
                    gtSAMgraphTM.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>
                                             (B(i - 1), B(i),
                                              gtsam::imuBias::ConstantBias(),
                                              gtsam::noiseModel::Diagonal::Sigmas(
                                                      sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));

//                initialEstimateTM.insert(X(i), propState_.pose());
                    initialEstimateTM.insert(V(i), propState_.v());
                    initialEstimateTM.insert(B(i), prevBias_);
                    addimu = true;
                }
            }

            iterLIOnext = iterLIO;
            iterLIOnext++;
            if(iterLIOnext != localPoseMap.end()){

                /// odometry factor
                noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances(
                        (Vector(6) << iterLIOnext->second[7], iterLIOnext->second[7], iterLIOnext->second[7],
                                iterLIOnext->second[8], iterLIOnext->second[8], iterLIOnext->second[8]).finished());
                gtsam::Pose3 poseFrom = QT2gtsamPose(iterLIO->second);
                gtsam::Pose3 poseTo = QT2gtsamPose(iterLIOnext->second);
                gtSAMgraphTM.add(BetweenFactor<Pose3>(X(i),X(i+1), poseFrom.between(poseTo),
                                                      odometryNoise));

            }
        }
        cout <<"[ Graph ] found: " << found;

        if (found == 0){
            mPoseMap.unlock();
            return;
        }

        if (found > 1){
            isamTM->update(gtSAMgraphTM, initialEstimateTM);
            isamTM->update();  // FixMe: unconstrained problem?
            isamTM->update();
            gtSAMgraphTM.resize(0);
            initialEstimateTM.clear();

            isamCurrentEstimateTM = isamTM->calculateEstimate();
            optimized = true;
        }

//        gtsam::LevenbergMarquardtParams lm_params;
//        lm_params.setLinearSolverType("MULTIFRONTAL_QR");
//        lm_params.setMaxIterations(30);
//        gtsam::LevenbergMarquardtOptimizer lm_optimizer(gtSAMgraphTM, initialEstimateTM, lm_params);
//        isamCurrentEstimateTM = lm_optimizer.optimize();
//        isamCurrentEstimateTM.print("current estimate:\n");

        // cout<<"Estimate size: "<<length<<endl;
        // update global pose
        iterIni = globalPoseMap.begin();

        Eigen::Matrix4d start;
        Eigen::Matrix4d end;
        Eigen::Matrix4d WVIO_T_body = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d WGPS_T_body = Eigen::Matrix4d::Identity();

        for (int i = 0; i < length; i++, iterIni++){
            // cout<<"i "<<i<<"length :"<<length<<endl;
            vector<double> globalPose(7,0);
            gtsamPose2Vector(isamCurrentEstimateTM.at<Pose3>(X(i)),globalPose);
            iterIni->second = globalPose;

            double t = iterIni->first;
            WVIO_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(localPoseMap[t][3], localPoseMap[t][4],
                                                               localPoseMap[t][5], localPoseMap[t][6]).toRotationMatrix();
            WVIO_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(localPoseMap[t][0], localPoseMap[t][1], localPoseMap[t][2]);
            WGPS_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(globalPose[3], globalPose[4],
                                                               globalPose[5], globalPose[6]).toRotationMatrix();
            WGPS_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(globalPose[0], globalPose[1], globalPose[2]);

            WGlobal_T_WLocal = WGPS_T_body * WVIO_T_body.inverse();

            if (i == 0 ) start = WGlobal_T_WLocal;
            if (i == length - 1 ) end = WGlobal_T_WLocal;
        }

        // update noise model
//        priorVelNoise  = gtsam::noiseModel::Gaussian::Covariance(isamTM->marginalCovariance(V(length-1)));
//        priorBiasNoise = gtsam::noiseModel::Gaussian::Covariance(isamTM->marginalCovariance(B(length-1)));
        imuQueOpt.clear();
        imuQueOpt = imuQueImu;
        // cout<<"Tgl change "<<deltaTransGL<<endl;
        // w. consistency check: GTSAM implementation needs CC as well
        // OR the process will just crash without warning (the opt. time takes more than 500 ms)
        // Tgl change too much, forfeit this optimization
        Eigen::Matrix4d TglDelta = start.inverse() * end;
        // cout<<"x y z: "<<TglDelta(0,3)<<" "<<TglDelta(1,3)<<" "<<TglDelta(2,3)<<endl;
        double deltaTransGL = sqrt(TglDelta(0,3)*TglDelta(0,3) +
                                   TglDelta(1,3)*TglDelta(1,3) +
                                   TglDelta(2,3)*TglDelta(2,3) );
        if ( deltaTransGL > 1){
            cout<<"[ WARN ] Reset when deltaTgl = " << deltaTransGL << endl;
//                resetOptimization(backupTgl);
            setTgl(Eigen::Matrix4d::Identity());
            isInitialized = false;
        }
//            else setTgl(end);

//            double opt_t = opt_time.toc();
//             cout<<"[ Graph ] optimization takes: "<<opt_t<<" ms"<<endl; // gtsam implementation usually takes less than 10 ms
        cout << "... optimization done. " << endl; // gtsam implementation usually takes less than 10 ms

//            if(opt_t> 100) cout<<"gtsam opt. takes more than 100 ms"<<endl;

        mPoseMap.unlock();
    }
    // waiting for poses to be accumulated
//        std::chrono::milliseconds dura(500);
//        std::this_thread::sleep_for(dura);

}

/* add all imu factors */
//void GraphOptimization::optimizeWithIMU(){
//
//    if(newGlobalLocPose){
//
//        if (reInitialize) reInitialize = false;
//
////            TicToc opt_time;
//        newGlobalLocPose = false;
//        // printf("global optimization using global localization and odometry\n");
////            TicToc globalOptimizationTime;
//
//        gtsam::NavState prevState_, propState_;
//        gtsam::imuBias::ConstantBias prevBias_;
//        gtsam::Vector3 preVel_;
//
//        // gtsam
//        NonlinearFactorGraph gtSAMgraphTM;
//        Values initialEstimateTM;
//        ISAM2 *isamTM;
//        Values isamCurrentEstimateTM;
//        ISAM2Params parameters;
//        parameters.factorization = ISAM2Params::QR;
//        parameters.relinearizeThreshold = 0.1;
//        parameters.relinearizeSkip = 1;
//        isamTM = new ISAM2(parameters);
//
//        //add param
//        mPoseMap.lock();
//        bool optimized = false, founded = false, addimu = false;
//        const double preIntegr_thre = 0.25, hz = 600;  // TODO: big integration difference between poses would crash?
//        int length = localPoseMap.size();
//        // cout<<" pose no. before opt "<< length<<endl;
//        map<double, vector<double> >::iterator iterIni, iterLIO, iterLIOnext, iterGlobalLoc;
//        iterIni = globalPoseMap.begin();  //using odomTOmap value for initial guess
//        int i = 0;
//        int found = 0;
//        // int outlierNO = 0;
//        for (iterLIO = localPoseMap.begin(); iterLIO != localPoseMap.end(); iterLIO++, i++, iterIni++){
//
//            gtsam::Pose3 poseGuess = QT2gtsamPose(iterIni->second);
//            initialEstimateTM.insert(X(i), poseGuess);
//
//            founded = false;
//            double t = iterLIO->first;
//            iterGlobalLoc = globalLocPoseMap.find(t); // finding global loc pose
//            if (iterGlobalLoc != globalLocPoseMap.end()){
//                /// BA prior factor
//                gtsam::Pose3 poseGlobal = QT2gtsamPose(iterGlobalLoc->second);
//                // seems to be overconfident
//                // noiseModel::Diagonal::shared_ptr corrNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1).finished()); // rad*rad, meter*meter
//                double tE = iterGlobalLoc->second[7] / 2;
//                double tQ = iterGlobalLoc->second[8] / 2;
//                noiseModel::Diagonal::shared_ptr corrNoise = noiseModel::Diagonal::Variances(
//                        (Vector(6) << tQ,tQ,tQ,tE,tE,tE).finished()); // rad*rad, meter*meter
//                gtSAMgraphTM.add(PriorFactor<Pose3>(X(i), poseGlobal, corrNoise));
//                found ++;
//                founded = true;
//
//                // for CC
//                Eigen::Matrix4d local = Eigen::Matrix4d::Identity();
//                Eigen::Matrix4d global = Eigen::Matrix4d::Identity();
//                global.block<3, 3>(0, 0) = Eigen::Quaterniond(iterGlobalLoc->second[3], iterGlobalLoc->second[4],
//                                                              iterGlobalLoc->second[5], iterGlobalLoc->second[6]).toRotationMatrix();
//                global.block<3, 1>(0, 3) = Eigen::Vector3d(iterGlobalLoc->second[0], iterGlobalLoc->second[1], iterGlobalLoc->second[2]);
//                local.block<3, 3>(0, 0) = Eigen::Quaterniond(iterLIO->second[3], iterLIO->second[4],
//                                                             iterLIO->second[5], iterLIO->second[6]).toRotationMatrix();
//                local.block<3, 1>(0, 3) = Eigen::Vector3d(iterLIO->second[0], iterLIO->second[1], iterLIO->second[2]);
//                backupTgl = global * local.inverse(); // get the newest Tgl as backup
//
//            }
//
//            if(i > 0 && found > 0){  // optimize once
//                isamTM->update(gtSAMgraphTM, initialEstimateTM);
//                isamTM->update();  // FixMe: unconstrained problem?
//                isamTM->update();
//                gtSAMgraphTM.resize(0);
//                initialEstimateTM.clear();
//
//                isamCurrentEstimateTM = isamTM->calculateEstimate();
//                optimized = true;
//            }
//
//            iterLIOnext = iterLIO;
//            iterLIOnext++;
//            if(iterLIOnext != localPoseMap.end()){
//
//                /// odometry factor
//                noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances(
//                        (Vector(6) << iterLIOnext->second[7], iterLIOnext->second[7], iterLIOnext->second[7],
//                                iterLIOnext->second[8], iterLIOnext->second[8], iterLIOnext->second[8]).finished());
//                gtsam::Pose3 poseFrom = QT2gtsamPose(iterLIO->second);
//                gtsam::Pose3 poseTo = QT2gtsamPose(iterLIOnext->second);
//                gtSAMgraphTM.add(BetweenFactor<Pose3>(X(i),X(i+1), poseFrom.between(poseTo),
//                                                      odometryNoise));
//
//                if (optimized ){
//                    if (addimu) {
//                        prevBias_ = isamCurrentEstimateTM.at<gtsam::imuBias::ConstantBias>(B(i));
////                    poseFrom = isamCurrentEstimateTM.at<gtsam::Pose3>(X(i));
//                        prevState_ = gtsam::NavState(isamCurrentEstimateTM.at<gtsam::Pose3>(X(i)),
//                                                     isamCurrentEstimateTM.at<gtsam::Vector3>(V(i)));
//                        cout << "[ Debug ] optimized frame " << i << ", v: " << prevState_.v().transpose() << ", ba: "
//                        << prevBias_.accelerometer().transpose() << ", bg: " << prevBias_.gyroscope().transpose() << endl;
//                    }
//                }else {
//                    prevBias_ = gtsam::imuBias::ConstantBias((gtsam::Vector(6) << 0, 0, 0,  // ba~0
//                            iterLIOnext->second[15], iterLIOnext->second[16], iterLIOnext->second[17]).finished());
//                    prevBias_ = gtsam::imuBias::ConstantBias((gtsam::Vector(6) << 0, 0, 0, 0,0,0).finished());
//
//                    prevState_ = gtsam::NavState(poseFrom, (gtsam::Vector(3) << iterLIOnext->second[9],
//                            iterLIOnext->second[10], iterLIOnext->second[11]).finished());
//                }
//
//
//                /// IMU preintegration
//                imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
//                int imuCnt=0;
//                while (!imuQueOpt.empty()){
//                    // pop and integrate imu data that is between two optimizations
//                    sensor_msgs::Imu *thisImu = &imuQueOpt.front();
//                    double imuTime = thisImu->header.stamp.toSec();
//                    if (imuTime < iterLIO->first)
//                        imuQueOpt.pop_front();
//                    else if (imuTime < iterLIOnext->first){
////                        cout << "imuTime: " << to_string(imuTime) << endl;
//                        double dt = (lastImuT_opt < 0) ? (1.0 / hz) : (imuTime - lastImuT_opt);
//                        if (dt != 0){
//                            imuIntegratorOpt_->integrateMeasurement(
//                                    gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
//                                    gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
//                            imuCnt++;
//                            lastImuT_opt = imuTime;
//                        }
//                        imuQueOpt.pop_front();
//                    }else
//                        break;
//                }
////                cout << "[ Graph ] integrated IMU " << imuCnt << " between " << i << " & " << i+1 << endl;
//                propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
//
//                gtsam::Pose3 poseBetw  = poseFrom.between(poseTo);
//                gtsam::Pose3 poseBetw_imu = prevState_.pose().between(propState_.pose());
//                poseBetw = poseBetw.inverse() * poseBetw_imu;
//                Eigen::AngleAxisd angleAxisd(poseBetw.rotation().matrix());
////                cout << "[ Debug ] integration errors: " << angleAxisd.angle() << " / " << poseBetw.translation().norm() << endl;
//                if (poseBetw.translation().norm() < preIntegr_thre) {
//
//                    if (!addimu){
//                        /// velocity prior
//                        gtsam::PriorFactor<gtsam::Vector3> priorVel(V(i), prevState_.v(), priorVelNoise);
//                        gtSAMgraphTM.add(priorVel);
//                        /// bias prior
//                        gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(i), prevBias_, priorBiasNoise);
//                        gtSAMgraphTM.add(priorBias);
//                        initialEstimateTM.insert(V(i), prevState_.v());
//                        initialEstimateTM.insert(B(i), prevBias_);
//                    }
//                    /// IMU factor
//                    const auto &preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements &>(*imuIntegratorOpt_);
//                    gtsam::ImuFactor imu_factor(X(i), V(i), X(i+1), V(i+1), B(i), preint_imu);
//                    gtSAMgraphTM.add(imu_factor);
//                    // add imu bias between factor
//                    gtSAMgraphTM.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>
//                                             (B(i), B(i+1),
//                                              gtsam::imuBias::ConstantBias(),
//                                              gtsam::noiseModel::Diagonal::Sigmas(
//                                                      sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
//
////                initialEstimateTM.insert(X(i), propState_.pose());
//                    initialEstimateTM.insert(V(i+1), propState_.v());
//                    initialEstimateTM.insert(B(i+1), prevBias_);
//                    addimu = true;
//                }else
//                    addimu = false;
//            }
//        }
//        cout <<"[ Graph ] found: " << found;
//
//        if (found == 0){
//            mPoseMap.unlock();
//            return;
//        }
//
//        isamTM->update(gtSAMgraphTM, initialEstimateTM);
//        isamTM->update();  // FixMe: unconstrained problem? dt=0
//        isamTM->update();
//        gtSAMgraphTM.resize(0);
//        initialEstimateTM.clear();
//
//        isamCurrentEstimateTM = isamTM->calculateEstimate();
//        optimized = true;
//
//
////        gtsam::LevenbergMarquardtParams lm_params;
////        lm_params.setLinearSolverType("MULTIFRONTAL_QR");
////        lm_params.setMaxIterations(30);
////        gtsam::LevenbergMarquardtOptimizer lm_optimizer(gtSAMgraphTM, initialEstimateTM, lm_params);
////        isamCurrentEstimateTM = lm_optimizer.optimize();
////        isamCurrentEstimateTM.print("current estimate:\n");
//
//        // cout<<"Estimate size: "<<length<<endl;
//        // update global pose
//        iterIni = globalPoseMap.begin();
//
//        Eigen::Matrix4d start;
//        Eigen::Matrix4d end;
//        Eigen::Matrix4d WVIO_T_body = Eigen::Matrix4d::Identity();
//        Eigen::Matrix4d WGPS_T_body = Eigen::Matrix4d::Identity();
//
//        for (int i = 0; i < length; i++, iterIni++){
//            // cout<<"i "<<i<<"length :"<<length<<endl;
//            vector<double> globalPose(7,0);
//            gtsamPose2Vector(isamCurrentEstimateTM.at<Pose3>(X(i)),globalPose);
//            iterIni->second = globalPose;
//
//            double t = iterIni->first;
//            WVIO_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(localPoseMap[t][3], localPoseMap[t][4],
//                                                               localPoseMap[t][5], localPoseMap[t][6]).toRotationMatrix();
//            WVIO_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(localPoseMap[t][0], localPoseMap[t][1], localPoseMap[t][2]);
//            WGPS_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(globalPose[3], globalPose[4],
//                                                               globalPose[5], globalPose[6]).toRotationMatrix();
//            WGPS_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(globalPose[0], globalPose[1], globalPose[2]);
//
//            WGlobal_T_WLocal = WGPS_T_body * WVIO_T_body.inverse();
//
//            if (i == 0 ) start = WGlobal_T_WLocal;
//            if (i == length - 1 ) end = WGlobal_T_WLocal;
//        }
//
//        // update noise model
////        priorVelNoise  = gtsam::noiseModel::Gaussian::Covariance(isamTM->marginalCovariance(V(length-1)));
////        priorBiasNoise = gtsam::noiseModel::Gaussian::Covariance(isamTM->marginalCovariance(B(length-1)));
//        imuQueOpt.clear();
//        imuQueOpt = imuQueImu;
//        // cout<<"Tgl change "<<deltaTransGL<<endl;
//        // w. consistency check: GTSAM implementation needs CC as well
//        // OR the process will just crash without warning (the opt. time takes more than 500 ms)
//        // Tgl change too much, forfeit this optimization
//        Eigen::Matrix4d TglDelta = start.inverse() * end;
//        // cout<<"x y z: "<<TglDelta(0,3)<<" "<<TglDelta(1,3)<<" "<<TglDelta(2,3)<<endl;
//        double deltaTransGL = sqrt(TglDelta(0,3)*TglDelta(0,3) +
//                                   TglDelta(1,3)*TglDelta(1,3) +
//                                   TglDelta(2,3)*TglDelta(2,3) );
//        if ( deltaTransGL > 1){
//            cout<<"[ WARN ] Reset when deltaTgl = " << deltaTransGL << endl;
////                resetOptimization(backupTgl);
//            setTgl(Eigen::Matrix4d::Identity());
//            isInitialized = false;
//        }
////            else setTgl(end);
//
////            double opt_t = opt_time.toc();
////             cout<<"[ Graph ] optimization takes: "<<opt_t<<" ms"<<endl; // gtsam implementation usually takes less than 10 ms
//        cout << "... optimization done. " << endl; // gtsam implementation usually takes less than 10 ms
//
////            if(opt_t> 100) cout<<"gtsam opt. takes more than 100 ms"<<endl;
//
//        mPoseMap.unlock();
//    }
//    // waiting for poses to be accumulated
////        std::chrono::milliseconds dura(500);
////        std::this_thread::sleep_for(dura);
//
//}

bool GraphOptimization::updatePosesGlobal(vector<Eigen::Quaterniond> &q_poses_,
                                          vector<Eigen::Vector3d> &t_poses_, int win_base){

    int locN = globalLocPoseMap.size();
    if (locN < 3) return false;
//    cout << "[Debug] globalLocPoseMap: " << locN << endl;

    // pop out old imu msg
    double currentCorrectionTime = localPoseMap.begin()->first;
    double lastImuQT = -1;
    while (!imuQueImu.empty() && imuQueImu.front().header.stamp.toSec() < currentCorrectionTime){
        lastImuQT = imuQueImu.front().header.stamp.toSec();
        imuQueImu.pop_front();
        imuQueOpt.pop_front();
    }

    if (locN > maxFrameNum - 55)
        optimize();
    else
        optimizeWithIMU();
    globalLocPoseMap.clear();

    if (globalPoseMap.empty()) return false;
//    assert(q_poses_.size() >= win_base + maxFrameNum);
//    assert(localPoseMap.size() == globalPoseMap.size());
//    assert(globalPoseMap.size() == maxFrameNum);

    auto iter = globalPoseMap.begin();
    for (int j = 0; j < maxFrameNum; ++j, iter++) {
        q_poses_[win_base+j] = Eigen::Quaterniond(iter->second[3], iter->second[4],
                                                  iter->second[5], iter->second[6]);
        t_poses_[win_base+j] = Eigen::Vector3d(iter->second[0], iter->second[1], iter->second[2]);
    }
    return true;
}

#endif //BALM_GRAPH_OPT_HPP
