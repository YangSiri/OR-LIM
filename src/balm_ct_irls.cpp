//
// Created by cyz on 2022/7/9.
//

//#include "balmclass_ct_Jacob.hpp"
#include "balmclass_ct.hpp"

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <queue>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PoseArray.h>

using namespace std;
double voxel_size[2] = {1, 1};

mutex mBuf;
queue<sensor_msgs::PointCloud2ConstPtr> surf_buf, corn_buf, full_buf;
queue<nav_msgs::Odometry::ConstPtr> odom_buf;

void surf_handler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    mBuf.lock();
    surf_buf.push(msg);
    mBuf.unlock();
}

void corn_handler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    mBuf.lock();
    corn_buf.push(msg);
    mBuf.unlock();
}

void full_handler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    mBuf.lock();
    full_buf.push(msg);
    mBuf.unlock();
}

void odom_handler(const nav_msgs::Odometry::ConstPtr &msg)
{
    mBuf.lock();
    odom_buf.push(msg);
    mBuf.unlock();
}

// point with time ratio
void cut_voxel_ct(unordered_map<VOXEL_LOC, OCTO_TREE*> &feat_map,
                  pcl::PointCloud<PointType>::Ptr pl_feat,
                  Eigen::Matrix3d R_p, Eigen::Vector3d t_p,
                  int feattype, int fnum, int capacity){

    uint plsize = pl_feat->size();
    double dura = pl_feat->points[plsize-1].curvature - pl_feat->points[0].curvature;
    for(uint i=0; i<plsize; i++){

        PointType &p_c = pl_feat->points[i];
        double time = 1;  // ratio
        time = (p_c.curvature - pl_feat->points[0].curvature) / dura;
        if (time < 0) time = 0;
        if (time > 1 || isnan(time)) time = 1;
        if (time == 1) time -= 1e-5;
        if (time == 0) time += 1e-5;

//        cout << "[Debug] ratio : " << time << endl;

        Eigen::Vector4d pvec_orig(p_c.x, p_c.y, p_c.z, time);
        Eigen::Vector3d pvec_tran = R_p * pvec_orig.topRows(3) + t_p;
        Eigen::Vector4d pvec_tran_t(pvec_tran.x(), pvec_tran.y(), pvec_tran.z(), time);

        float loc_xyz[3];
        for(int j=0; j<3; j++){
            loc_xyz[j] = pvec_tran[j] / voxel_size[feattype];
            if(loc_xyz[j] < 0)
                loc_xyz[j] -= 1.0;
        }

        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        auto iter = feat_map.find(position);
        if(iter != feat_map.end()){

            iter->second->plvec_orig[fnum]->push_back(pvec_orig);
            iter->second->plvec_tran[fnum]->push_back(pvec_tran_t);
            iter->second->is2opt = true;
//            if (iter->second->plvec_orig[fnum]->size() > 10000)
//                cout << "current size: " << iter->second->plvec_orig[fnum]->size() << endl;
        }else{

            OCTO_TREE *ot = new OCTO_TREE(feattype, capacity);
            ot->plvec_orig[fnum]->push_back(pvec_orig);
            ot->plvec_tran[fnum]->push_back(pvec_tran_t);

            ot->voxel_center[0] = (0.5+position.x) * voxel_size[feattype];
            ot->voxel_center[1] = (0.5+position.y) * voxel_size[feattype];
            ot->voxel_center[2] = (0.5+position.z) * voxel_size[feattype];
            ot->quater_length = voxel_size[feattype] / 4.0;
            feat_map[position] = ot;
        }
    }
}

void transformPointCloud_CT(const Eigen::Quaterniond &R1, const Eigen::Quaterniond &R2,
                            const Eigen::Vector3d &t1, const Eigen::Vector3d &t2,
                            const pcl::PointCloud<PointType>::Ptr &p1,
                            pcl::PointCloud<PointType>::Ptr p2) {

    int n = p1->points.size();
    p2->points.resize(n);
    double dura = p1->points[n-1].curvature - p1->points[0].curvature;

    for (int i = 0; i < n; ++i) {

        Eigen::Vector3d p_T;
        double ratio = (p1->points[i].curvature - p1->points[0].curvature) / dura;

        p_T = (R1.slerp(ratio, R2)) * (p1->points[i].getVector3fMap().cast<double>());

        p_T(0) += (1-ratio) * t1(0) + ratio * t2(0);
        p_T(1) += (1-ratio) * t1(1) + ratio * t2(1);
        p_T(2) += (1-ratio) * t1(2) + ratio * t2(2);

        p2->points[i] = p1->points[i];
        p2->points[i].x = p_T(0);
        p2->points[i].y = p_T(1);
        p2->points[i].z = p_T(2);
    }

}

#define MappingResolution 0.06
int main(int argc, char **argv){
    ros::init(argc, argv, "balm_CT_IRLS");
    ros::NodeHandle n;

    double surf_filter_length = 0.4;
    double corn_filter_length = 0.2;

    // todo : params
    int window_size ;
    int margi_size ;
    int filter_num = 1;
    int thread_num = 4;
    int skip_num = 0;
    int pub_skip = 1;
    string projPath = "/home/cyz/workspace/catkin_localizeInHDmap/src/FAST_LIO_LC-master/FAST-LIO/PCD/";
    bool ifoptSurf, ifoptCorn;
    string cornerCloudTopic, surfCloudTopic, fullCloudTopic, poseTopic;  // topic names

    n.param<double>("surf_filter_length", surf_filter_length, 0.15);
    n.param<double>("corn_filter_length", corn_filter_length, 0.2);
    n.param<double>("root_surf_voxel_size", voxel_size[0], 0.8);
    n.param<double>("root_corn_voxel_size", voxel_size[1], 1);
    n.param<double>("surf_feat_eigen_limit", feat_eigen_limit[0], 9);  // recut/marginalized feature threshold
    n.param<double>("corn_feat_eigen_limit", feat_eigen_limit[1], 4);
    n.param<double>("surf_opt_feat_eigen_limit", opt_feat_eigen_limit[0], 13);  // select voxel used for BALM
    n.param<double>("corn_opt_feat_eigen_limit", opt_feat_eigen_limit[1], 9);
    n.param<int>("pub_skip", pub_skip, 1);
    n.param<int>("filter_num",filter_num,3);  // point num of every frame within one voxel
    n.param<int>("skip_num", skip_num, 0);
    n.getParam("/projpath", projPath);  // global param add '/' before name
    n.param<bool>("ifoptSurf", ifoptSurf, true);
    n.param<bool>("ifoptCorn", ifoptCorn, false);
    n.param<int>("window_size", window_size,26);
    n.param<int>("margi_size",  margi_size, 8);
//    n.param<string>("cornerCloudTopic",cornerCloudTopic,"/laser_cloud_less_sharp");
    n.param<string>("cornerCloudTopic",cornerCloudTopic,"/cloud_registered_body");
    n.param<string>("surfCloudTopic",  surfCloudTopic,  "/cloud_registered_body");
//    n.param<string>("surfCloudTopic",  surfCloudTopic,  "/laser_cloud_less_flat");
//    n.param<string>("surfCloudTopic",  surfCloudTopic,  "/curLineCloud");
    n.param<string>("fullCloudTopic",  fullCloudTopic,  "/cloud_registered_body");
    n.param<string>("poseTopic",       poseTopic,       "/Odometry");
//    n.param<string>("poseTopic",       poseTopic,       "/odom_pose");

    cout << "[ BALM ] projPath : "  << projPath << endl;
    cout << "[ BALM ] ifoptSurf : " << ifoptSurf << endl;
    cout << "[ BALM ] ifoptCorn : " << ifoptCorn << endl;
    cout << "[ BALM ] window_size : "<< window_size << endl;
    cout << "[ BALM ] margi_size : " << margi_size << endl;
    cout << "[ BALM ] cornerCloudTopic : " << cornerCloudTopic << endl;
    cout << "[ BALM ] surfCloudTopic : " << surfCloudTopic << endl;
    cout << "[ BALM ] fullCloudTopic : " << fullCloudTopic << endl;
    cout << "[ BALM ] poseTopic : " << poseTopic << endl;
    cout << "[ BALM ] surf_filter_length : " << surf_filter_length << endl;
    cout << "[ BALM ] root_surf_voxel_size : " << voxel_size[0] << endl;
    int jump_flag = skip_num;
    printf("skip_num : %d\n", skip_num);

//    ros::Subscriber sub_corn = n.subscribe<sensor_msgs::PointCloud2>("/corn_last", 100, corn_handler);
//    ros::Subscriber sub_surf = n.subscribe<sensor_msgs::PointCloud2>("/surf_last", 100, surf_handler);
//    ros::Subscriber sub_full = n.subscribe<sensor_msgs::PointCloud2>("/full_last", 100, full_handler);
//    ros::Subscriber sub_odom = n.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 100, odom_handler);

    /// personal odometry test
    ros::Subscriber sub_corn = n.subscribe<sensor_msgs::PointCloud2>(cornerCloudTopic, 100, corn_handler);
    ros::Subscriber sub_surf = n.subscribe<sensor_msgs::PointCloud2>(surfCloudTopic, 100, surf_handler);
    ros::Subscriber sub_full = n.subscribe<sensor_msgs::PointCloud2>(fullCloudTopic, 100, full_handler);
    ros::Subscriber sub_odom = n.subscribe<nav_msgs::Odometry>(poseTopic, 100, odom_handler);

    ros::Publisher pub_corn = n.advertise<sensor_msgs::PointCloud2>("/map_corn", 10);
    ros::Publisher pub_surf = n.advertise<sensor_msgs::PointCloud2>("/map_surf", 10);
    ros::Publisher pub_full = n.advertise<sensor_msgs::PointCloud2>("/map_full", 10);
    ros::Publisher pub_test = n.advertise<sensor_msgs::PointCloud2>("/map_test", 10);
    ros::Publisher pub_odom = n.advertise<nav_msgs::Odometry>("/odom_rviz_last", 10);
    ros::Publisher pub_pose = n.advertise<geometry_msgs::PoseArray>("/poseArrayTopic", 10);

    LM_SLWD_VOXEL opt_lsv(window_size, filter_num, thread_num);

    pcl::PointCloud<PointType>::Ptr pl_corn(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr pl_surf(new pcl::PointCloud<PointType>);
    vector<pcl::PointCloud<PointType>::Ptr> pl_full_buf;

    vector<Eigen::Quaterniond> q_poses;
    vector<Eigen::Vector3d> t_poses;
    vector<double> posesTimes;
    vector<bool> frameConverg;
    Eigen::Quaterniond q_odom, q_gather_pose(1, 0, 0, 0) ,q_last(1, 0, 0, 0);
    Eigen::Vector3d t_odom, t_gather_pose(0, 0, 0), t_last(0, 0, 0);
    int plcount = 0, window_base = 0;

    unordered_map<VOXEL_LOC, OCTO_TREE*> surf_map, corn_map;
    Eigen::Matrix4d trans(Eigen::Matrix4d::Identity());
    geometry_msgs::PoseArray parray;
    parray.header.frame_id = "base_link";

    while(n.ok()){

        ros::spinOnce();
        if((ifoptCorn && corn_buf.empty()) || surf_buf.empty() || full_buf.empty() || odom_buf.empty())
            continue;

        mBuf.lock();
        uint64_t time_odom = odom_buf.front()->header.stamp.toNSec() / 1000;

        if (ifoptCorn){
            uint64_t time_corn = corn_buf.front()->header.stamp.toNSec() / 1000;
            if(time_odom != time_corn){
                time_odom < time_corn ? odom_buf.pop() : corn_buf.pop();
                mBuf.unlock();
                continue;
            }
        }
        uint64_t time_surf = surf_buf.front()->header.stamp.toNSec() / 1000;
        uint64_t time_full = full_buf.front()->header.stamp.toNSec() / 1000;

//        cout << "[ Debug ] time_corn " << time_corn << endl;
//        cout << "[ Debug ] time_surf " << time_surf << endl;
//        cout << "[ Debug ] time_full " << time_full << endl;
//        cout << "[ Debug ] time_odom " << time_odom << endl << endl;


        if(time_odom != time_surf)
        {
            time_odom < time_surf ? odom_buf.pop() : surf_buf.pop();
            mBuf.unlock();
            continue;
        }
        if(time_odom != time_full)
        {
            time_odom < time_full ? odom_buf.pop() : full_buf.pop();
            mBuf.unlock();
            continue;
        }

        ros::Time ct(full_buf.front()->header.stamp);
        pcl::PointCloud<PointType>::Ptr pl_full(new pcl::PointCloud<PointType>);
        rosmsg2ptype(*surf_buf.front(), *pl_surf);
        if (ifoptCorn)
            rosmsg2ptype(*corn_buf.front(), *pl_corn);
        rosmsg2ptype(*full_buf.front(), *pl_full);
        corn_buf.pop(); surf_buf.pop(); full_buf.pop();

//        pcl::io::savePCDFileBinaryCompressed(projPath + "corner.pcd", *pl_corn);
//        pcl::io::savePCDFileBinaryCompressed(projPath + "surf.pcd", *pl_surf);
//        pcl::io::savePCDFileBinaryCompressed(projPath + "full.pcd", *pl_full);

        q_odom.w() = odom_buf.front()->pose.pose.orientation.w;
        q_odom.x() = odom_buf.front()->pose.pose.orientation.x;
        q_odom.y() = odom_buf.front()->pose.pose.orientation.y;
        q_odom.z() = odom_buf.front()->pose.pose.orientation.z;
        t_odom.x() = odom_buf.front()->pose.pose.position.x;
        t_odom.y() = odom_buf.front()->pose.pose.position.y;
        t_odom.z() = odom_buf.front()->pose.pose.position.z;
        odom_buf.pop();
        mBuf.unlock();

        Eigen::Vector3d delta_t(q_last.matrix().transpose()*(t_odom-t_last));
        Eigen::Quaterniond delta_q(q_last.matrix().transpose() * q_odom.matrix());
        q_last = q_odom;  t_last = t_odom;

        // fixme : useless???
        t_gather_pose = t_gather_pose + q_gather_pose * delta_t;
        q_gather_pose = q_gather_pose * delta_q;
        if(jump_flag < skip_num)
        {
            jump_flag++;
            continue;
        }
        jump_flag = 0;

        // save pose initials
        if(plcount == 0){
            q_poses.push_back(q_gather_pose);
            t_poses.push_back(t_gather_pose);
        }
        else{
            t_poses.push_back(t_poses[plcount-1] + q_poses[plcount-1] * t_gather_pose);
            q_poses.push_back(q_poses[plcount-1]*q_gather_pose);
        }
        posesTimes.emplace_back(ct.toSec());

        parray.header.stamp = ct;
        geometry_msgs::Pose apose;
        apose.orientation.w = q_poses[plcount].w();
        apose.orientation.x = q_poses[plcount].x();
        apose.orientation.y = q_poses[plcount].y();
        apose.orientation.z = q_poses[plcount].z();
        apose.position.x = t_poses[plcount].x();
        apose.position.y = t_poses[plcount].y();
        apose.position.z = t_poses[plcount].z();
        parray.poses.push_back(apose);
        frameConverg.emplace_back(false);

//        pub_pose.publish(parray);

        pl_full_buf.push_back(pl_full);
        plcount++;
        assert(pl_full_buf.size() == plcount || q_poses.size() == plcount || t_poses.size() == plcount);

        OCTO_TREE::voxel_windowsize = plcount - window_base;
        q_gather_pose.setIdentity();  t_gather_pose.setZero();  ///

//        down_sampling_voxel(*pl_corn, corn_filter_length);
//        down_sampling_voxel(*pl_surf, surf_filter_length);

        int frame_head = plcount-1 - window_base;
        if(ifoptSurf){
            cut_voxel_ct(surf_map, pl_surf, q_poses.back().matrix(), t_poses.back(), 0, frame_head, window_size);
            for(auto iter=surf_map.begin(); iter!=surf_map.end(); ++iter)
                if(iter->second->is2opt){
                    iter->second->root_centors.clear();
                    iter->second->recut(0, frame_head, iter->second->root_centors);
                }

        }
        if(ifoptCorn){
            cut_voxel_ct(corn_map, pl_corn, q_poses.back().matrix(), t_poses.back(), 1, frame_head, window_size);
            for(auto iter=corn_map.begin(); iter!=corn_map.end(); ++iter)
                if(iter->second->is2opt){
                    iter->second->root_centors.clear();
                    iter->second->recut(0, frame_head, iter->second->root_centors);
                }
        }

        // map refine
        if(plcount >= window_base + window_size) {

            cout << "[ Debug ] Begin local map based BA at " << window_base << endl;
            for (int i = 0; i < window_size; i++) {

                opt_lsv.so3_poses[i].setQuaternion(q_poses[window_base + i]);
//                opt_lsv.so3_poses[i] = q_poses[window_base + i];  // for ceres
                opt_lsv.t_poses[i] = t_poses[window_base + i];
            }

            if (window_base > margi_size) {

                bool converged = false;
                if (ifoptSurf)
                    for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) {
                        if (iter->second->is2opt)
                            iter->second->traversal_opt(opt_lsv);
                    }

                if (ifoptCorn)
                    for (auto iter = corn_map.begin(); iter != corn_map.end(); ++iter) {
                        if (iter->second->is2opt)
                            iter->second->traversal_opt(opt_lsv);
                    }
                if (ifoptCorn || ifoptSurf)
//                    converged = opt_lsv.damping_iter();  // autoDiff
                    converged = opt_lsv.damping_iter_Analytical();  // analyticalDiff

                pcl::PointCloud<PointType> pl_send_bef, pl_send;
                for (int i = 1; i < window_size; i += pub_skip) {

//                    trans.block<3, 3>(0, 0) = opt_lsv.so3_poses[i].matrix();
//                    trans.block<3, 1>(0, 3) = opt_lsv.t_poses[i];

                    pcl::PointCloud<PointType>::Ptr pcloud(new pcl::PointCloud<PointType>());
//                    pcl::transformPointCloud(*pl_full_buf[window_base + i], *pcloud, trans);
                    transformPointCloud_CT(opt_lsv.so3_poses[i-1].unit_quaternion(),
                                           opt_lsv.so3_poses[i].unit_quaternion(),
                                           opt_lsv.t_poses[i-1], opt_lsv.t_poses[i],
                                           pl_full_buf[window_base + i], pcloud);
//                    transformPointCloud_CT(opt_lsv.so3_poses[i-1], opt_lsv.so3_poses[i],   // for ceres
//                                           opt_lsv.t_poses[i-1], opt_lsv.t_poses[i],
//                                           pl_full_buf[window_base + i], pcloud);
                    pl_send += *pcloud;

//                    pcl::transformPointCloud(*pl_full_buf[window_base + i], *pcloud, trans);
                    transformPointCloud_CT(q_poses[window_base + i-1],q_poses[window_base + i],
                                           t_poses[window_base + i-1], t_poses[window_base + i],
                                           pl_full_buf[window_base + i], pcloud);
                    pl_send_bef += *pcloud;

                    Eigen::Matrix<double ,3, 1> diff(opt_lsv.t_poses[i] - t_poses[window_base + i]);

                    if (diff.squaredNorm() < 1){  // update when converged?

                        q_poses[window_base + i] = opt_lsv.so3_poses[i].unit_quaternion();
//                        q_poses[window_base + i] = opt_lsv.so3_poses[i];  // for ceres
                        t_poses[window_base + i] = opt_lsv.t_poses[i];
                        frameConverg[window_base+i] = true;
                    }else frameConverg[window_base+i] = false;
                }
                pub_func(pl_send, pub_full, ct);
//                if(!converged)
                pcl::io::savePCDFileBinaryCompressed(projPath+"submap"+to_string(window_base)+".pcd", pl_send);
                pcl::io::savePCDFileBinaryCompressed(projPath+"submapBef"+to_string(window_base)+".pcd", pl_send_bef);

//            for(int i=0; i<margi_size; i++)
//                pl_full_buf[window_base + i] = nullptr;

                // update the poses in window of all poses
                for (int i = window_base; i < window_base + window_size; i++) {
                    parray.poses[i].orientation.w = q_poses[i].w();
                    parray.poses[i].orientation.x = q_poses[i].x();
                    parray.poses[i].orientation.y = q_poses[i].y();
                    parray.poses[i].orientation.z = q_poses[i].z();
                    parray.poses[i].position.x = t_poses[i].x();
                    parray.poses[i].position.y = t_poses[i].y();
                    parray.poses[i].position.z = t_poses[i].z();
                }
                parray.header.frame_id = converged? to_string(window_base+window_size): to_string(0);
                cout << "[ BALM ] publish seq: " << parray.header.frame_id << endl;
                pub_pose.publish(parray);
            }

            // marginalize
            if (ifoptSurf)
                for(auto iter=surf_map.begin(); iter!=surf_map.end(); ++iter)
                    if(iter->second->is2opt){
                        iter->second->root_centors.clear();
                        iter->second->marginalize(0, margi_size, q_poses, t_poses, window_base, iter->second->root_centors);
                    }

            if (ifoptCorn)
                for(auto iter=corn_map.begin(); iter!=corn_map.end(); ++iter)
                    if(iter->second->is2opt){
                        iter->second->root_centors.clear();
                        iter->second->marginalize(0, margi_size, q_poses, t_poses, window_base, iter->second->root_centors);
                    }

            window_base += margi_size;
            opt_lsv.free_voxel();
        }
    }

    /// save poses & map
    FILE *fp;
    string file = projPath + "keyposesBALM.txt";
    fp = fopen(file.data(),"w");
    for(int i=margi_size; i<plcount-margi_size; i+=pub_skip){

        fprintf(fp, "%lf %f %f %f %f %f %f %f\n", posesTimes[i],
                t_poses[i].x(), t_poses[i].y(), t_poses[i].z(),
                q_poses[i].x(), q_poses[i].y(), q_poses[i].z(), q_poses[i].w());
    }
    fclose(fp);
    cout << "Poses saved to " << file << endl;

    pcl::PointCloud<PointType>::Ptr pl_global(new pcl::PointCloud<PointType>);
    for(int i = margi_size; i < plcount-margi_size; i += pub_skip){

        trans.block<3, 3>(0, 0) = q_poses[i].matrix();
        trans.block<3, 1>(0, 3) = t_poses[i];

        pcl::PointCloud<PointType> pcloud;
        pcl::transformPointCloud(*pl_full_buf[i], pcloud, trans);
        *pl_global += pcloud;
//        cout << "[ Debug " << i << " ] global map size : " << pl_global->size() << endl;
    }
    pcl::VoxelGrid<PointType > voxelGrid;
    voxelGrid.setLeafSize(MappingResolution, MappingResolution, MappingResolution);
    voxelGrid.setInputCloud(pl_global);
    voxelGrid.filter(*pl_global);
    pcl::io::savePCDFileBinaryCompressed(projPath + "globalmapBALM.pcd", *pl_global);
    cout << "Global map saved." << endl;
}


