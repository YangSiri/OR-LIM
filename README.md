# OR-LIM
OR-LIM: Observability-aware robust LiDAR-Inertial-Mapping  under High Dynamic Sensor Motion


*__Abstract__*
LiDAR-based Simultaneous Localization And Mapping (SLAM) has been an impactful 
way for the reconstruction of environmental map. Yet, consistent mapping in sensing-degenerated and perceptually-limited scenes (e.g. multi-story buildings) or under high 
dynamic sensor motion (e.g. rotating platform) is still challenging. In this paper, we 
present a novel LiDAR-inertial-mapping (LIM) system with exceptional observability. 
At its core, it combines a robust real-time LiDAR-inertial-odometry (LIO) module with 
an efficient surfel-map-smoothing (SMS) module that seamlessly optimize the sensor 
poses and scene geometry in the meantime. The planar surfels are hierarchically 
generated and growed from point cloud map to provide reliable correspondences for 
the fixed-lag optimization. Moreover, the normals of surfels are analyzed for the 
observability evaluation of each frame. To maintain the global consistency, factor graph 
is utilized integrating the information from IMU propagation, LIO as well as the SMS. 
The robustness is extensively tested on the datasets collected by a low-cost multi-beam 
LiDAR (MBL) mounted on a rotating platform. The experiments conducted on 
complex multi-story building and large-scale outdoor scenes with various settings of 
sensor motion have shown the superior performance of our system over multiple state-of-the-art methods.


![image](https://user-images.githubusercontent.com/33154113/201501219-6d4f0977-cd02-46e8-8440-33a00b2d8728.png)

![image](https://user-images.githubusercontent.com/33154113/201502140-ff169b33-28be-4cf4-8898-eb2256b206e5.png)


Demo videos at https://www.youtube.com/watch?v=R8B6AtgdH8E

