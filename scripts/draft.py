                    self.x_corridor_center_first_time_corridor_frame , passability_view = self.find_a_virtual_corridor(z_min, z_max, self.corridor_middle_point_depth_first_time_camera_frame, min_clearance)
                    # once virtual corridor is found, we can consider corridor defined for next iterations
                    # calculate x coordinate of corridor center in robot base frame at start yaw frame
                    if self.x_corridor_center_first_time_corridor_frame is not None:
                        Point_corridor_center_first_time_3d_camera_frame_temp = np.array([0.0, 0.0, self.corridor_middle_point_depth_first_time_camera_frame]) # y is 0 because we are considering corridor center point which is at the same height as camera, and we are only interested in x and z for corridor center point. and we will transform this point to robot frame to get x coordinate of corridor center in robot frame.
                        self.Point_corridor_center_first_time_3d_robot_frame_temp = self.R_rc @ Point_corridor_center_first_time_3d_camera_frame_temp + self.t_rc
                        self.Point_corridor_center_first_time_2d_robot_frame_temp = self.Point_corridor_center_first_time_3d_robot_frame_temp[:2] # we only care about x and z coordinates in robot frame for corridor center point, and we will consider this as the corridor center point in robot frame for control. and we will ignore the y coordinate in robot frame because we are considering corridor center point which is at the same height as camera, so the y coordinate in robot frame should be 0 or close to 0.

                        # Store FIXED distance ( starting distance) of corridor center along corridor axis
                        self.corridor_center_distance_along_corridor_axis_first_time = np.dot(
                            self.Point_corridor_center_first_time_2d_robot_frame_temp,
                            self.corridor_forward_2d_unit_vector_first_time_robot_frame
                        )#Distance of corridor center along corridor axis at start time

                        self.corridor_center_distance_perpendicular_to_corridor_axis_first_time = self.x_corridor_center_first_time_corridor_frame #Distance of corridor center perpendicular to corridor axis at start time. this value should be close to 0 because we are considering the corridor center is on the corridor axis. but it may not be exactly 0 because of noise in the data from frame detection node and also because of the fact that we are considering the mid frame point as reference for corridor definition which may not be exactly on the corridor axis because of noise and errors in the data from frame detection node. but it should be close to 0 ideally.
                        
                        self.Point_corridor_center_first_time_2d_robot_frame = (
                            self.corridor_center_distance_along_corridor_axis_first_time * self.corridor_forward_2d_unit_vector_first_time_robot_frame
                            + self.corridor_center_distance_perpendicular_to_corridor_axis_first_time  * self.corridor_lateral_2d_unit_vector_first_time_robot_frame
                        )
                        rospy.loginfo(f"corridor_center_distance_along_corridor_axis_first_time: {self.corridor_center_distance_along_corridor_axis_first_time:.3f} m, corridor_center_distance_perpendicular_to_corridor_axis_first_time: {self.corridor_center_distance_perpendicular_to_corridor_axis_first_time:.3f} m")
                        rospy.loginfo(f"corridor center point in robot frame temp is {self.Point_corridor_center_first_time_2d_robot_frame_temp} m, corridor center point in robot frame calculated from corridor frame is {self.Point_corridor_center_first_time_2d_robot_frame} m")
                        self.start_yaw_at_corridor_definition = self.current_yaw  # store the yaw at corridor definition time
                        self.start_pose_at_corridor_definition = self.current_pose  # store the pose at corridor definition time

                        type_of_passability_check = 1  # after corridor definiton, first do corridor passability check
                        self.define_new_corridor_req_from_frame_detection = False  # corridor defined, no need to define again until next activation or next corridor definition request
                       

                    else:
                        self.define_new_corridor_req_from_frame_detection = True  # corridor is not yet defined, so need to run corridor definition again in next iteration. this can happen when virtual corridor finding fails to find a valid corridor center. in that case we can try again in next iteration because new data may come in and it may help to find a valid virtual corridor center. 
                        self.Point_corridor_center_first_time_2d_robot_frame = None # because corridor is not defined yet, we cannot calculate Point_corridor_center_first_time_2d_robot_frame. so set it to None. it will be calculated in next iteration once corridor is defined. and during this iteration, since corridor is not defined, we will skip passability check and wait for next iteration when corridor is defined.
                