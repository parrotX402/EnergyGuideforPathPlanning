#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import random
import time
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose

class Respawn():
    def __init__(self):

        # 读取目标点的sdf文件
        self.modelPath = "/opt/ros/noetic/share/turtlebot3_gazebo/models/turtlebot3_square/goal_box/model.sdf"
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
 
        self.stage = 4
        self.goal_position = Pose()
        self.init_goal_x = 0.5
        self.init_goal_y = 1
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.modelName = 'goal'
        self.obstacle_1 = 0.6, 0.6
        self.obstacle_2 = 0.6, -0.6
        self.obstacle_3 = -0.6, 1
        self.obstacle_4 = -0.6, -0.6
        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        # 订阅 model_states 的话题
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.check_model = False
        self.index = 0

    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True

    def respawnModel(self):
        while True:
            if not self.check_model:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
                rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x,
                              self.goal_position.position.y)
                break
            else:
                pass

    def deleteModel(self):
        t0 = time.time()
        
        while True:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox(self.modelName) #删除模型，生成新的goal之后，删除旧的goal
                break
            else:
                rospy.sleep(0.05)
           
            if time.time() - t0 > 2.0:
                rospy.logwarn("[Respawn] deleteModel timeout, force continue")
                self.check_model = False
                break

    def getPosition(self, delete=False):

        if delete:
            self.deleteModel()
        if not hasattr(self, 'num_goals'):
            self.num_goals = 0

            # 前10次固定在 (0, 0)
        if self.num_goals < 1:
            self.goal_position.position.x = -1
            self.goal_position.position.y = 1 
            print("目标点固定在 (0, 0) for 前 100 次, 当前 num_goals = {}".format(self.num_goals))


        else:
        # 超过10次后，随机选择目标点
           
                goal_x_list = [ 1.8,  0.9, 0.75]
                goal_y_list = [-2, 1, 2]
                self.index = random.randrange(len(goal_x_list))

                print("抽到的序号是 index = {}".format(self.index))

                # 如果当前 index 和上次相同，重新选择
                if self.last_index == self.index:
                    print("目标点重复，重新选择")
                    position_check = True
                else:
                    self.last_index = self.index
                    position_check = False

                # 设置新的目标点坐标
                self.goal_position.position.x = goal_x_list[self.index]
                self.goal_position.position.y = goal_y_list[self.index]
                print('Goal_point({}, {})'.format(self.goal_position.position.x, self.goal_position.position.y))

    # 增加计数器
        self.num_goals += 1

        # 短暂停留以模拟目标点重生
        time.sleep(0.2)
        self.respawnModel()

        # 保存当前目标点为 "上次的目标点"
        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        # 返回新坐标点
        return self.goal_position.position.x, self.goal_position.position.y
