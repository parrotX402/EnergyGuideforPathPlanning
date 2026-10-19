#!/usr/bin/env python3
import rospy
import math
from gazebo_msgs.msg import ModelState

def moving():
    rospy.init_node('Slalom_2_node')
    pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
    
    model_name = 'Slalom_2' 
    rate = rospy.Rate(50) 
    
    rospy.loginfo("动态障碍物控制节点已启动，支持环境重置")

    while not rospy.is_shutdown():
        try:
            t = rospy.get_time()
            
            # 计算位置逻辑
            new_state = ModelState()
            new_state.model_name = model_name
            new_state.pose.position.x = -1.0
            new_state.pose.position.y = 1.2 * math.sin(0.5 * t)
            new_state.pose.position.z = 0.25
            new_state.pose.orientation.w = 1.0
            
            pub_model.publish(new_state)
            
            # 正常休眠
            rate.sleep()
            
        except rospy.exceptions.ROSTimeMovedBackwardsException:
            # 关键：当发现时间回退（环境重置）时，直接跳过本次报错，进入下一个循环
            rospy.logwarn("检测到仿真重置，正在重新同步时间...")
            continue 
        except rospy.ROSInterruptException:
            break

if __name__ == '__main__':
    moving()
