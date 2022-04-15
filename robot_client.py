import naoqi
from naoqi import ALProxy
import qi
import motion
import argparse
import sys
import numpy as np

import redis

class robot_redis_client:
    def __init__(self, session, ip='192.168.0.121'):
        # self.autonomous_life = session.service("ALAutonomousLife")
        self.autonomous_life = ALProxy("ALAutonomousLife", ip, 9559)
        # self.tts = session.service("ALTextToSpeech")
        self.tts = ALProxy("ALTextToSpeech", ip, 9559)
        # self.motion_service = session.service("ALMotion")
        self.motion_service = ALProxy("ALMotion", ip, 9559)
        # self.posture_service = session.service("ALRobotPosture")
        self.posture_service = ALProxy("ALRobotPosture", ip, 9559)

        self.autonomous_life.setAutonomousAbilityEnabled("All", False)
        self.posture_service.goToPosture("Stand", 1.0)

        # joint control parameters
        # self.motion_service.setStiffnesses("Head", 0.0)
        self.motion_service.setStiffnesses("RArm", 1.0)
        # self.joint_names = ["KneePitch", "HipPitch", "HipRoll", "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll"]
        self.joint_names = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll"]
        self.chain_name = "RArm"
        self.frame = motion.FRAME_TORSO
        self.fractionMaxSpeed = 0.5
        self.target_position = [0.3, -0.2, 0.8]
        self.z_offset = 0.82

        self.message_id = 0
        self.r = redis.Redis(host='localhost', port=6379, db=0)

        self.sub_say = self.r.pubsub()
        self.sub_joint = self.r.pubsub()
        self.sub_cartesian = self.r.pubsub()
        self.sub_retrieve_data = self.r.pubsub()
        self.sub_go_home_pose = self.r.pubsub()
        self.sub_set_stiffness = self.r.pubsub()

        self.listening_thread_say = None
        self.listening_thread_joint = None
        self.listening_thread_cartesian = None
        self.listening_thread_retrieve_data = None
        self.listening_thread_go_home_pose = None
        self.listening_thread_set_stiffness = None

    def initialize_subscribers(self):
        self.sub_say.subscribe(**{'say': self.say_message_handler})
        self.sub_joint.subscribe(**{'target_joint_values': self.joint_message_handler})
        self.sub_cartesian.subscribe(**{'target_hand_position': self.cartesian_message_handler})
        self.sub_retrieve_data.subscribe(**{'retrieve_joint_values': self.retrieve_data_message_handler})
        self.sub_go_home_pose.subscribe(**{'go_home_pose': self.go_home_pose_message_handler})
        self.sub_set_stiffness.subscribe(**{'set_stiffness': self.set_stiffness_message_handler})

    def start_to_listen(self):
        self.listening_thread_say = self.sub_say.run_in_thread(sleep_time=0.01)
        self.listening_thread_joint = self.sub_joint.run_in_thread(sleep_time=0.01)
        self.listening_thread_cartesian = self.sub_cartesian.run_in_thread(sleep_time=2.0)
        self.listening_thread_retrieve_data = self.sub_retrieve_data.run_in_thread(sleep_time=0.01)
        self.listening_thread_go_home_pose = self.sub_go_home_pose.run_in_thread(sleep_time=0.01)
        self.listening_thread_set_stiffness = self.sub_set_stiffness.run_in_thread(sleep_time=0.01)

    def stop(self):
        self.listening_thread_say.stop()
        self.listening_thread_joint.stop()
        self.listening_thread_cartesian.stop()
        self.listening_thread_retrieve_data.stop()
        self.listening_thread_go_home_pose.stop()
        self.listening_thread_set_stiffness.stop()

    ''' Blocking Handlers '''
    def say_message_handler(self, message):
        print("[Robot Client]: Receive say command data: {}".format(message['data']))
        id = self.tts.post.say(message['data']) # non-blocking function
        self.tts.wait(id, 0)
        self.r.publish('finished_say', str(0))
        # self.message_id += 1

    def go_home_pose_message_handler(self, message):
        print("[Robot Client]: Receive command to go to home posture")
        id = self.posture_service.post.goToPosture("Stand", 1.0)
        self.posture_service.wait(id, 0)
        self.r.publish('finished_go_home_pose', str(0))

    def set_stiffness_message_handler(self, message):
        print("[Robot Client]: Receive command to set stiffness to {}".format(message['data']))
        # self.motion_service.killTasksUsingResources([self.chain_name])
        # print("Killed motion tasks before setting stiffness")
        id = self.motion_service.post.setStiffnesses("RArm", float(message['data']))
        self.motion_service.wait(id, 0)
        self.r.publish('finished_set_stiffness', str(0))

    ''' Non-blocking Handlers'''
    def joint_message_handler(self, message):
        print("[Robot Client]: Receive joint command data: {}".format(message['data']))

        # Send joint command
        joint_values = [float(i) for i in message['data'].split()]
        self.motion_service.setAngles(self.joint_names, joint_values, self.fractionMaxSpeed)
        self.message_id += 1
        print("[Robot Client]: Finish sending joint command")

        # # Get real cartesian position of end-effector in world frame
        # real_pose = self.motion_service.getTransform("RArm", motion.FRAME_TORSO, True)
        # real_position = []
        # for i in range(3):
        #     real_position.append(real_pose[i*4 + 3])
        # real_position[2] = real_position[2] + self.z_offset
        #
        # # Calculate difference between target and real position
        # diff = np.array([real_position[0] - self.target_position[0],
        #                  real_position[1] - self.target_position[1],
        #                  real_position[2] - self.target_position[2]])
        # diff_value = np.linalg.norm(diff)
        #
        # print("[Robot Client]: Real position is: {}".format(real_position))
        # print("[Robot Client]: Target position is: {}".format(self.target_position))
        # print("[Robot Client]: Error distance is: {}".format(diff_value))
        # print("********************************************")

    def retrieve_data_message_handler(self, message):
        print("[Robot Client]: Receive command to retrieve robot data")
        joint_values = ''
        # joint_angles = self.motion_service.getAngles(self.chain_name, True)
        # for i in range(len(self.joint_names)):
        #     joint_values += str(joint_angles[i]) + ' '

        for joint in self.joint_names:
            joint_angle = self.motion_service.getAngles(joint, True)[0]
            joint_values += str(joint_angle) + ' '

        # print("latest_joint_values: " + joint_values)
        self.r.publish('latest_joint_values', joint_values)

    def cartesian_message_handler(self, message):
        print("[Robot Client]: Receive cartesian command data: {}".format(message['data']))
        target_pos = [float(i) for i in message['data'].split()]
        target_ori = [0.0, 0.0, 0.0]
        target_pose = target_pos + target_ori
        axisMask = 7 # for position-control only

        self.motion_service.setPositions(self.chain_name, self.frame, target_pose, self.fractionMaxSpeed, axisMask)
        print("[Robot Client]: Finish sending cartesian command")


def main(session, ip):
    robot_client = robot_redis_client(session, ip)
    robot_client.initialize_subscribers()
    robot_client.start_to_listen()
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("User forced to quit")
        robot_client.posture_service.goToPosture("StandInit", 1.0)
        # robot_client.autonomous_life.setAutonomousAbilityEnabled("All", True)
        pass

    # robot_client.motion_service.setStiffnesses("RArm", 0.0)
    robot_client.stop()
    print("[Robot Client]: Finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="10.15.3.171",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
        print("Successfully connected")
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    main(session, args.ip)
