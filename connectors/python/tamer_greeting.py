import copy
import cv2
import mediapipe as mp

import threading
import keyboard

import time
from time import sleep

import numpy as np
import math
from scipy.integrate import quad

from social_interaction_cloud.action import ActionRunner
from social_interaction_cloud.basic_connector import BasicSICConnector

import sys
sys.path.append('/Users/ullrich/ullrich_ws/Socket_Connection/Socket_Connection')
import pepper_connector
from pepper_connector import socket_connection as connect

class Agent:
    def __init__(self, server_ip, pepper_ip="10.15.3.171", camera_mode=0):
        # SIC connector variable
        self.sic = BasicSICConnector(server_ip)
        self.action_runner = None
        self.pepper_ip = pepper_ip

        # Camera variable
        self.camera_mode = camera_mode
        self.camera_connector = None

        # MediaPipe variable
        self.mp_drawing = None
        self.mp_drawing_styles = None
        self.mp_pose = None
        self.mp_hands = None
        self.pose_detector = None

        # state variable
        self.request_for_state = False
        self.last_pose = None
        self.current_pose = None
        self.current_state = None

        # action variable
        self.num_actions = 3 # i.e., bump-fist, high-five, hand-wave
        self.get_latest_vision_info = False
        self.set_eye_color = False
        self.delta_t = 1.0
        # self.accuracy_rate = [0.0, 0.0, 0.0]

        # state history and feedback history:
        # experience of each action (i.e., each row of the experiences) is in the form of:
        # [[s1, t1_start, t1_end, h1], [s2, t2_start, t2_end, h2], ...]
        self.experiences = [[], [], []]
        self.event_history = []
        self.feedback_history = []
        self.max_experiences_num = 1000
        self.max_feedback_num = 20
        self.feedback_window_len = 0.8

        # thread control variable
        self.user_interrupt = False
        self.start_new_episode = False
        self.episode_time_out = False
        self.stopped = False

    def start_connect_and_initialize(self):
        # connect to the SIC server
        self.sic.start()
        self.action_runner = ActionRunner(self.sic)

        # do some initialization jobs
        self.action_runner.run_waiting_action('set_language', 'en-US')
        # self.action_runner.run_waiting_action('set_idle')
        self.initialize_camera()
        self.initialize_pose_detector()


        # self.action_runner.run_action('set_eye_color', 'white')
        # self.action_runner.run_vision_listener('people', self.get_face_position, True)

    def initialize_camera(self):
        # "mode 0" uses laptop web-camera for testing
        if self.camera_mode == 0:
            cam_port = 0
            self.camera_connector = cv2.VideoCapture(cam_port)

        # "mode 1" uses pepper camera
        elif self.camera_mode == 1:
            self.camera_connector = connect(self.pepper_ip, 12345, 0, 2)

        else:
            print("[Initialize Camera]: Invalid camera mode! Errors happened")

    def initialize_pose_detector(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose_detector = self.mp_pose.Pose(min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)
        self.hand_detector = self.mp_hands.Hands(static_image_mode=True,
                                                 max_num_hands=2,
                                                 model_complexity=0,
                                                 min_detection_confidence=0.5,
                                                 min_tracking_confidence=0.5)

    def get_latest_state(self):
        current_state = None

        # web-camera mode
        if self.camera_mode == 0:
            print("[Get Latest State]: Using web camera")
            success, image = self.camera_connector.read()
            if not success:
                print("[Get Latest State]: Web Camera failed to get image")
                return current_state, image
        # pepper camera mode
        else:
            print("[Get Latest State]: Using pepper camera")
            # self.request_for_image = True
            # while self.latest_image is None:
            #     pass
            # image = self.latest_image
            image = self.camera_connector.get_img()

        pose, image = self.estimate_pose(image)

        # return state as None if no human is detected
        if pose is None:
            pass
        else:
            # update pose memory
            if self.last_pose is None:
                self.last_pose = pose
            else:
                self.last_pose = self.current_pose
            self.current_pose = pose

            current_vel = (self.current_pose - self.last_pose) / self.delta_t # in the form of 1d np-array

            # the dimension of the state is 20 (5 landmarks x 2 dim-cordinate x 2 dim-vel)
            current_state = np.concatenate([pose, current_vel])

        return current_state, image

        # self.request_for_state = True
        #
        # while not self.get_latest_vision_info:
        #     if self.user_interrupt:
        #         break
        #     pass
        # # if not self.get_latest_vision_info:
        # #     self.current_state = 320
        # #     current_state = 320
        # #     return current_state
        #
        # current_state = self.current_state
        # self.get_latest_vision_info = False
        #
        # return current_state

    def estimate_pose(self, image):
        image.flags.writeable = False
        image_height, image_width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect and estimate human body pose and hand pose
        pose_results = self.pose_detector.process(image)
        hand_results = self.hand_detector.process(image)

        state_estimation = None
        # return None and flip the camera image when no human is detected
        if not pose_results.pose_landmarks or not hand_results.multi_hand_landmarks:
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.flip(image, 1)
            return state_estimation, image

        # get body pose landmarks
        right_elbow_x = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width
        right_elbow_y = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height
        right_elbow_z = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].z * image_width

        right_wrist_x = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width
        right_wrist_y = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height
        right_wrist_z = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].z * image_width

        right_pinky_x = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_PINKY].x * image_width
        right_pinky_y = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_PINKY].y * image_height
        right_pinky_z = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_PINKY].z * image_width

        right_index_x = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX].x * image_width
        right_index_y = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX].y * image_height
        right_index_z = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX].z * image_width

        right_thumb_x = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].x * image_width
        right_thumb_y = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].y * image_height
        right_thumb_z = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].z * image_width

        pose_estimation = np.array([right_elbow_x, right_elbow_y, right_elbow_z,
                                    right_wrist_x, right_wrist_y, right_wrist_z,
                                    right_pinky_x, right_pinky_y, right_pinky_z,
                                    right_index_x, right_index_y, right_index_z,
                                    right_thumb_x, right_thumb_y, right_thumb_z])

        # get hand pose landmarks
        hand_estimation = None
        for hand_landmarks in hand_results.multi_hand_landmarks:
            wrist_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * image_width
            wrist_y = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * image_height
            wrist_z = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].z * image_width

            thumb_tip_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x * image_width
            thumb_tip_y = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y * image_height
            thumb_tip_z = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].z * image_width

            index_tip_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
            index_tip_y = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
            index_tip_z = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].z * image_width

            middle_tip_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width
            middle_tip_y = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height
            middle_tip_z = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z * image_width

            ring_tip_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width
            ring_tip_y = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height
            ring_tip_z = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].z * image_width

            pinky_tip_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].x * image_width
            pinky_tip_y = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y * image_height
            pinky_tip_z = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].z * image_width

            hand_estimation = np.array([wrist_x, wrist_y, wrist_z,
                                        thumb_tip_x, thumb_tip_y, thumb_tip_z,
                                        index_tip_x, index_tip_y, index_tip_z,
                                        middle_tip_x, middle_tip_y, middle_tip_z,
                                        ring_tip_x, ring_tip_y, ring_tip_z,
                                        pinky_tip_x, pinky_tip_y, pinky_tip_z])

        state_estimation = np.concatenate([pose_estimation, hand_estimation])

        # print("wrist_x: {:0.3f}, wrist_y: {:0.3f}, wrist_z: {:0.3f}".
        #       format(pose_estimation[0],
        #              pose_estimation[1],
        #              pose_estimation[2]))

        # print("elbow_x: {:0.3f}, elbow_y: {:0.3f}, elbow_z: {:0.3f}".
        #       format(pose_estimation[3],
        #              pose_estimation[4],
        #              pose_estimation[5]))

        # print(" ##################################### ")

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.mp_drawing.draw_landmarks(image,
                                       pose_results.pose_landmarks,
                                       self.mp_pose.POSE_CONNECTIONS,
                                       landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

        for hand_landmarks in hand_results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(image,
                                           hand_landmarks,
                                           self.mp_hands.HAND_CONNECTIONS,
                                           self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                           self.mp_drawing_styles.get_default_hand_connections_style())

        image = cv2.flip(image, 1)

        return state_estimation, image

    def select_action(self, state):
        action_values = []
        for a in range(self.num_actions):
            value = self.get_action_value(state, a)
            action_values.append(value)

        chosen_action = self.argmax(action_values)
        return chosen_action

    def take_action(self, action):
        # action 0 represent "bump fist"
        if action == 0:
            self.action_runner.run_waiting_action('say', 'Bump fist')
        # action 1 represent "high five"
        elif action == 1:
            self.action_runner.run_waiting_action('say', 'High five')
        # action 2 represent "hand wave"
        elif action == 2:
            self.action_runner.run_waiting_action('say', 'Hand wave')
        else:
            print("*** Undefined action was selected! Something went wrong ***")

    def take_quiet_action(self, action, correct_action):
        if action == correct_action:
            self.action_runner.run_waiting_action('set_eye_color', 'green')
            sleep(1.0)
            self.action_runner.run_waiting_action('set_eye_color', 'white')
        else:
            self.action_runner.run_waiting_action('set_eye_color', 'red')
            sleep(1.0)
            self.action_runner.run_waiting_action('set_eye_color', 'white')

    def update_sampling_label(self, current_time):
        # every event is in the form of [state, action, action_start_time, action_end_time]
        # event_history is in the form of [e0, e1, e2, ...]
        last_event_id_to_delete = -1
        for i in range(len(self.event_history)):
            e = self.event_history[i]
            action_end_time = e[3]
            # update action value and delete from event history,
            # if the event happened long time ago that any feedback from now on will have no effect on it
            if (current_time - action_end_time) > self.feedback_window_len:
                h = self.assign_reward_label(e)
                action = e[1]
                # experience of each action is in the form of:
                # [[s1, t1_start, t1_end, h1], [s2, t2_start, t2_end, h2], ...]
                self.experiences[action].append([e[0], e[2], e[3], h]) # "UpdateModel" function in the paper
                last_event_id_to_delete = i

        del self.event_history[0:(last_event_id_to_delete + 1)]

        for i in range(len(self.experiences)):
            while len(self.experiences[i]) > self.max_experiences_num:
                del self.experiences[i][0]

        # while len(self.experiences) > self.max_experiences_num:
        #     del self.experiences[0]

    def assign_reward_label(self, event):
        def f_delay(x):
            if x >= -0.8 and x <= -0.2:
                return 1/0.6
            else:
                return 0.0

        def prob_integral(e, h):
            # every human feedback h is in the form of [h_time, h_value]
            return quad(f_delay, e[2] - h[0], e[3] - h[0])[0]

        estimated_h = 0.0
        for h in self.feedback_history:
            estimated_h += h[1] * prob_integral(event, h)

        return estimated_h

    def update_feedback_history(self):
        if len(self.event_history) > 0:
            earliest_event_time = self.event_history[0][2]  # the event start time
            last_feedback_id_to_delete = -1

            for i in range(len(self.feedback_history)):
                feedback = self.feedback_history[i]
                h_time = feedback[0]
                # human feedback become useless if it happened even earlier than the earliest event
                if earliest_event_time - h_time >= 0.0:
                    last_feedback_id_to_delete = i

            # delete outdated feedback from history
            del self.feedback_history[0:(last_feedback_id_to_delete + 1)]

        else:
            while len(self.feedback_history) > self.max_feedback_num:
                del self.feedback_history[0]

    def add_new_feedback(self, feedback_value, feedback_time):
        # every human feedback h is in the form of [h_time, h_value]
        self.feedback_history.append([feedback_time, feedback_value])

    def update_event_history(self, state, action, action_start_time, action_end_time):
        self.event_history.append([state, action, action_start_time, action_end_time])

    # follow algorithm 1 using k-nearest neighbours
    # - "Training a Robot via Human Feedback: A Case Study"
    def get_action_value(self, state, action):
        # experience of each action (i.e., each row of the experiences) is in the form of:
        # [[s1, t1_start, t1_end, h1], [s2, t2_start, t2_end, h2], ...]
        experiences_length = len(self.experiences[action])
        k = math.floor(math.sqrt(experiences_length))

        if k == 0:
            action_value = 0.0
            return action_value
        else:
            preds_sum = 0.0

            experiences = copy.deepcopy(self.experiences[action]) # [[s1, t1_start, t1_end, h1], [s2, t2_start, t2_end, h2], ...]
            distances = []
            # Calculate all the state distances respectively
            # between current state and every experienced state
            for exp in experiences:
                dist = self.state_distance(exp[0], state) # normalize the dist between [0, 1]
                distances.append(dist)

            for i in range(k):
                min_dist = min(distances)
                min_index = distances.index(min_dist)

                h = experiences[min_index][3]
                prediction_i = h * max(1.0 - min_dist/2.0, 1.0/(1.0 + 5 * min_dist))
                preds_sum += prediction_i

                experiences.pop(min_index)
                distances.pop(min_index)

            action_value = preds_sum / k

            return action_value

    def state_distance(self, state_a, state_b):
        a_normalized = (state_a - np.min(state_a))/(np.max(state_a) - np.min(state_a))
        b_normalized = (state_b - np.min(state_b))/(np.max(state_b) - np.min(state_b))

        euclidean_dist = np.linalg.norm(a_normalized - b_normalized)

        return euclidean_dist

    def get_face_position(self, x, y):
        if self.request_for_state:
            self.current_state = x - 640/2

            self.get_latest_vision_info = True
            self.request_for_state = False

    def argmax(self, q_values):
        """argmax with random tie-breaking
            Args:
            q_values (Numpy array): the array of action values
            Returns:
            action (int): an action with the highest value
            """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return np.random.choice(ties)

    def stop(self):
        self.sic.stop()
        self.stopped = True

    def inform_start_new_episode(self, correct_action):
        self.action_runner.run_waiting_action('say', 'New episode start')
        if correct_action == 0:
            self.action_runner.run_waiting_action('say', 'Please do the bump fist')
            self.action_runner.run_waiting_action('say', 'Start')
        elif correct_action == 1:
            self.action_runner.run_waiting_action('say', 'Please do the high five')
            self.action_runner.run_waiting_action('say', 'Start')
        else:
            self.action_runner.run_waiting_action('say', 'Please do the hand wave')
            self.action_runner.run_waiting_action('say', 'Start')

        self.episode_time_out = False
        self.start_new_episode = True

        # wait some time for the episode_control thread to change the flag to start the new episode
        sleep(0.1)

    def inform_end_new_episode(self, total_actions, correct_actions):
        self.action_runner.run_waiting_action('say', 'Episode finishes')
        accuracy = correct_actions * 1.0 / total_actions
        # self.action_runner.run_waiting_action('say', 'Accuracy for last episode is: {}'.format(accuracy))
        sleep(2.0)

    def generate_oracle_feedback(self, selected_action, correct_action, feedback_time):
        if selected_action == correct_action:
            feed_value = 1.0
            self.add_new_feedback(feed_value, feedback_time)
        else:
            feed_value = -1.0
            self.add_new_feedback(feed_value, feedback_time)


def collect_human_feedback(agent):
    while not agent.user_interrupt:
        human_feedback = keyboard.read_hotkey()
        # agent.action_runner.run_action('set_ear_color', 'red')
        # press "a" to give positive feedback
        if human_feedback == "a" or human_feedback == "A":
            current_time = time.time()
            feed_value = 1.0
            agent.add_new_feedback(feed_value, current_time)
        # press "d" to give negative feedback
        elif human_feedback == "d" or human_feedback == "D":
            current_time = time.time()
            feed_value = -1.0
            agent.add_new_feedback(feed_value, current_time)
        elif human_feedback == "p" or human_feedback == "P":
            print("User forced to quit")
            agent.user_interrupt = True
            break
        else:
            pass

        # agent.action_runner.run_action('set_ear_color', 'white')


def reassign_correct_action(action_id):
    if action_id == 2:
        next_action_id = 0
    else:
        next_action_id = action_id + 1

    return next_action_id


def episode_control(agent):
    while not keyboard.is_pressed("q"):
        if agent.start_new_episode:
            sleep(8.0)
            agent.episode_time_out = True
            agent.start_new_episode = False

    agent.episode_time_out = True
    print("[Episode Control Thread]: Thread safely quit")



if __name__ == '__main__':
    agent = Agent('127.0.0.1', camera_mode=0)
    print("Finish construct agent")

    agent.start_connect_and_initialize()
    print("Finish initialize connect")

    # human_interface_thread = threading.Thread(target=collect_human_feedback, args=(agent,))
    # human_interface_thread.start()
    episode_control_thread = threading.Thread(target=episode_control, args=(agent,))
    episode_control_thread.start()

    # follow the algorithm 5 in Brad's Ph.D. dissertation (Section 3.3.6)
    event = 0
    episode = 0
    current_correct_action = 0
    while not keyboard.is_pressed("q"):
        episode_total_actions = 0
        episode_total_correct_actions = 0
        agent.inform_start_new_episode(current_correct_action)
        while not agent.episode_time_out:
            print("Episode [{}] Event [{}]: Starting ... ".format(episode, event))

            action_start_time = time.time()
            agent.update_sampling_label(action_start_time)
            print("Episode [{}] Event [{}]: Finish updating sampling label".format(episode, event))

            agent.update_feedback_history()
            print("Episode [{}] Event [{}]: Finish updating feedback history".format(episode, event))

            current_state, image = agent.get_latest_state()
            print("Episode [{}] Event [{}]: Finish getting current state".format(episode, event))

            # render the state
            cv2.imshow("Cam Image Episode [{}] Event [{}]".format(episode, event), image)
            cv2.waitKey(1)
            print("Episode [{}] Event [{}]: Finish showing current image".format(episode, event))

            if current_state is None:
                agent.action_runner.run_waiting_action('say', 'Fail to detect human pose')
                print("Episode [{}] Event [{}]: Fail to detect human pose. Event finished".format(episode, event))
                print("*********************************************************************************")
                event += 1
                continue

            current_action = agent.select_action(current_state)
            print("Episode [{}] Event [{}]: Finish selecting action".format(episode, event))

            agent.take_action(current_action)
            # agent.take_quiet_action(current_action, current_correct_action)
            print("Episode [{}] Event [{}]: Finish taking action".format(episode, event))

            action_end_time = time.time()

            for i in range(1):
                agent.generate_oracle_feedback(current_action, current_correct_action, action_end_time)

            agent.update_event_history(current_state, current_action, action_start_time, action_end_time)
            print("Episode [{}] Event [{}]: Finish updating event history".format(episode, event))
            print("*********************************************************************************")

            if current_action == current_correct_action:
                episode_total_correct_actions += 1

            event += 1
            episode_total_actions += 1


        agent.inform_end_new_episode(episode_total_actions, episode_total_correct_actions)
        current_correct_action = reassign_correct_action(current_correct_action)
        event = 0
        episode += 1

    print("[Main Thread]: All finished, agent is going to stop")
    agent.stop()