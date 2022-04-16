import copy

import threading
import keyboard

import time
from time import sleep

import numpy as np
import math
from scipy.integrate import quad

from social_interaction_cloud.action import ActionRunner
from social_interaction_cloud.basic_connector import BasicSICConnector


class Agent:
    def __init__(self, server_ip):
        # connector variable
        self.sic = BasicSICConnector(server_ip)
        self.action_runner = None

        # state variable
        self.request_for_state = False
        self.current_state = None

        # action variable
        self.num_actions = 3 # i.e., turn-left, turn-right, stay
        self.get_latest_vision_info = False
        self.set_eye_color = False

        # state history and feedback history:
        # experience of each action (i.e., each row of the experiences) is in the form of:
        # [[s1, t1_start, t1_end, h1], [s2, t2_start, t2_end, h2], ...]
        self.experiences = [[], [], []]
        self.event_history = []
        self.max_experiences_num = 1000
        self.feedback_history = []
        self.max_feedback_num = 20
        self.feedback_window_len = 0.8

        # thread control variable
        self.user_interrupt = False

    def start_connect_and_initialize(self):
        # connect to the SIC server
        self.sic.start()
        self.action_runner = ActionRunner(self.sic)

        # do some initialization jobs
        self.action_runner.run_waiting_action('set_language', 'en-US')
        self.action_runner.run_waiting_action('set_idle')
        self.action_runner.run_action('set_eye_color', 'white')
        self.action_runner.run_vision_listener('people', self.get_face_position, True)

    def get_latest_state(self):
        self.request_for_state = True

        while not self.get_latest_vision_info:
            if self.user_interrupt:
                break
            pass
        # if not self.get_latest_vision_info:
        #     self.current_state = 320
        #     current_state = 320
        #     return current_state

        current_state = self.current_state
        self.get_latest_vision_info = False

        return current_state

    def select_action(self, state):
        action_values = []
        for a in range(self.num_actions):
            value = self.get_action_value(state, a)
            action_values.append(value)

        chosen_action = self.argmax(action_values)
        return chosen_action

    def take_action(self, action):
        # action 0 represent "stay still"
        if action == 0:
            self.action_runner.run_action('set_eye_color', 'red')
            sleep(1.5)
            self.action_runner.run_action('set_eye_color', 'green')
        # action 1 represent "turn left"
        elif action == 1:
            self.action_runner.run_action('set_eye_color', 'red')
            self.action_runner.run_waiting_action('turn', 20)
            self.action_runner.run_action('set_eye_color', 'green')
        # action 2 represent "turn right"
        elif action == 2:
            self.action_runner.run_action('set_eye_color', 'red')
            self.action_runner.run_waiting_action('turn', -20)
            self.action_runner.run_action('set_eye_color', 'green')
        else:
            print("*** Undefined action was selected! Something went wrong ***")

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
                dist = self.state_distance(exp[0], state) / 640.0 # normalize the dist between [0, 1]
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
        euclidean_dist = math.sqrt((state_a - state_b)**2)

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


if __name__ == '__main__':
    agent = Agent('127.0.0.1')
    agent.start_connect_and_initialize()

    human_interface_thread = threading.Thread(target=collect_human_feedback, args=(agent,))
    human_interface_thread.start()

    # follow the algorithm 5 in Brad's Ph.D. dissertation (Section 3.3.6)
    event = 0
    while not agent.user_interrupt:
        print("Event [{}]: Starting ... ".format(event))
        action_start_time = time.time()
        agent.update_sampling_label(action_start_time)
        print("Event [{}]: Finish updating sampling label".format(event))
        agent.update_feedback_history()
        print("Event [{}]: Finish updating feedback history".format(event))

        current_state = agent.get_latest_state()
        print("Event [{}]: Finish getting current state".format(event))
        current_action = agent.select_action(current_state)
        print("Event [{}]: Finish selecting action".format(event))
        agent.take_action(current_action)
        print("Event [{}]: Finish taking action".format(event))

        action_end_time = time.time()

        agent.update_event_history(current_state, current_action, action_start_time, action_end_time)
        print("Event [{}]: Finish updating event history".format(event))
        print("*********************************************************************************")
        event += 1

    agent.stop()