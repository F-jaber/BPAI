import cv2

import time
from time import sleep

from social_interaction_cloud.action import ActionRunner
from social_interaction_cloud.basic_connector import BasicSICConnector

import sys
sys.path.append('/Users/ullrich/ullrich_ws/Socket_Connection/Socket_Connection')
import pepper_connector
from pepper_connector import socket_connection as connect


if __name__ == '__main__':
    server_ip = '127.0.0.1'
    pepper_ip = '10.15.3.171'
    sic = BasicSICConnector(server_ip)
    sic.start()
    action_runner = ActionRunner(sic)
    camera_connector = connect(pepper_ip, 12345, 0, 2)

    event = 0
    max_event = 8
    episode = 0
    while True:
        action_runner.run_waiting_action('say', 'Going to do a new episode')
        action_runner.run_waiting_action('say', 'Start')
        while event < max_event:
            image = camera_connector.get_img()
            cv2.imshow("Cam Image", image)
            cv2.waitKey(1)
            sleep(1.0)
            print("Episode [{}] Event [{}]: Finished".format(episode, event))
            event += 1

        action_runner.run_waiting_action('say', 'Episode finished')
        print("Episode [{}] Finished.".format(episode))
        print("************************************************")
        episode += 1
        event = 0
        sleep(2.0)


