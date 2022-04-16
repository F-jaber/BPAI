from time import sleep

from social_interaction_cloud.action import ActionRunner
from social_interaction_cloud.basic_connector import BasicSICConnector


class Example:
    """ Example of recording an replaying a motion. A 'motion timeline' is created (JSON as string).
    In the example this timeline is stored temporarily in a variable, but it can easily be written to a file
    to be loaded in again later for reuse.

    Important to know is that to move the robot's joints the stiffness of these joints need to be first set to 0.
    For the robot to move those arms later (e.g. to play a motion timeline or for animated speech) the stiffness
    need to be set back to 100.

    The labels for the joints are the default Naoqi labels that can be found here:
    http://doc.aldebaran.com/2-8/family/nao_technical/bodyparts_naov6.html
    """

    def __init__(self, server_ip: str):
        self.joints = ['LArm', 'RArm']
        self.recorded_motion = None
        self.sic = BasicSICConnector(server_ip)
        self.action_runner = ActionRunner(self.sic)

    def run(self) -> None:
        self.sic.start()

        self.action_runner.load_waiting_action('set_language', 'en-US')
        self.action_runner.load_waiting_action('wake_up')
        self.action_runner.run_loaded_actions()

        self.action_runner.load_waiting_action('set_stiffness', self.joints, 0)
        self.action_runner.load_waiting_action('say', 'You can hold my arms now.')
        self.action_runner.run_loaded_actions()

        self.action_runner.run_waiting_action('say', 'You can move my arms around now.')
        self.action_runner.run_action('start_record_motion', self.joints)

        sleep(3)

        self.action_runner.run_waiting_action('say', 'We are done in 3, 2, 1.')
        self.action_runner.load_waiting_action('stop_record_motion', additional_callback=self.motion_recording_callback)
        self.action_runner.load_waiting_action('say', 'and done.')
        self.action_runner.run_loaded_actions()

        self.action_runner.run_waiting_action('set_stiffness', self.joints, 100)

        if self.recorded_motion:
            self.action_runner.run_waiting_action('say', 'Amazing. Let me replay that')
            self.action_runner.run_waiting_action('play_motion', self.recorded_motion)
        else:
            self.action_runner.run_waiting_action('say', 'Something went wrong during recording.')

        self.action_runner.run_waiting_action('rest')
        self.sic.stop()

    def motion_recording_callback(self, motion: str) -> None:
        self.recorded_motion = motion


example = Example('127.0.0.1')
example.run()