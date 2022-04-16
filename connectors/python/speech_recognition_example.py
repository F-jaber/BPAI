from enum import Enum
from functools import partial

from social_interaction_cloud.action import ActionRunner
from social_interaction_cloud.basic_connector import BasicSICConnector


class YESNO(Enum):
    """Enum class representing the different answer options to a Yesno question."""
    NO = 0
    YES = 1
    DONTKNOW = 2


class RecognitionManager:
    """Example of a class managing the speech recognition attempts."""

    def __init__(self, max_attempts: int = 2):
        self.max_attempts = max_attempts
        self.attempts = 0
        self.success = False

    def can_recognize(self):
        return not self.success and not self.attempts > self.max_attempts

    def attempt_failed(self):
        self.attempts += 1

    def attempt_succeeded(self):
        self.success = True

    def reset(self):
        self.attempts = 0
        self.success = False


class Example:
    """Example that uses speech recognition. Prerequisites are the availability of a dialogflow_key_file,
    a dialogflow_agent_id, and a running Dialogflow service. For help meeting these Prerequisites see
    https://socialrobotics.atlassian.net/wiki/spaces/CBSR/pages/260276225/The+Social+Interaction+Cloud+Manual

    Two types of questions are included. The first is an entity question where we are interested
    in a specific entity in the answer. In this case the name of person that is interacting with the robot.
    The second is a yesno question. The answer can be yes, no, or don't know (or any synonyms).
    """

    def __init__(self, server_ip: str, dialogflow_key_file: str, dialogflow_agent_id: str):
        self.sic = BasicSICConnector(server_ip, 'en-US', dialogflow_key_file, dialogflow_agent_id)
        self.action_runner = ActionRunner(self.sic)

        self.recognition_manager = RecognitionManager(2)
        self.user_model = {}

    def run(self) -> None:
        # Start the Social Interaction Cloud
        self.sic.start()

        # Make sure the language is set to English and wake up the robot (executed in parallel).
        self.action_runner.load_waiting_action('set_language', 'en-US')
        self.action_runner.load_waiting_action('wake_up')
        self.action_runner.run_loaded_actions()

        # Example of an entity question.
        # As long as no answer has successfully been recognized and the number of attempts did not exceed the maximum:
        # the robot ask the question and starts speech recognition.
        # See on_entity for more info on the partial function.
        while self.recognition_manager.can_recognize():
            self.action_runner.run_waiting_action('say', 'Hi I am Nao. What is your name?')
            self.action_runner.run_waiting_action('speech_recognition', 'answer_name', 3,
                                                  additional_callback=partial(self.on_entity, 'answer_name', 'name'))
        # Don't forget to reset the recognition manager for a next question.
        self.recognition_manager.reset()

        # If the person's name is successfully recognized use it to great them.
        # Else resort to a default greeting.
        if 'name' in self.user_model:
            self.action_runner.run_waiting_action('say', f'Nice to meet you {self.user_model["name"]}')
        else:
            self.action_runner.run_waiting_action('say', 'Nice to meet you')

        # Example of a yesno question.
        while self.recognition_manager.can_recognize():
            self.action_runner.run_waiting_action('say', 'Do you like chocolate?')
            self.action_runner.run_waiting_action('speech_recognition', 'answer_yesno', 3,
                                                  additional_callback=partial(self.on_yesno, 'likes_chocolate'))
        self.recognition_manager.reset()

        if 'likes_chocolate' in self.user_model:
            if self.user_model['likes_chocolate'] == YESNO.YES:
                self.action_runner.run_waiting_action('say', 'Great, I like chocolate too!')
            elif self.user_model['likes_chocolate'] == YESNO.NO:
                self.action_runner.run_waiting_action('say', "Yeah I don't like chocolate either.")
            else:
                self.action_runner.run_waiting_action('say', 'It is a difficult choice, I know.')

        # Let the robot rest while you continue coding and stop the Social Interaction Cloud
        self.action_runner.run_waiting_action('rest')
        self.sic.stop()

    def on_entity(self, intent: str, entity: str, detection_result: dict) -> None:
        """on_entity is a generic callback function for retrieving a recognized entity from the detection result.
        The recognized entity is stored in the user model.

        Note that a callback function for a speech recognition action requires only the detection_result
        as argument. This method has three arguments. Intent and entity should be provided before sending
        the speech recognition action. Hence why a partial function is created. It fills in intent and entity
        and leaves detection_result open to be filled in later.

        More information on creating a partial function:
        https://docs.python.org/3/library/functools.html#functools.partial"""

        if detection_result and 'intent' in detection_result and detection_result['intent'] == intent \
                and 'parameters' in detection_result and entity in detection_result['parameters'] \
                and detection_result['parameters'][entity]:
            self.user_model[entity] = detection_result['parameters'][entity]
            self.recognition_manager.attempt_succeeded()
        else:
            self.recognition_manager.attempt_failed()

    def on_yesno(self, user_model_id: str, detection_result: dict) -> None:
        """on_yesno is a generic callback function for retrieving the answer from a yesno question.
        A yesno question can have three answer: yes, no, or I don't know (or synonyms).
        The answer is stored in the user model using the YESNO enum."""

        if detection_result and 'intent' in detection_result and detection_result['intent']:
            if detection_result['intent'] == 'answer_no':
                self.user_model[user_model_id] = YESNO.NO
            elif detection_result['intent'] == 'answer_yes':
                self.user_model[user_model_id] = YESNO.YES
            elif detection_result['intent'] == 'answer_dontknow':
                self.user_model[user_model_id] = YESNO.DONTKNOW
            self.recognition_manager.attempt_succeeded()
        else:
            self.recognition_manager.attempt_failed()


example = Example('127.0.0.1',
                  '<dialogflow_key_file.json>',
                  '<dialogflow_agent_id>')
example.run()
