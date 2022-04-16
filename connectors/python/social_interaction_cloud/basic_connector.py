from enum import Enum
from functools import partial
from threading import Condition, Event, Thread
from time import sleep

from social_interaction_cloud.abstract_connector import AbstractSICConnector


class RobotPosture(Enum):
    STAND = 'Stand'
    STANDINIT = 'StandInit'
    STANDZERO = 'StandZero'
    CROUCH = 'Crouch'
    SIT = 'Sit'  # only for Nao
    SITONCHAIR = 'SitOnChair'  # only for Nao
    SITRELAX = 'SitRelax'  # only for Nao
    LYINGBELLY = 'LyingBelly'  # only for Nao
    LYINGBACK = 'LyingBack'  # only for Nao
    UNKNOWN = 'Unknown'  # this is not a valid posture


class BasicSICConnector(AbstractSICConnector):
    """
    Basic implementation of AbstractSICConnector. It serves a connector to the Social Interaction Cloud.
    The base mechanism is that a callback function can be registered for each robot action. When the action returns a
    result (e.g. a ActionDone event) the callback is called once and removed. Only for touch and vision events a
    persistent callback can be registered.

    """

    def __init__(self, server_ip: str, dialogflow_language: str = None,
                 dialogflow_key_file: str = None, dialogflow_agent_id: str = None):
        """
        :param server_ip: IP address of Social Interaction Cloud server
        :param dialogflow_language: the full language key to use in Dialogflow (e.g. en-US)
        :param dialogflow_key_file: path to Google's Dialogflow key file (JSON)
        :param dialogflow_agent_id: ID number of Dialogflow agent to be used (project ID)
        """
        self.__listeners = {}
        self.__conditions = []
        self.__vision_listeners = {}
        self.__touch_listeners = {}
        self.robot_state = {'posture': RobotPosture.UNKNOWN,
                            'is_awake': False,
                            'battery_charge': 100,
                            'is_charging': False,
                            'hot_devices': []}

        super(BasicSICConnector, self).__init__(server_ip=server_ip)

        if dialogflow_language and dialogflow_key_file and dialogflow_agent_id:
            self.enable_service('intent_detection')
            sleep(1)  # give the service some time to load
            self.set_dialogflow_language(dialogflow_language)
            self.set_dialogflow_key(dialogflow_key_file)
            self.set_dialogflow_agent(dialogflow_agent_id)

    ###########################
    # Event handlers          #
    ###########################

    def on_event(self, event: str) -> None:
        self.__notify_listeners(event)
        self.__notify_touch_listeners(event)

    def on_posture_changed(self, posture: str) -> None:
        self.__notify_listeners('onPostureChanged', posture)
        self.robot_state['posture'] = RobotPosture[posture.upper()]

    def on_awake_changed(self, is_awake: bool) -> None:
        self.__notify_listeners('onAwakeChanged', is_awake)
        self.robot_state['is_awake'] = is_awake

    def on_audio_language(self, language_key: str) -> None:
        self.__notify_listeners('onAudioLanguage', language_key)

    def on_audio_intent(self, detection_result: dict) -> None:
        self.__notify_listeners('onAudioIntent', detection_result)

    def on_text_transcript(self, transcript: str) -> None:
        self.__notify_listeners('onTextTranscript', transcript)

    def on_text_sentiment(self, sentiment: str) -> None:
        self.__notify_listeners('onTextSentiment', sentiment)

    def on_new_audio_file(self, audio_file: str) -> None:
        self.__notify_listeners('onNewAudioFile', audio_file)

    def on_new_picture_file(self, picture_file: str) -> None:
        self.__notify_listeners('onNewPictureFile', picture_file)

    def on_person_detected(self, x: int, y: int) -> None:
        self.__notify_vision_listeners('onPersonDetected', x, y)

    def on_face_recognized(self, identifier: str) -> None:
        self.__notify_vision_listeners('onFaceRecognized', identifier)

    def on_emotion_detected(self, emotion: str) -> None:
        self.__notify_vision_listeners('onEmotionDetected', emotion)

    def on_corona_check_passed(self) -> None:
        self.__notify_vision_listeners('onCoronaCheckPassed')

    def on_battery_charge_changed(self, percentage: int) -> None:
        self.__notify_listeners('onBatteryChargeChanged', percentage)
        self.robot_state['battery_charge'] = percentage

    def on_charging_changed(self, is_charging: bool) -> None:
        self.__notify_listeners('onChargingChanged', is_charging)
        self.robot_state['is_charging'] = is_charging

    def on_hot_device_detected(self, hot_devices: list) -> None:
        self.__notify_listeners('onHotDeviceDetected', hot_devices)
        self.robot_state['hot_devices'] = hot_devices

    def on_robot_motion_recording(self, motion: str) -> None:
        self.__notify_listeners('onRobotMotionRecording', motion)

    def on_browser_button(self, button: str) -> None:
        self.__notify_listeners('onBrowserButton', button)

    ###########################
    # Speech Recognition      #
    ###########################

    def speech_recognition(self, context: str, max_duration: int, callback: callable = None) -> None:
        """
        Initiate a speech recognition attempt using Google's Dialogflow using a context.
        For more information on contexts see: https://cloud.google.com/dialogflow/docs/contexts-overview

        The robot will stream audio for at most max_duraction seconds to Dialogflow to recognize something.
        The result (or a 'fail') is returned via the callback function.

        :param context: Google's Dialogflow context label (str)
        :param max_duration: maximum time to listen in seconds (int)
        :param callback: callback function that will be called when a result (or fail) becomes available
        :return:
        """
        enhanced_callback, fail_callback, lock = self.__build_speech_recording_callback(callback)
        self.__register_listener('onAudioIntent', enhanced_callback)
        self.__register_listener('IntentDetectionDone', fail_callback)
        Thread(target=self.__recognizing, args=(context, lock, max_duration)).start()

    def record_audio(self, duration: int, callback: callable = None) -> None:
        """
        Records audio for a number of duration seconds. The location of the audio is returned via the callback function.

        :param duration: number of second of audio that will be recorded.
        :param callback: callback function that will be called when the audio is recorded.
        :return:
        """
        success_callback, _, lock = self.__build_speech_recording_callback(callback)
        self.__register_listener('onNewAudioFile', success_callback)
        Thread(target=self.__recording, args=(lock, duration)).start()

    def __recognizing(self, context: str, lock: Event, max_duration: int) -> None:
        self.stop_listening()
        self.set_dialogflow_context(context)
        self.start_listening(max_duration)
        lock.wait()
        self.__unregister_listener('onAudioIntent')
        self.__unregister_listener('IntentDetectionDone')

    def __recording(self, lock: Event, max_duration: int) -> None:
        self.stop_listening()
        self.set_record_audio(True)
        self.start_listening(max_duration)
        lock.wait()
        self.set_record_audio(False)
        self.__unregister_listener('onNewAudioFile')

    @staticmethod
    def __build_speech_recording_callback(embedded_callback: callable = None):
        lock = Event()

        def success_callback(*args):
            lock.set()
            if embedded_callback:
                embedded_callback(*args)

        def fail_callback():
            if not lock.is_set():
                lock.set()
                if embedded_callback:
                    embedded_callback(None)

        return success_callback, fail_callback, lock

    ###########################
    # Vision                  #
    ###########################

    def take_picture(self, callback: callable = None) -> None:
        """
        Take a picture. Location of the stored picture is returned via callback.

        :param callback:
        :return:
        """
        self.__register_listener('onNewPictureFile', callback)
        super(BasicSICConnector, self).take_picture()

    def start_face_recognition(self, callback: callable = None) -> None:
        """
        Start face recognition. Each time a face is detected, the callback function is called with the recognition result.

        :param callback:
        :return:
        """
        self.__start_vision_recognition('onFaceRecognized', callback)

    def stop_face_recognition(self) -> None:
        """
        Stop face recognition.

        :return:
        """
        self.__stop_vision_recognition('onFaceRecognized')

    def start_people_detection(self, callback: callable = None) -> None:
        """
        Start people detection. Each time a person is detected, the callback function is called.

        :param callback:
        :return:
        """
        self.__start_vision_recognition('onPersonDetected', callback)

    def stop_people_detection(self) -> None:
        """
        Stop people detection.

        :return:
        """
        self.__stop_vision_recognition('onPersonDetected')

    def start_emotion_detection(self, callback: callable = None) -> None:
        """
        Start emotion detection. Each time an emotion becomes available the callback function is called with the emotion.

        :param callback:
        :return:
        """
        self.__start_vision_recognition('onEmotionDetected', callback)

    def stop_emotion_detection(self) -> None:
        """
        Stop emotion detection.

        :return:
        """
        self.__stop_vision_recognition('onEmotionDetected')

    def start_corona_checker(self, callback: callable = None) -> None:
        """
        Start the corona checker. Each time a valid QR code becomes available the callback function is called.

        :param callback:
        :return:
        """
        self.__start_vision_recognition('onCoronaCheckPassed', callback)

    def stop_corona_checker(self) -> None:
        """
        Stop the corona checker.

        :return:
        """
        self.__stop_vision_recognition('onCoronaCheckPassed')

    def __start_vision_recognition(self, event: str, callback: callable = None) -> None:
        if not self.__vision_listeners:
            self.stop_looking()
            self.start_looking(0)
        self.__register_vision_listener(event, callback)

    def __stop_vision_recognition(self, event: str) -> None:
        self.__unregister_vision_listener(event)
        if not self.__vision_listeners:
            self.stop_looking()

    ###########################
    # Touch                   #
    ###########################

    def subscribe_touch_listener(self, touch_event: str, callback: callable) -> None:
        """
        Subscribe a touch listener. The callback function will be called each time the touch_event becomes available.

        :param touch_event:
        :param callback:
        :return:
        """
        self.__touch_listeners[touch_event] = callback

    def unsubscribe_touch_listener(self, touch_event: str) -> None:
        """
        Unsubscribe touch listener.

        :param touch_event:
        :return:
        """
        del self.__touch_listeners[touch_event]

    ###########################
    # Robot actions           #
    ###########################

    def set_language(self, language_key: str, callback: callable = None) -> None:
        if callback:
            self.__register_listener('LanguageChanged', callback)
        super(BasicSICConnector, self).set_language(language_key)

    def set_idle(self, callback: callable = None) -> None:
        if callback:
            self.__register_listener('SetIdle', callback)
        super(BasicSICConnector, self).set_idle()

    def set_non_idle(self, callback: callable = None) -> None:
        if callback:
            self.__register_listener('SetNonIdle', callback)
        super(BasicSICConnector, self).set_non_idle()

    def say(self, text: str, callback: callable = None) -> None:
        if callback:
            self.__register_listener('TextDone', callback)
        super(BasicSICConnector, self).say(text)

    def say_animated(self, text: str, callback: callable = None) -> None:
        if callback:
            self.__register_listener('TextDone', callback)
        super(BasicSICConnector, self).say_animated(text)

    def do_gesture(self, gesture: str, callback: callable = None) -> None:
        if callback:
            self.__register_listener('GestureDone', callback)
        super(BasicSICConnector, self).do_gesture(gesture)

    def play_audio(self, audio_file: str, callback: callable = None) -> None:
        if callback:
            self.__register_listener('PlayAudioDone', callback)
        super(BasicSICConnector, self).play_audio(audio_file)

    def load_audio(self, audio_file: str, callback: callable = None) -> None:
        if callback:
            self.__register_listener('LoadAudioDone', callback)
        super(BasicSICConnector, self).load_audio(audio_file)

    def play_loaded_audio(self, audio_identifier: int, callback: callable = None) -> None:
        if callback:
            self.__register_listener('PlayAudioDone', callback)
        super(BasicSICConnector, self).play_loaded_audio(audio_identifier)

    def clear_loaded_audio(self, callback: callable = None) -> None:
        if callback:
            self.__register_listener('ClearLoadedAudioDone', callback)
        super(BasicSICConnector, self).clear_loaded_audio()

    def set_eye_color(self, color: str, callback: callable = None) -> None:
        if callback:
            self.__register_listener('EyeColourDone', callback)
        super(BasicSICConnector, self).set_eye_color(color)

    def set_ear_color(self, color: str, callback: callable = None) -> None:
        if callback:
            self.__register_listener('EarColourDone', callback)
        super(BasicSICConnector, self).set_ear_color(color)

    def set_head_color(self, color: str, callback: callable = None) -> None:
        if callback:
            self.__register_listener('HeadColourDone', callback)
        super(BasicSICConnector, self).set_head_color(color)

    def set_led_color(self, leds: list, colors: list, duration: int = 0, callback: callable = None) -> None:
        if callback:
            self.__register_listener('LedColorDone', callback)
        super(BasicSICConnector, self).set_led_color(leds, colors, duration)

    def start_led_animation(self, led_group: str, anim_type: str, colors: list, speed: int,
                            callback: callable = None) -> None:
        if callback:
            self.__register_listener('LedAnimationDone', callback)
        super(BasicSICConnector, self).start_led_animation(led_group, anim_type, colors, speed)

    def turn(self, degrees: int, callback: callable = None) -> None:
        if callback:
            self.__register_listener('TurnDone', callback)
        super(BasicSICConnector, self).turn(degrees)

    def wake_up(self, callback: callable = None) -> None:
        if callback:
            self.__register_listener('WakeUpDone', callback)
        super(BasicSICConnector, self).wake_up()

    def rest(self, callback: callable = None) -> None:
        if callback:
            self.__register_listener('RestDone', callback)
        super(BasicSICConnector, self).rest()

    def set_breathing(self, enable: bool, callback: callable = None) -> None:
        if callback:
            self.__register_listener('Breathing' + ('Enabled' if enable else 'Disabled'), callback)
        super(BasicSICConnector, self).set_breathing(enable)

    def go_to_posture(self, posture: Enum, speed: int = 100, callback: callable = None) -> None:
        """
        The robot will try for 3 times to reach a position.
        go_to_posture's callback returns a bool indicating whether the given posture was successfully reached.
        """
        if callback:
            self.__register_listener('GoToPostureDone', partial(self.__posture_callback,
                                                                target_posture=posture,
                                                                embedded_callback=callback))
        super(BasicSICConnector, self).go_to_posture(posture.value, speed)

    def __posture_callback(self, target_posture: str, embedded_callback: callable) -> None:
        if self.robot_state['posture'] == target_posture:  # if posture was successfully reached
            embedded_callback(True)  # call the listener to signal a success
        else:  # if the posture was not reached
            embedded_callback(False)  # call the listener to signal a failure

    def set_stiffness(self, joints: list, stiffness: int, duration: int = 1000, callback: callable = None) -> None:
        if callback:
            self.__register_listener('SetStiffnessDone', callback)
        super(BasicSICConnector, self).set_stiffness(joints, stiffness, duration)

    def play_motion(self, motion: str, callback: callable = None) -> None:
        if callback:
            self.__register_listener('PlayMotionDone', callback)
        super(BasicSICConnector, self).play_motion(motion)

    def start_record_motion(self, joint_chains: list, framerate: int = 5, callback: callable = None) -> None:
        if callback:
            self.__register_listener('RecordMotionStarted', callback)
        super(BasicSICConnector, self).start_record_motion(joint_chains, framerate)

    def stop_record_motion(self, callback: callable = None) -> None:
        if callback:
            self.__register_listener('onRobotMotionRecording', callback)
        super(BasicSICConnector, self).stop_record_motion()

    def browser_show(self, html: str, callback: callable = None) -> None:
        super(BasicSICConnector, self).browser_show(html)

    ###########################
    # Robot action Listeners  #
    ###########################

    def subscribe_condition(self, condition: Condition) -> None:
        """
        Subscribe a threading.Condition object that will be notified each time a registered callback is called.

        :param condition: Condition object that will be notified
        :return:
        """
        self.__conditions.append(condition)

    def unsubscribe_condition(self, condition: Condition) -> None:
        """
        Unsubscribe the threading.Condition object.

        :param condition: Condition object to unsubscribe
        :return:
        """
        if condition in self.__conditions:
            self.__conditions.remove(condition)

    def __notify_conditions(self) -> None:
        for condition in self.__conditions:
            with condition:
                condition.notify()

    def __register_listener(self, event: str, callback: callable) -> None:
        self.__listeners[event] = callback

    def __unregister_listener(self, event: str) -> None:
        del self.__listeners[event]

    def __register_vision_listener(self, event: str, callback: callable) -> None:
        self.__vision_listeners[event] = callback

    def __unregister_vision_listener(self, event: str) -> None:
        del self.__vision_listeners[event]

    def __notify_listeners(self, event: str, *args) -> None:
        if event in self.__listeners:
            listener = self.__listeners[event]
            listener(*args)
            self.__notify_conditions()

    def __notify_vision_listeners(self, event: str, *args) -> None:
        if event in self.__vision_listeners:
            listener = self.__vision_listeners[event]
            listener(*args)
            self.__notify_conditions()

    def __notify_touch_listeners(self, event: str, *args) -> None:
        if event in self.__touch_listeners:
            listener = self.__touch_listeners[event]
            listener(*args)
            self.__notify_conditions()

    ###########################
    # Management              #
    ###########################

    def start(self) -> None:
        self.__clear_listeners()
        super(BasicSICConnector, self).start()

    def stop(self) -> None:
        self.__clear_listeners()
        super(BasicSICConnector, self).stop()

    def __clear_listeners(self) -> None:
        self.__listeners = {}
        self.__conditions = []
        self.__vision_listeners = {}
        self.__touch_listeners = {}
