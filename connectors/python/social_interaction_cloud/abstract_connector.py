from enum import Enum
from io import open
from itertools import chain, product
from pathlib import Path
from threading import Event, Thread
from time import strftime
from tkinter import Tk, Checkbutton, Label, Entry, IntVar, StringVar, Button, E, W

from google.protobuf.json_format import MessageToDict
from redis import Redis
from simplejson import dumps

from .detection_result_pb2 import DetectionResult


class AbstractSICConnector(object):
    """
    Abstract class that can be used as a template for a connector to connect with the Social Interaction Cloud.
    """

    def __init__(self, server_ip: str):
        """
        :param server_ip:
        """
        topics = ['events', 'browser_button', 'detected_person', 'recognised_face', 'audio_language', 'audio_intent',
                  'audio_newfile', 'picture_newfile', 'detected_emotion',
                  'robot_audio_loaded'  # , memory_data, gui_data?
                  'robot_posture_changed', 'robot_awake_changed', 'robot_battery_charge_changed',
                  'robot_charging_changed', 'robot_hot_device_detected', 'robot_motion_recording',
                  'text_transcript', 'text_sentiment', 'corona_check']
        device_types = {
            1: ['cam', 'Camera'],
            2: ['mic', 'Microphone'],
            3: ['robot', 'Robot'],
            4: ['speaker', 'Speaker'],
            5: ['browser', 'Browser']
            # puppet, gui_controller, logger, ga?
        }
        self.device_types = Enum(
            value='DeviceType',
            names=chain.from_iterable(
                product(v, [k]) for k, v in device_types.items()
            )
        )
        topic_map = {
            self.device_types['cam']: ['action_video', 'action_take_picture'],
            self.device_types['mic']: ['action_audio', 'dialogflow_language', 'dialogflow_context', 'dialogflow_key',
                                       'dialogflow_agent', 'dialogflow_record'],
            self.device_types['robot']: ['action_gesture', 'action_eyecolour', 'action_earcolour', 'action_headcolour',
                                         'action_idle', 'action_turn', 'action_wakeup', 'action_rest',
                                         'action_set_breathing', 'action_posture', 'action_stiffness',
                                         'action_play_motion', 'action_record_motion',  # 'action_motion_file',
                                         'action_led_color', 'action_led_animation'],
            self.device_types['speaker']: ['audio_language', 'action_say', 'action_say_animated', 'action_play_audio',
                                           'action_stop_talking', 'action_load_audio', 'action_clear_loaded_audio'],
            self.device_types['browser']: ['render_html']
        }
        self.__topic_map = {}
        for k, v in topic_map.items():
            for x in v:
                self.__topic_map[x] = k
        self.devices = {}
        for device_type in self.device_types:
            self.devices[device_type] = []

        self.time_format = '%H-%M-%S'

        if server_ip.startswith('127.') or server_ip.startswith('192.') or server_ip == 'localhost':
            self.username = 'default'
            self.password = 'changemeplease'
            self.redis = Redis(host=server_ip, username=self.username, password=self.password, ssl=True,
                               ssl_ca_certs='cert.pem')
        else:
            self.__dialog1 = Tk()
            self.username = StringVar()
            self.password = StringVar()
            self.provide_user_information()
            self.redis = Redis(host=server_ip, username=self.username, password=self.password, ssl=True)
        self.__dialog2 = Tk()
        self.__checkboxes = {}
        self.select_devices()
        all_topics = []
        for device_list in self.devices.values():
            for device in device_list:
                for topic in topics:
                    all_topics.append(device + '_' + topic)
        self.__pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
        self.__pubsub.subscribe(**dict.fromkeys(all_topics, self.__listen))
        self.__pubsub_thread = self.__pubsub.run_in_thread(sleep_time=0.001)

        self.__running_thread = Thread(target=self.__run)
        self.__stop_event = Event()

        self.__running = False

    def provide_user_information(self) -> None:
        Label(self.__dialog1, text='Username:').grid(row=1, column=1, sticky=E)
        Entry(self.__dialog1, width=15, textvariable=self.username).grid(row=1, column=2, sticky=W)
        Label(self.__dialog1, text='Password:').grid(row=2, column=1, sticky=E)
        Entry(self.__dialog1, width=15, show='*', textvariable=self.password).grid(row=2, column=2, sticky=W)
        Button(self.__dialog1, text='OK', command=self.__provide_user_information_done).grid(row=3, column=1,
                                                                                             columnspan=2)
        self.__dialog1.bind('<Return>', (lambda event: self.__provide_user_information_done()))
        self.__dialog1.mainloop()

    def __provide_user_information_done(self):
        self.__dialog1.destroy()
        self.username = self.username.get()
        self.password = self.password.get()

    def select_devices(self) -> None:
        time = self.redis.time()
        devices = self.redis.zrevrangebyscore(name='user:' + self.username, min=(time[0] - 60), max='+inf')
        devices.sort()
        row = 1
        for device in devices:
            var = IntVar()
            self.__checkboxes[device] = var
            Checkbutton(self.__dialog2, text=device, variable=var).grid(row=row, column=1, sticky=W)
            Label(self.__dialog2, text='').grid(row=row, column=2, sticky=E)
            row += 1
        Button(self.__dialog2, text='(De)Select All', command=self.__select_devices_toggle).grid(row=row, column=1,
                                                                                                 sticky=W)
        Button(self.__dialog2, text='OK', command=self.__select_devices_done).grid(row=row, column=2, sticky=E)
        self.__dialog2.mainloop()

    def __select_devices_toggle(self):
        none_selected = True
        for var in self.__checkboxes.values():
            if var.get() == 1:
                none_selected = False
                break
        for var in self.__checkboxes.values():
            var.set(1 if none_selected else 0)

    def __select_devices_done(self):
        self.__dialog2.destroy()
        for name, var in self.__checkboxes.items():
            if var.get() == 1:
                split = name.decode('utf-8').split(':')
                device_type = self.device_types[split[1]]
                self.devices[device_type].append(self.username + '-' + split[0])

    ###########################
    # Event handlers          #
    ###########################

    def on_event(self, event: str) -> None:
        """Triggered upon any event This can be either an event related to some action called here,
        or an event related to one of the robot's touch sensors, i.e. one of:
        RightBumperPressed, RightBumperReleased, LeftBumperPressed, LeftBumperReleased, BackBumperPressed,
        BackBumperReleased, FrontTactilTouched, FrontTactilReleased, MiddleTactilTouched, MiddleTactilReleased,
        RearTactilTouched, RearTactilReleased, HandRightBackTouched, HandRightBackReleased, HandRightLeftTouched,
        HandRightLeftReleased, HandRightRightTouched, HandRightRightReleased, HandLeftBackTouched, HandLeftBackReleased,
        HandLeftLeftTouched, HandLeftLeftReleased, HandLeftRightTouched, or HandLeftRightReleased
        See: http://doc.aldebaran.com/2-8/family/nao_technical/contact-sensors_naov6.html"""
        pass

    def on_posture_changed(self, posture: str) -> None:
        """
        Trigger when the posture has changed.
        :param posture: new posture.
        :return:
        """
        pass

    def on_awake_changed(self, is_awake: bool) -> None:
        """
        Trigger when the robot wakes up or goes to rest.
        :param is_awake: true if the robot just woke up, false if it just went to rest.
        :return:
        """
        pass

    def on_person_detected(self, x: int, y: int) -> None:
        """Triggered when some person was detected in front of the robot (after a startWatching action was called).
        The (given x,y) coordinates are of the center of the person's face.
        This is only sent when the people detection service is running. Will be sent as long as a person is detected."""
        pass

    def on_face_recognized(self, identifier: str) -> None:
        """Triggered when a specific face was detected in front of the robot (after a startWatching action was called).
        Only sent when the face recognition service is running. Will be sent as long as the face is recognised.
        The identifiers of recognised faces are stored in a file, and will thus persist over a restart of the agent."""
        pass

    def on_audio_language(self, language_key: str) -> None:
        """Triggered whenever a language change was requested (for example by the user).
        Given is the full language key (e.g. nl-NL or en-US)."""
        pass

    def on_audio_loaded(self, identifier: int) -> None:
        """Gives the unique identifier for the audio that was just loaded (see load_audio)"""
        pass

    def on_audio_intent(self, detection_result: dict) -> None:
        """Triggered whenever an intent was detected (by Dialogflow) on a user's speech.
        Given is the name of the intent, a list of optional parameters (following from the dialogflow spec), and a confidence value.
        See https://cloud.google.com/dialogflow/docs/intents-loaded_actions-parameters.
        The recognized text itself is provided as well, even when no intent was actually matched (i.e. a failure).
        These are sent as soon as an intent is recognised, which is always after some start_listening action,
        but might come in some time after the final stop_listening action (if there was some intent detected at least).
        Intents will keep being recognised until stop_listening is called. In that case, this function can still be triggered,
        containing the recognized text but no intent (and a confidence value of 0)."""
        pass

    def on_text_transcript(self, transcript: str) -> None:
        """Triggered directly when any text is recognised by Dialogflow.
        This can give many non-final results, but is useful for matching short texts (like yes/no) directly."""
        pass

    def on_text_sentiment(self, sentiment: str) -> None:
        """Based on all text transcripts (see on_text_transcript), gives a positive or negative sentiment
        for all these texts (when the sentiment_analysis service is running)."""
        pass

    def on_new_audio_file(self, audio_file: str) -> None:
        """Triggered whenever a new recording has been stored to an audio (WAV) file. See set_record_audio.
        Given is the name to the recorded file (which is in the folder required by the play_audio function).
        All audio received between the last start_listening and stop_listening calls is recorded."""
        pass

    def on_new_picture_file(self, picture_file: str) -> None:
        """Triggered whenever a new picture has been stored to an image (JPG) file. See take_picture.
        Given is the path to the taken picture."""
        pass

    def on_emotion_detected(self, emotion: str) -> None:
        """Triggered whenever an emotion has been detected by the emotion detection service (when running)."""
        pass

    def on_battery_charge_changed(self, percentage: int) -> None:
        """
        Triggered when the battery level changes.
        :param percentage: battery level (0-100)
        :return:
        """
        pass

    def on_charging_changed(self, is_charging: bool) -> None:
        """
        Triggered when the robot is connected (True) or disconnected (False) from a power source.
        Warning: is not always accurate, see:
        http://doc.aldebaran.com/2-8/naoqi/sensors/albattery-api.html#BatteryPowerPluggedChanged
        :param is_charging:
        :return:
        """
        pass

    def on_hot_device_detected(self, hot_devices: list) -> None:
        """
        Triggered when one or more body parts of the robot become too hot.
        :param hot_devices: list of body parts that are too hot.
        :return:
        """
        pass

    def on_robot_motion_recording(self, motion: str) -> None:
        """
        Triggered when a motion recording (JSON format) becomes available .

        :param motion:
        :return:
        """
        pass

    def on_browser_button(self, button: str) -> None:
        """
        Triggered when a button has been pressed in the browser

        :param button:
        :return:
        """
        pass

    def on_corona_check_passed(self) -> None:
        """Triggered when a valid Corona QR code has been detected by the corona_checker service"""
        pass

    ###########################
    # Dialogflow Actions      #
    ###########################

    def set_dialogflow_key(self, key_file: str) -> None:
        """Required for setting up Dialogflow: the path to the (JSON) keyfile."""
        contents = Path(key_file).read_text()
        self.__send('dialogflow_key', contents)

    def set_dialogflow_agent(self, agent_name: str) -> None:
        """Required for setting up Dialogflow: the name of the agent to use (i.e. the project id)."""
        self.__send('dialogflow_agent', agent_name)

    def set_dialogflow_language(self, language_key: str) -> None:
        """Required for setting up Dialogflow: the full key of the language to use (e.g. nl-NL or en-US)."""
        self.__send('dialogflow_language', language_key)

    def set_dialogflow_context(self, context: str) -> None:
        """Indicate the Dialogflow context to use for the next speech-to-text (or to intent)."""
        self.__send('dialogflow_context', context)

    def start_listening(self, seconds: int) -> None:
        """Tell the robot (and Dialogflow) to start listening to audio (and potentially recording it).
        Intents will be continuously recognised. If seconds>0, it will automatically stop listening.
        A ListeningStarted event will be sent once the feed starts, and ListeningDone once it ends."""
        self.__send('action_audio', str(seconds))

    def stop_listening(self) -> None:
        """Tell the robot (and Dialogflow) to stop listening to audio.
        Note that a potentially recognized intent might come in up to a second after this call."""
        self.__send('action_audio', '-1')

    ###########################
    # Robot Actions           #
    ###########################

    def set_language(self, language_key: str) -> None:
        """For changing the robot's speaking language: the full key of the language to use
        (e.g. nl-NL or en-US). A LanguageChanged event will be sent when the change has propagated."""
        self.__send('audio_language', language_key)

    def set_record_audio(self, should_record: bool) -> None:
        """Indicate if audio should be recorded (see on_new_audio_file)."""
        self.__send('dialogflow_record', '1' if should_record else '0')

    def set_idle(self) -> None:
        """Put the robot into 'idle mode': always looking straight ahead.
        A SetIdle event will be sent when the robot has transitioned into the idle mode."""
        self.__send('action_idle', 'true')

    def set_non_idle(self) -> None:
        """Put the robot back into its default 'autonomous mode' (looking towards sounds).
        A SetNonIdle event will be sent when the robot has transitioned out of the idle mode."""
        self.__send('action_idle', 'false')

    def start_looking(self, seconds: int) -> None:
        """Tell the robot (and any recognition module) to start the camera feed.
        If seconds>0, it will automatically stop looking.
        A WatchingStarted event will be sent once the feed starts, and WatchingDone once it ends."""
        self.__send('action_video', str(seconds))

    def stop_looking(self) -> None:
        """Tell the robot (and any recognition module) to stop looking."""
        self.__send('action_video', '-1')

    def say(self, text: str) -> None:
        """A string that the robot should say (in the currently selected language!).
        A TextStarted event will be sent when the speaking starts and a TextDone event after it is finished."""
        self.__send('action_say', text)

    def say_animated(self, text: str) -> None:
        """A string that the robot should say (in the currently selected language!) in an animated fashion.
        This means that the robot will automatically try to add (small) animations to the text.
        Moreover, in this function, special tags are supported, see:
        http://doc.aldebaran.com/2-8/naoqi/audio/altexttospeech-tuto.html#using-tags-for-voice-tuning
        A TextStarted event will be sent when the speaking starts and a TextDone event after it is finished."""
        self.__send('action_say_animated', text)

    def stop_talking(self) -> None:
        """Cancel any currently running say(_animated)"""
        self.__send('action_stop_talking', '')

    def do_gesture(self, gesture: str) -> None:
        """Make the robot perform the given gesture. The list of available gestures (not tags!) is available on:
        http://doc.aldebaran.com/2-8/naoqi/motion/alanimationplayer-advanced.html (Nao)
        http://doc.aldebaran.com/2-5/naoqi/motion/alanimationplayer-advanced.html (Pepper)
        You can also install custom animations with Choregraphe.
        A GestureStarted event will be sent when the gesture starts and a GestureDone event when it is finished."""
        self.__send('action_gesture', gesture)

    def load_audio(self, audio_file: str) -> None:
        """Preloads the given audio file on the robot. See on_audio_loaded and play_loaded_audio.
        A LoadAudioStarted event will be sent when the audio starts loading and a LoadAudioDone event when it is finished."""
        with open(audio_file, 'rb') as file:
            self.__send('action_load_audio', file.read())

    def play_audio(self, audio_file: str) -> None:
        """Plays the given audio file on the robot's speakers.
        A PlayAudioStarted event will be sent when the audio starts and a PlayAudioDone event after it is finished.
        Any previously playing audio will be cancelled first."""
        with open(audio_file, 'rb') as file:
            self.__send('action_play_audio', file.read())

    def play_loaded_audio(self, identifier: int) -> None:
        """Plays the given preloaded audio file on the robot's speakers. See load_audio and on_audio_loaded.
        A PlayAudioStarted event will be sent when the audio starts and a PlayAudioDone event after it is finished.
        Any previously playing audio will be cancelled first."""
        self.__send('action_play_audio', identifier)

    def clear_loaded_audio(self) -> None:
        """Clears all preloaded audio (see load_audio) from the robot.
        A ClearLoadedAudioStarted event will be sent when the audio starts clearing,
        and a ClearLoadedAudioDone event after it is all cleared up."""
        self.__send('action_clear_loaded_audio', '')

    def set_eye_color(self, color: str) -> None:
        """Sets the robot's eye LEDs to one of the following colours:
        white, red, green, blue, yellow, magenta, cyan, greenyellow or rainbow.
        An EyeColourStarted event will be sent when the change starts and a EyeColourDone event after it is done."""
        self.__send('action_eyecolour', color)

    def set_ear_color(self, color: str) -> None:
        """Sets the robot's ear LEDs to one of the following colours:
        white, red, green, blue, yellow, magenta, cyan, greenyellow or rainbow.
        An EarColourStarted event will be sent when the change starts and a EarColourDone event after it is done."""
        self.__send('action_earcolour', color)

    def set_head_color(self, color: str) -> None:
        """Sets the robot's head LEDs to one of the following colours:
        white, red, green, blue, yellow, magenta, cyan, greenyellow or rainbow.
        A HeadColourStarted event will be sent when the change starts and a HeadColourDone event after it is done."""
        self.__send('action_headcolour', color)

    def set_led_color(self, leds: list, colors: list, duration: int = 0) -> None:
        """A list of LEDs (see http://doc.aldebaran.com/2-5/naoqi/sensors/alleds.html#list-group-led),
        and a corresponding list of colors to give to the LEDs. Optionally a duration for the transitions (default=instant).
        A LedColorStarted event will be sent when the color change starts and a LedColorDone event after it is done."""
        self.__send('action_led_color', dumps(leds) + ';' + dumps(colors) + ';' + str(duration))

    def start_led_animation(self, led_group: str, anim_type: str, colors: list, speed: int) -> None:
        """A LED group name (eyes, chest, feet, all), an animation type (rotate, blink, alternate),
        a corresponding list of colors, and a speed setting (milliseconds).
        A LedAnimationStarted event will be sent when the animation starts and a LedAnimationDone event after it is done."""
        self.__send('action_led_animation', 'start;' + led_group + ';' + anim_type + ';' + dumps(colors) + ';' + str(speed))

    def stop_led_animation(self) -> None:
        """Abort any currently running LED animation (see start_led_animation)."""
        self.__send('action_led_animation', 'stop')

    def take_picture(self) -> None:
        """Instructs the robot to take a picture. See on_new_picture_file.
        The people detection or face recognition service must be running for this action to work."""
        self.__send('action_take_picture', '')

    def turn(self, degrees: int) -> None:
        """Instructs the Pepper robot to make a turn of the given amount of degrees (-360 to 360).
        A TurnStarted event will be sent when the robot starts turning and a TurnDone event after it is done."""
        self.__send('action_turn', str(degrees))

    def wake_up(self) -> None:
        """Instructs the robot to execute the default wake_up behavior. Also see on_on_awake_changed.
        See: http://doc.aldebaran.com/2-8/naoqi/motion/control-stiffness-api.html?highlight=wakeup#ALMotionProxy::wakeUp
        A WakeUpStarted event will be sent when the robot starts waking up and a WakeUpDone event after it is done."""
        self.__send('action_wakeup', '')

    def rest(self) -> None:
        """Instructs the robot to execute the default wake_up behavior. Also see on_on_awake_changed.
        See: http://doc.aldebaran.com/2-8/naoqi/motion/control-stiffness-api.html?highlight=wakeup#ALMotionProxy::rest
        A RestStarted event will be sent when the robot starts going into rest mode and a RestDone event after it is done."""
        self.__send('action_rest', '')

    def set_breathing(self, enable: bool) -> None:
        """
        Enable/disable the default breathing animation of the robot.
        See: http://doc.aldebaran.com/2-8/naoqi/motion/idle-api.html?highlight=breathing#ALMotionProxy::setBreathEnabled__ssCR.bCR
        A BreathingEnabled or BreathingDisabled event will be sent when the change is done (depending on the given input).
        """
        self.__send('action_set_breathing', 'Body;' + '1' if enable else '0')

    def go_to_posture(self, posture: str, speed: int = 100) -> None:
        """
        Let the robot go to a predefined posture. Also see on_posture_changed.

        Predefined postures for Pepper are: Stand or StandInit, StandZero, and Crouch.
        See: http://doc.aldebaran.com/2-5/family/pepper_technical/postures_pep.html#pepper-postures

        Predefined postures for Nao are: Stand, StandInit, StandZero, Crouch, Sit, SitRelax, LyingBelly, and LyingBack.
        See: http://doc.aldebaran.com/2-8/family/nao_technical/postures_naov6.html#naov6-postures

        A GoToPostureStarted event will be sent when the posture change starts and GoToPostureDone when it finished.

        :param posture: target posture
        :param speed: optional speed parameter to set the speed of the posture change. Default is 1.0 (100% speed).
        :return:
        """
        self.__send('action_posture', posture + ';' + str(speed) if 1 <= speed <= 100 else posture + ';100')

    def set_stiffness(self, chains: list, stiffness: int, duration: int = 1000) -> None:
        """
        Set the stiffness for one or more joint chains.
        Suitable joint chains for Nao are: Head, RArm, LArm, RLeg, LLeg
        Suitable joint chains for Pepper are: Head, RArm, LArm, Leg, Wheels

        A SetStiffnessStarted event will be sent when the stiffness change starts and SetStiffnessDone when it finished.

        :param chains: list of joints.
        :param stiffness: stiffness value between 0 and 100.
        :param duration: stiffness transition time in milliseconds.
        :return:
        """
        self.__send('action_stiffness', dumps(chains) + ';' + str(stiffness) + ';' + str(duration))

    def play_motion(self, motion: str) -> None:
        """
        Play a motion.

        Suitable joints and angles for Nao:
        https://developer.softbankrobotics.com/nao6/nao-documentation/nao-developer-guide/kinematics-data/joints
        Suitable joints and angles for Pepper:
        https://developer.softbankrobotics.com/pepper-naoqi-25/pepper-documentation/pepper-developer-guide/kinematics-data/joints

        A PlayMotionStarted event will be sent when the motion sequence starts and PlayMotionDone when it finished.

        :param motion: json string in the following format
        {'robot': 'nao/pepper', 'compress_factor_angles': int, 'compress_factor_times: int
        'motion': {'joint1': { 'angles': [...], 'times': [...]}, 'joint2': {...}}}
        :return:
        """
        self.__send('action_play_motion', motion)

    def start_record_motion(self, joint_chains: list, framerate: int = 5) -> None:
        """
        Start recording of the angles over time of a given list of joints and or joint chains with an optional framerate.

        Suitable joints and joint chains for nao:
        http://doc.aldebaran.com/2-8/family/nao_technical/bodyparts_naov6.html#nao-chains
        Suitable joints and joint chains for pepper:
        http://doc.aldebaran.com/2-8/family/pepper_technical/bodyparts_pep.html

        A RecordMotionStarted event will be sent when the recording starts and RecordMotionDone when it finished.

        :param joint_chains: a list with one or more joints or joint chains
        :param framerate: optional number of recordings per second. The default is 5 fps.
        :return:
        """
        self.__send('action_record_motion', 'start;' + dumps(joint_chains) + ';' + str(framerate))

    def stop_record_motion(self) -> None:
        """
        Stop recording of an active motion recording (started by start_record_motion)

        :return:
        """
        self.__send('action_record_motion', 'stop')

    ###########################
    # Browser Actions         #
    ###########################

    def browser_show(self, html: str) -> None:
        """
        Show the given HTML body on the currently connected browser page.
        :param html: the HTML contents (put inside a <body>).
        By default, the Bootstrap rendering library is loaded: https://getbootstrap.com/docs/4.6/
        Moreover, various classes can be used (on e.g. divs) to automatically create dynamic elements:
        - listening_icon: shows a microphone that is enabled or disabled when the robot is listening or not.
        - speech_text: shows a live-stream of the currently recognized text (by e.g. dialogflow).
        - vu_logo: renders a VU logo.
        - english_flag: renders a English flag (changes the audio language when tapped on).
        - chatbox: allows text input (to e.g. dialogflow).
        Finally, each button element will automatically trigger an event when clicked (see on_browser_button).
        :return:
        """
        self.__send('render_html', html)

    ###########################
    # Management              #
    ###########################

    def enable_service(self, name: str) -> None:
        """
        Enable the given service (for the previously selected devices)
        :param name: people_detection, face_recognition, emotion_detection, corona_checker, intent_detection or sentiment_analysis
        :return:
        """
        pipe = self.redis.pipeline()
        if name == 'people_detection' or name == 'face_recognition' or name == 'emotion_detection' or name == 'corona_checker':
            for cam in self.devices[self.device_types['cam']]:
                pipe.publish(name, cam)
        elif name == 'intent_detection' or name == 'sentiment_analysis':
            for mic in self.devices[self.device_types['mic']]:
                pipe.publish(name, mic)
        else:
            print('Unknown service: ' + name)
        pipe.execute()

    def start(self) -> None:
        """Start the application"""
        self.__running = True
        self.__running_thread.start()

    def stop(self) -> None:
        """Stop listening to incoming events (which is done in a thread) so the Python application can close."""
        self.__running = False
        self.__stop_event.set()
        print('Trying to exit gracefully...')
        try:
            self.__pubsub_thread.stop()
            self.redis.close()
            print('Graceful exit was successful.')
        except Exception as err:
            print('Graceful exit has failed: ' + err.message)

    def __run(self) -> None:
        while self.__running:
            self.__stop_event.wait()

    def __listen(self, message) -> None:
        raw_channel = message['channel'].decode('utf-8')
        split = raw_channel.index('_') + 1
        channel = raw_channel[split:]
        data = message['data']

        if channel == 'events':
            self.on_event(event=data.decode('utf-8'))
        elif channel == 'browser_button':
            self.on_browser_button(button=data.decode('utf-8'))
        elif channel == 'detected_person':
            coordinates = data.decode('utf-8').split(',')
            self.on_person_detected(int(coordinates[0]), int(coordinates[1]))
        elif channel == 'recognised_face':
            self.on_face_recognized(identifier=data.decode('utf-8'))
        elif channel == 'audio_language':
            self.on_audio_language(language_key=data.decode('utf-8'))
        elif channel == 'audio_intent':
            detection_result = DetectionResult()
            detection_result.ParseFromString(data)
            self.on_audio_intent(detection_result=MessageToDict(detection_result))
        elif channel == 'audio_newfile':
            audio_file = strftime(self.time_format) + '.wav'
            with open(audio_file, 'wb') as wav:
                wav.write(data)
            self.on_new_audio_file(audio_file=audio_file)
        elif channel == 'picture_newfile':
            picture_file = strftime(self.time_format) + '.jpg'
            with open(picture_file, 'wb') as jpg:
                jpg.write(data)
            self.on_new_picture_file(picture_file=picture_file)
        elif channel == 'detected_emotion':
            self.on_emotion_detected(emotion=data.decode('utf-8'))
        elif channel == 'robot_posture_changed':
            self.on_posture_changed(posture=data.decode('utf-8'))
        elif channel == 'robot_awake_changed':
            self.on_awake_changed(is_awake=(data.decode('utf-8') == '1'))
        elif channel == 'robot_battery_charge_changed':
            self.on_battery_charge_changed(percentage=int(data.decode('utf-8')))
        elif channel == 'robot_charging_changed':
            self.on_charging_changed(is_charging=(data.decode('utf-8') == '1'))
        elif channel == 'robot_hot_device_detected':
            self.on_hot_device_detected(hot_devices=data.decode('utf-8').split(';'))
        elif channel == 'robot_motion_recording':
            self.on_robot_motion_recording(motion=data.decode('utf-8'))
        elif channel == 'text_transcript':
            self.on_text_transcript(transcript=data.decode('utf-8'))
        elif channel == 'text_sentiment':
            self.on_text_sentiment(sentiment=data.decode('utf-8'))
        elif channel == 'corona_check':
            self.on_corona_check_passed()
        else:
            print('Unknown channel: ' + channel)

    def __send(self, channel: str, data) -> None:
        pipe = self.redis.pipeline()
        target_type = self.__topic_map[channel]
        for device in self.devices[target_type]:
            pipe.publish(device + '_' + channel, data)
        pipe.execute()
