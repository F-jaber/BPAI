package org.bitbucket.socialroboticshub.actions.audiovisual;

import java.util.List;

import org.bitbucket.socialroboticshub.actions.RobotAction;

import eis.iilang.Parameter;

abstract class PlayAudioAction extends RobotAction {
	PlayAudioAction(final List<Parameter> parameters) {
		super(parameters);
	}

	@Override
	public String getTopic() {
		return "action_play_audio";
	}

	@Override
	public String getData() {
		return EIStoString(getParameters().get(0));
	}

	@Override
	public String getExpectedEvent() {
		return "PlayAudioStarted";
	}
}
