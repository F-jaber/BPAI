package org.bitbucket.socialroboticshub.actions.animation;

import java.util.List;

import org.bitbucket.socialroboticshub.actions.RobotAction;

import eis.iilang.Numeral;
import eis.iilang.Parameter;

public class TurnAction extends RobotAction {
	public final static String NAME = "turn";

	/**
	 * @param parameters A list of 1 integer (-360 to 360 degrees)
	 */
	public TurnAction(final List<Parameter> parameters) {
		super(parameters);
	}

	@Override
	public boolean isValid() {
		return getParameters().size() == 1 && getParameters().get(0) instanceof Numeral;
	}

	@Override
	public String getTopic() {
		return "action_turn";
	}

	@Override
	public String getData() {
		return EIStoString(getParameters().get(0));
	}

	@Override
	public String getExpectedEvent() {
		return "TurnStarted";
	}
}
