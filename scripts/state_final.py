#!/usr/bin/env python3
"""Terminal state that holds a completed detection result."""

import rospkg
import rospy
from Frame_data import BaseState, FrameContext

class final_state(BaseState):
    """
    Keep the published result stable until a reset request arrives.

    No new detection is performed in this state. The state machine continues
    publishing the completed result from :class:`FrameContext` until the
    passability/traversal pipeline explicitly starts the next cycle.

    Attributes:
        name:
            Fixed state-machine key ``"final_state"``.
    """

    def __init__(self):
        """
        Initialize the terminal state with its registered transition key.

        Notes:
            The completed detection remains in the shared frame context; this
            object does not copy or clear result fields.
        """
        super().__init__("final_state")

    def do_action(self, ctx: FrameContext):
        """
        Return to idle after the downstream traversal cycle finishes.

        Args:
            ctx:
                Shared context containing the reset flag.

        Returns:
            ``"idle_state"`` when
            ``ctx.go_to_idle_from_finish_state`` is true; otherwise ``None``.
        """
        # The result remains latched until an external controller acknowledges it.
        if ctx.go_to_idle_from_finish_state:
            return "idle_state"

        return  None #stay in final state
