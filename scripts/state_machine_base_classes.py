#!/usr/bin/env python3
"""Small frame-driven state machine used by the door detector."""

import rospy
import time


class StateMachine:
    """
    Register states, execute the active state, and apply its transitions.

    A state's ``do_action`` method returns another registered state name to
    request a transition, or ``None`` to remain active for the next frame.

    Attributes:
        states:
            Mapping from state names to :class:`Frame_data.BaseState` objects.

        current_state:
            State currently receiving frame updates.

        ctx:
            Most recent shared frame context.

        state_start_time:
            Wall-clock timestamp recorded when the current state was entered.
    """

    def __init__(self, ctx=None):
        """
        Initialize an empty state registry.

        Args:
            ctx:
                Optional initial frame context. It can be replaced on every
                call to :meth:`update`.
        """
        self.states = {}
        self.current_state = None
        self.ctx = ctx
        self.state_start_time = None  # Track when the current state was entered

    def add_state(self, state) -> None:
        """
        Register a state under its ``state.name`` value.

        Args:
            state:
                State object implementing the ``BaseState`` interface.

        Notes:
            Registering another state with the same name replaces the previous
            mapping.
        """
        # Names, rather than class references, keep transition decisions simple
        # for states that only need to return a string.
        self.states[state.name] = state

    def set_state(self, name) -> None:
        """
        Switch to a registered state by name.

        The current state's exit hook runs before the new state's entry hook.
        State residence time is printed for runtime diagnostics.

        Args:
            name:
                Key of the state to activate.

        Raises:
            KeyError:
                If ``name`` has not been registered.
        """
        # Exit and entry hooks run only on a real transition, never once per frame.
        if self.current_state:
            start_exit_action = time.time()
            self.current_state.exit_action(self.ctx)
            exit_action_duration = time.time() - start_exit_action
            elapsed_time = time.time() - self.state_start_time
            #print(f"[INFO]     Exit action took {exit_action_duration:.4f} seconds")
            rospy.logdebug("← Exiting state: %s after %.2f seconds", self.current_state.name, elapsed_time)
        self.current_state = self.states[name]
        
        self.state_start_time = time.time() # Record entry time
        start_entry_action = time.time()
        self.current_state.entry_action(self.ctx)
        entry_action_duration = time.time() - start_entry_action

        rospy.logdebug("→ Entering state: %s", name)
        #print(f"[INFO]     entry_action took {entry_action_duration:.4f} s")


    def update(self, ctx) -> None:
        """
        Run one state-machine step with the newest frame context.

        Args:
            ctx:
                Shared context for the current synchronized frame.

        Notes:
            Unknown transition names are ignored, leaving the current state
            active. If no state has been selected, the method returns without
            invoking any state hook.
        """
        if not self.current_state:
            return
        # keep latest context
        self.ctx = ctx
        start_do_action = time.time()
        # Each update advances at most one state. The new state's work begins
        # with the next synchronized frame, which keeps frame ownership clear.
        next_state_name = self.current_state.do_action(ctx)
        do_action_duration = time.time() - start_do_action
        #print(f"[INFO]     do_action took {do_action_duration:.4f} seconds")
        if next_state_name and next_state_name in self.states:
            self.set_state(next_state_name)
