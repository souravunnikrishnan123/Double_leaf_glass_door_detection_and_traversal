#!/usr/bin/env python3
"""Small frame-driven state machine used by the door detector."""

import rospy
import time


class StateMachine:
    """Register states, execute the active state, and apply its transitions.

    A state's ``do_action`` method returns another registered state name to
    request a transition, or ``None`` to remain active for the next frame.
    """

    def __init__(self, ctx=None):
        self.states = {}
        self.current_state = None
        self.ctx = ctx
        self.state_start_time = None  # Track when the current state was entered

    def add_state(self, state) -> None:
        """Register a state under its ``state.name`` value."""
        self.states[state.name] = state

    def set_state(self, name) -> None:
        """Switch to a new state by name."""
        if self.current_state:
            start_exit_action = time.time()
            self.current_state.exit_action(self.ctx)
            exit_action_duration = time.time() - start_exit_action
            elapsed_time = time.time() - self.state_start_time
            #print(f"[INFO]     Exit action took {exit_action_duration:.4f} seconds")
            print(f"[INFO] ← Exiting state: {self.current_state.name} after {elapsed_time:.2f} seconds")
        self.current_state = self.states[name]
        
        self.state_start_time = time.time() # Record entry time
        start_entry_action = time.time()
        self.current_state.entry_action(self.ctx)
        entry_action_duration = time.time() - start_entry_action

        print(f"[INFO] → Entering state: {name}")
        #print(f"[INFO]     entry_action took {entry_action_duration:.4f} s")


    def update(self, ctx) -> None:
        """Run one state-machine step with the newest frame context."""
        if not self.current_state:
            return
        # keep latest context
        self.ctx = ctx
        start_do_action = time.time()
        next_state_name = self.current_state.do_action(ctx)
        do_action_duration = time.time() - start_do_action
        #print(f"[INFO]     do_action took {do_action_duration:.4f} seconds")
        if next_state_name and next_state_name in self.states:
            self.set_state(next_state_name)
