

class StateMachine:
    """Handles transitions and execution of states."""
    def __init__(self, ctx=None):
        self.states = {}
        self.current_state = None
        self.ctx = ctx

    def add_state(self, state) -> None:
        self.states[state.name] = state

    def set_state(self, name) -> None:
        """Switch to a new state by name."""
        if self.current_state:
            self.current_state.exit_action(self.ctx)
        self.current_state = self.states[name]
        print(f"[INFO] → Entering state: {name}")
        self.current_state.entry_action(self.ctx)

    def update(self, ctx) -> None:
        """Call the update loop of the current state."""
        if not self.current_state:
            return
        # keep latest context
        self.ctx = ctx
        next_state_name = self.current_state.do_action(ctx)
        if next_state_name and next_state_name in self.states:
            self.set_state(next_state_name)
