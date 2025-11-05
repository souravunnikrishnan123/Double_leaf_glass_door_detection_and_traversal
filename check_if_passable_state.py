from Frame_data import BaseState , FrameContext

class check_if_passable_state(BaseState):
    def __init__(self, name):
        super().__init__(name)
        self.name = "check_if_passable_state"

    def do_action(self, ctx: FrameContext):
        # Determine if the door is passable based on door_state_label
        if ctx.door_state_label == "open":
            ctx.passable_ratio = 1.0
        elif ctx.door_state_label == "closed":
            ctx.passable_ratio = 0.0
        else:
            ctx.passable_ratio = 0.5  # partially open or unknown

        return None  # Stay in the current state