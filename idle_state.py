from Frame_data import BaseState, FrameContext

class idle_state(BaseState):
    def __init__(self):
        super().__init__("idle")
    def do_action(self, ctx: FrameContext):
        return "searching_door_plane_state"