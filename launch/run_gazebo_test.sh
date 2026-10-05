#!/usr/bin/env bash
# Configure and launch one Gazebo test run from test_plan_gazebo.json.
#
# Usage: ./run_gazebo_test.sh <run_number>
#   e.g. ./run_gazebo_test.sh 25
#
# 1. Writes the run's robot pose (x, y, yaw) and door hinge positions into normal.launch.
# 2. Moves the door under test to x=2 in university_corridor.world (other doors to x=6, x=10).
# 3. Starts Gazebo, waits GAZEBO_LOAD_SECONDS, then starts the door detector.
# 4. Once Gazebo is fully loaded and the detector is subscribed, publishes one
#    true on /trigger_start_door_frame_detection to start a detection cycle.
# Ctrl+C stops the detector; Gazebo is then shut down as well.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"   # launch/ -> robodog_glass_door_detection -> src
PLAN="$SCRIPT_DIR/test_plan_gazebo.json"
LAUNCH="$SRC_DIR/unitree_ros/unitree_gazebo/launch/normal.launch"
WORLD="$SRC_DIR/unitree_ros/unitree_gazebo/worlds/university_corridor.world"
GAZEBO_LOAD_SECONDS="${GAZEBO_LOAD_SECONDS:-30}"
REFERENCE_DOOR_DISTANCE_M="2.0"
ROBOT_MODEL="go1_gazebo"   # rname:=go1 below spawns the robot as <rname>_gazebo

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 <run_number>   (e.g. $0 25)" >&2
    exit 2
fi
RUN="$1"

source /opt/ros/noetic/setup.bash
source "$SRC_DIR/../devel/setup.bash"
set -eo pipefail

python3 - "$RUN" "$PLAN" "$LAUNCH" "$WORLD" <<'EOF'
import json, re, sys

run_no, plan_path, launch_path, world_path = int(sys.argv[1]), *sys.argv[2:]

DOOR_MODELS = {1: "wall_with_door", 2: "second_wall_with_door", 3: "third_wall_with_door"}
DISTANCE_TO_X = {1.7: 0.0, 2.0: -0.3, 2.3: -0.6}
TEST_DOOR_X, OTHER_DOOR_XS = 2, [6, 10]
HINGE = {"outward": 1.57, "inward": -1.57}

runs = {r["run"]: r for r in json.load(open(plan_path))["runs"]}
if run_no not in runs:
    sys.exit(f"Run {run_no} not found in {plan_path} (available: {min(runs)}-{max(runs)})")
run = runs[run_no]
if run["distance_m"] not in DISTANCE_TO_X:
    sys.exit(f"No x mapping for distance_m={run['distance_m']} (known: {sorted(DISTANCE_TO_X)})")

def sub_once(pattern, repl, text, what):
    new, n = re.subn(pattern, repl, text, flags=re.S)
    if n != 1:
        sys.exit(f"Expected exactly one {what}, found {n}")
    return new

# --- normal.launch: robot pose and door hinge positions ---
launch = open(launch_path).read()
for arg, value in [("x", DISTANCE_TO_X[run["distance_m"]]),
                   ("y", run["lateral_offset_m"]),
                   ("yaw", run["yaw_offset_rad"])]:
    launch = sub_once(rf'(<arg name="{arg}" default=")[^"]*(")', rf"\g<1>{value}\g<2>",
                      launch, f'<arg name="{arg}"> in {launch_path}')
# Doors not under test go back to outward so every run starts from the same scene.
hinges = {door: HINGE["outward"] for door in DOOR_MODELS}
hinges[run["door"]] = HINGE[run["door_opening"]]
for door, position in hinges.items():
    launch = sub_once(rf'(name="open_door_{door}"[^>]*joint_positions: \[)[^\]]*(\])',
                      rf"\g<1>{position}\g<2>", launch, f"open_door_{door} node in {launch_path}")
open(launch_path, "w").write(launch)

# --- university_corridor.world: door under test at x=2 ---
world = open(world_path).read()
other_xs = iter(OTHER_DOOR_XS)
door_xs = {door: TEST_DOOR_X if door == run["door"] else next(other_xs) for door in DOOR_MODELS}
for door, model in DOOR_MODELS.items():
    # The model's own <pose> is the first one after its <model> tag, before any <link>.
    world = sub_once(rf'(<model name="{model}">(?:(?!<link|<model).)*?<pose>)[^<]*(</pose>)',
                     rf"\g<1>{door_xs[door]} 0 0 0 0 0\g<2>", world, f"model pose for {model} in {world_path}")
open(world_path, "w").write(world)

print(f"Run {run_no}: door {run['door']} ({DOOR_MODELS[run['door']]}), opening {run['door_opening']} "
      f"(hinge {hinges[run['door']]})")
print(f"  robot: x={DISTANCE_TO_X[run['distance_m']]} (distance {run['distance_m']} m), "
      f"y={run['lateral_offset_m']}, yaw={run['yaw_offset_rad']}")
print("  door x positions: " + ", ".join(f"{DOOR_MODELS[d]}={x}" for d, x in door_xs.items()))
EOF

if pgrep -x gzserver >/dev/null; then
    echo "gzserver is already running; stop the previous simulation first." >&2
    exit 1
fi

# Job control gives each roslaunch its own process group, so Ctrl+C reaches only the
# foreground job and the cleanup below can stop Gazebo's whole group.
set -m

cleanup() {
    trap - EXIT INT TERM
    if [[ -n "${TRIGGER_PID:-}" ]]; then
        kill -- "-$TRIGGER_PID" 2>/dev/null || true
    fi
    if [[ -n "${GAZEBO_PID:-}" ]] && kill -0 "$GAZEBO_PID" 2>/dev/null; then
        echo "Stopping Gazebo..."
        kill -INT -- "-$GAZEBO_PID" 2>/dev/null || true
        for _ in $(seq 30); do
            kill -0 "$GAZEBO_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$GAZEBO_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# Gazebo counts as fully loaded once the robot is in the world, physics is running
# (spawn_model unpauses it only after a successful spawn) and the spawn and door
# hinge helpers from normal.launch have exited. Output is captured before matching
# because grep -q closing a pipe early would fail the pipeline under pipefail.
gazebo_fully_loaded() {
    local names physics nodes
    names=$(timeout 5 rostopic echo -n 1 /gazebo/model_states/name 2>/dev/null) || return 1
    grep -qxF -- "- $ROBOT_MODEL" <<<"$names" || return 1
    physics=$(timeout 5 rosservice call /gazebo/get_physics_properties 2>/dev/null) || return 1
    [[ "$physics" == *"pause: False"* ]] || return 1
    nodes=$(timeout 5 rosnode list 2>/dev/null) || return 1
    ! grep -qE '^/(urdf_spawner|spawn_birdseye|open_door_[0-9]+)$' <<<"$nodes"
}

detector_subscribed() {
    local info
    info=$(timeout 5 rostopic info /trigger_start_door_frame_detection 2>/dev/null) || return 1
    [[ "${info#*Subscribers:}" == *" /glass_door_detection "* ]]
}

# Each true message starts exactly one detection cycle, so publish once.
trigger_detection_when_loaded() {
    until gazebo_fully_loaded; do sleep 2; done
    until detector_subscribed; do sleep 1; done
    echo "Gazebo fully loaded; publishing true on /trigger_start_door_frame_detection"
    rostopic pub -1 /trigger_start_door_frame_detection std_msgs/Bool "data: true" >/dev/null
}

roslaunch unitree_gazebo normal.launch rname:=go1 wname:=university_corridor &
GAZEBO_PID=$!

echo "Waiting ${GAZEBO_LOAD_SECONDS}s for Gazebo to load..."
sleep "$GAZEBO_LOAD_SECONDS"
if ! kill -0 "$GAZEBO_PID" 2>/dev/null; then
    echo "Gazebo exited during startup; not starting the detector." >&2
    exit 1
fi

# Runs in its own process group (set -m), so Ctrl+C reaches only the detector;
# cleanup stops it if the trigger has not been sent yet.
trigger_detection_when_loaded &
TRIGGER_PID=$!

roslaunch robodog_glass_door_detection door_detection.launch \
    input_mode:=gazebo debug:=False reference_door_distance_m:="$REFERENCE_DOOR_DISTANCE_M"
