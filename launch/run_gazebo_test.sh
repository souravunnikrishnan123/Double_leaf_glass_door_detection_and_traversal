#!/usr/bin/env bash
# Configure and launch Gazebo test runs from test_plan_gazebo.json, one after another.
#
# Usage: ./run_gazebo_test.sh <run_number> [<run_number> ...]
#   e.g. ./run_gazebo_test.sh 25
#        ./run_gazebo_test.sh 51, 52, 53     (commas, spaces or both)
#
# For each run:
# 1. Writes the run's robot pose (x, y, yaw) and door hinge positions into normal.launch.
# 2. Moves the door under test to x=2 in university_corridor.world (other doors to x=6, x=10),
#    turned around (yaw π) when the run's door_opening_side is left.
# 3. Starts Gazebo, waits GAZEBO_LOAD_SECONDS, then starts the door detector.
# 4. Once Gazebo is fully loaded and the detector is subscribed, publishes one
#    true on /trigger_start_door_frame_detection to start a detection cycle. A run whose
#    Gazebo is not fully loaded GAZEBO_LOAD_TIMEOUT_SECONDS after the run started ends here.
# 5. Ends the run when /door_traversal/state reaches DONE or ABORT, or RUN_TIMEOUT_SECONDS
#    after Gazebo was fully loaded: stops the detector as Ctrl+C would, then Gazebo, and
#    starts the next run once both have exited.
# Ctrl+C stops the current run and skips the remaining ones.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"   # launch/ -> robodog_glass_door_detection -> src
PLAN="$SCRIPT_DIR/test_plan_gazebo.json"
LAUNCH="$SRC_DIR/unitree_ros/unitree_gazebo/launch/normal.launch"
WORLD="$SRC_DIR/unitree_ros/unitree_gazebo/worlds/university_corridor.world"
GAZEBO_LOAD_SECONDS="${GAZEBO_LOAD_SECONDS:-30}"
GAZEBO_LOAD_TIMEOUT_SECONDS="${GAZEBO_LOAD_TIMEOUT_SECONDS:-600}"   # counted from the run's start
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-300}"   # per run, counted from Gazebo being fully loaded
REFERENCE_DOOR_DISTANCE_M="2.0"
ROBOT_MODEL="go1_gazebo"   # rname:=go1 below spawns the robot as <rname>_gazebo

usage() {
    echo "Usage: $0 <run_number> [<run_number> ...]   (e.g. $0 25  or  $0 51, 52, 53)" >&2
    exit 2
}

read -r -a RUNS <<<"${*//,/ }"
[[ ${#RUNS[@]} -gt 0 ]] || usage
for run in "${RUNS[@]}"; do
    [[ "$run" =~ ^[0-9]+$ ]] || usage
done

source /opt/ros/noetic/setup.bash
source "$SRC_DIR/../devel/setup.bash"
set -eo pipefail

# Writes run $1's robot pose and door setup into normal.launch and the world file.
# With "check" as $2 it only validates the run against the plan and writes nothing.
configure_run() {
    python3 - "$1" "$PLAN" "$LAUNCH" "$WORLD" "${2:-write}" <<'EOF'
import json, re, sys

run_no, plan_path, launch_path, world_path, mode = int(sys.argv[1]), *sys.argv[2:]

DOOR_MODELS = {1: "wall_with_door", 2: "second_wall_with_door", 3: "third_wall_with_door"}
DISTANCE_TO_X = {1.7: 0.0, 2.0: -0.3, 2.3: -0.6}
TEST_DOOR_X, OTHER_DOOR_XS = 2, [6, 10]
HINGE = {"outward": 1.57, "inward": -1.57}
# Every door model has its moving leaf on the robot's right. Turning the door under test
# around puts the leaf on the left.
DOOR_YAW = {"right": 0, "left": 3.14159}

runs = {r["run"]: r for r in json.load(open(plan_path))["runs"]}
if run_no not in runs:
    sys.exit(f"Run {run_no} not found in {plan_path} (available: {min(runs)}-{max(runs)})")
run = runs[run_no]
if run["distance_m"] not in DISTANCE_TO_X:
    sys.exit(f"No x mapping for distance_m={run['distance_m']} (known: {sorted(DISTANCE_TO_X)})")
side = run.get("door_opening_side")
if side not in DOOR_YAW:
    sys.exit(f"Run {run_no}: door_opening_side must be one of {sorted(DOOR_YAW)}, got {side!r}")
if mode == "check":
    sys.exit(0)

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
# The hinge axis stays +z when the door is turned around, so a positive angle then swings
# the leaf away from the robot: outward and inward swap signs.
hinges[run["door"]] = HINGE[run["door_opening"]] * (-1 if side == "left" else 1)
for door, position in hinges.items():
    launch = sub_once(rf'(name="open_door_{door}"[^>]*joint_positions: \[)[^\]]*(\])',
                      rf"\g<1>{position}\g<2>", launch, f"open_door_{door} node in {launch_path}")
open(launch_path, "w").write(launch)

# --- university_corridor.world: door under test at x=2, turned around if it opens left ---
world = open(world_path).read()
other_xs = iter(OTHER_DOOR_XS)
door_poses = {door: f"{TEST_DOOR_X} 0 0 0 0 {DOOR_YAW[side]}" if door == run["door"]
              else f"{next(other_xs)} 0 0 0 0 0" for door in DOOR_MODELS}
for door, model in DOOR_MODELS.items():
    # The model's own <pose> is the first one after its <model> tag, before any <link>.
    world = sub_once(rf'(<model name="{model}">(?:(?!<link|<model).)*?<pose>)[^<]*(</pose>)',
                     rf"\g<1>{door_poses[door]}\g<2>", world, f"model pose for {model} in {world_path}")
open(world_path, "w").write(world)

print(f"Run {run_no}: door {run['door']} ({DOOR_MODELS[run['door']]}), opening {run['door_opening']} "
      f"on the {side} (hinge {hinges[run['door']]})")
print(f"  robot: x={DISTANCE_TO_X[run['distance_m']]} (distance {run['distance_m']} m), "
      f"y={run['lateral_offset_m']}, yaw={run['yaw_offset_rad']}")
print("  door poses: " + ", ".join(f"{DOOR_MODELS[d]}=[{p}]" for d, p in door_poses.items()))
EOF
}

# Check every run before starting the first, so a typo cannot stop the batch midway.
for run in "${RUNS[@]}"; do
    configure_run "$run" check
done

# Lists processes left over from an earlier launch: roslaunch, any node it started
# (each gets a separate __name:= argument) and Gazebo. roscore's own rosout is
# expected. Arguments are checked one by one, so a shell command that merely
# mentions these strings does not count.
leftover_ros_processes() {
    local dir arg args
    for dir in /proc/[0-9]*; do
        mapfile -d '' -t args 2>/dev/null <"$dir/cmdline" || continue
        for arg in "${args[@]:0:2}"; do
            case "${arg##*/}" in
                roslaunch | gzserver | gzclient)
                    echo "  ${dir#/proc/} ${args[*]}"
                    continue 2
                    ;;
            esac
        done
        for arg in "${args[@]}"; do
            if [[ "$arg" == __name:=* && "$arg" != __name:=rosout ]]; then
                echo "  ${dir#/proc/} ${args[*]}"
                continue 2
            fi
        done
    done
}

# A leftover node with the same name as a new one (robot_state_publisher, for one)
# makes the master shut both down, which leaves the run without the robot's TF.
require_clean_slate() {
    local leftovers
    leftovers=$(leftover_ros_processes)
    if [[ -n "$leftovers" ]]; then
        echo "Processes from an earlier launch are still running; stop them first:" >&2
        cut -c1-160 <<<"$leftovers" >&2
        echo "  e.g. kill $(awk '{print $1}' <<<"$leftovers" | xargs)" >&2
        exit 1
    fi
}

require_clean_slate

# Job control gives each roslaunch its own process group, so Ctrl+C at the terminal
# does not reach them directly; this script catches it and stops them in order.
set -m

STATE_LOG="$(mktemp -t door_traversal_state.XXXXXX)"
RESULTS=()
CURRENT_RUN=""

# Sends SIGINT, as Ctrl+C would, to a background job's process group and waits up to
# 30 s for it to exit before killing it. roslaunch starts each node in its own session
# and shuts them all down before it exits.
stop_job() {
    local pid=$1 name=$2
    kill -0 "$pid" 2>/dev/null || return 0
    echo "Stopping $name..."
    kill -INT -- "-$pid" 2>/dev/null || true
    for _ in $(seq 30); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 1
    done
    kill -KILL -- "-$pid" 2>/dev/null || true
}

# Stops whatever the current run has started: the detector first, then Gazebo, the
# same order as Ctrl+C on the detector followed by cleanup in a single run.
stop_run() {
    if [[ -n "${STATE_ECHO_PID:-}" ]]; then
        kill -- "-$STATE_ECHO_PID" 2>/dev/null || true
    fi
    if [[ -n "${TRIGGER_PID:-}" ]]; then
        kill -- "-$TRIGGER_PID" 2>/dev/null || true
    fi
    if [[ -n "${DETECTOR_PID:-}" ]]; then
        stop_job "$DETECTOR_PID" "the detector"
    fi
    if [[ -n "${GAZEBO_PID:-}" ]]; then
        stop_job "$GAZEBO_PID" "Gazebo"
        # The next run refuses to start while anything from this one is still running.
        for _ in $(seq 30); do
            [[ -n "$(leftover_ros_processes)" ]] || break
            sleep 1
        done
    fi
    STATE_ECHO_PID="" TRIGGER_PID="" DETECTOR_PID="" GAZEBO_PID=""
}

cleanup() {
    trap - EXIT INT TERM
    stop_run
    rm -f "$STATE_LOG"
    if [[ -n "$CURRENT_RUN" ]]; then
        RESULTS+=("Run $CURRENT_RUN: stopped early")
    fi
    if [[ ${#RESULTS[@]} -gt 0 ]]; then
        echo "Results:"
        printf '  %s\n' "${RESULTS[@]}"
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
trigger_detection_when_subscribed() {
    until detector_subscribed; do sleep 1; done
    echo "Publishing true on /trigger_start_door_frame_detection"
    rostopic pub -1 /trigger_start_door_frame_detection std_msgs/Bool "data: true" >/dev/null
}

# Waits until Gazebo is fully loaded and sets LOADED_AT, from which the run timeout counts.
# Fails, setting RUN_RESULT, if Gazebo or the detector exits first or loading takes longer
# than GAZEBO_LOAD_TIMEOUT_SECONDS from the run's start.
wait_for_gazebo_loaded() {
    until gazebo_fully_loaded; do
        if ! kill -0 "$GAZEBO_PID" 2>/dev/null; then
            RUN_RESULT="Gazebo exited before it was fully loaded"
            return 1
        fi
        if ! kill -0 "$DETECTOR_PID" 2>/dev/null; then
            RUN_RESULT="detector exited on its own"
            return 1
        fi
        if (( SECONDS - RUN_START >= GAZEBO_LOAD_TIMEOUT_SECONDS )); then
            RUN_RESULT="Gazebo not fully loaded after ${GAZEBO_LOAD_TIMEOUT_SECONDS}s"
            return 1
        fi
        sleep 2
    done
    LOADED_AT=$SECONDS
    echo "Gazebo fully loaded after $((LOADED_AT - RUN_START))s; the ${RUN_TIMEOUT_SECONDS}s run timeout starts now"
}

# Waits until the traversal reaches DONE or ABORT, the run times out or the detector
# exits on its own, and sets RUN_RESULT to which one happened.
wait_for_run_end() {
    local end_state
    while true; do
        # The first of the two to appear decides the result.
        end_state=$(grep -m1 -oE '^data: "(DONE|ABORT)"$' "$STATE_LOG" || true)
        if [[ -n "$end_state" ]]; then
            end_state=${end_state#data: \"}
            RUN_RESULT=${end_state%\"}
            return
        fi
        if (( SECONDS - LOADED_AT >= RUN_TIMEOUT_SECONDS )); then
            RUN_RESULT="neither DONE nor ABORT ${RUN_TIMEOUT_SECONDS}s after Gazebo was fully loaded"
            return
        fi
        if ! kill -0 "$DETECTOR_PID" 2>/dev/null; then
            RUN_RESULT="detector exited on its own"
            return
        fi
        sleep 2
    done
}

run_one() {
    CURRENT_RUN=$1
    RUN_START=$SECONDS
    configure_run "$CURRENT_RUN"
    require_clean_slate

    roslaunch unitree_gazebo normal.launch rname:=go1 wname:=university_corridor &
    GAZEBO_PID=$!

    echo "Waiting ${GAZEBO_LOAD_SECONDS}s for Gazebo to load..."
    sleep "$GAZEBO_LOAD_SECONDS"
    if ! kill -0 "$GAZEBO_PID" 2>/dev/null; then
        echo "Gazebo exited during startup; not starting the detector." >&2
        exit 1
    fi

    roslaunch robodog_glass_door_detection door_detection.launch \
        input_mode:=gazebo debug:=False reference_door_distance_m:="$REFERENCE_DOOR_DISTANCE_M" &
    DETECTOR_PID=$!

    # The traversal node exits shortly after publishing DONE or ABORT, so a subscriber
    # stays up for the whole run instead of polling the topic.
    # PYTHONUNBUFFERED writes each message to the file as it arrives.
    : >"$STATE_LOG"
    PYTHONUNBUFFERED=1 rostopic echo /door_traversal/state >"$STATE_LOG" 2>&1 &
    STATE_ECHO_PID=$!

    if wait_for_gazebo_loaded; then
        trigger_detection_when_subscribed &
        TRIGGER_PID=$!
        wait_for_run_end
    fi
    echo "Run $CURRENT_RUN: $RUN_RESULT ($((SECONDS - RUN_START))s); ending the run"
    stop_run
    RESULTS+=("Run $CURRENT_RUN: $RUN_RESULT")
    CURRENT_RUN=""
}

for i in "${!RUNS[@]}"; do
    echo "=== Run ${RUNS[i]} ($((i + 1)) of ${#RUNS[@]}) ==="
    run_one "${RUNS[i]}"
done
