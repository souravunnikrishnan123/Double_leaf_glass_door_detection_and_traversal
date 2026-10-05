#!/usr/bin/env bash
# Record a simulation run for later side-by-side video.
#
#   ./record_run.sh                       # uses the defaults below
#   ./record_run.sh -p /my/robot/cam      # different robot camera topic
#   ./record_run.sh -r                    # record RAW instead of compressed
#
# Records the COMPRESSED image topics by default. Raw bgr8 is about 40x larger
# (1280x720 at 30 Hz is ~5 GB per minute) and on a laptop the disk writes alone
# will drag the real-time factor down further. bag_to_video.py reads compressed
# topics natively, so there is no reason to pay for raw here.
#
# /clock is recorded too: it is small, and it lets you confirm afterwards that
# the run really was on simulation time.
set -euo pipefail

BIRDSEYE=/birdseye/image_raw
POV=/camera/color/image_raw          # <-- CHANGE to your robot's camera topic
OUT="run_$(date +%Y%m%d_%H%M%S)"
RAW=0

while getopts "b:p:o:rh" opt; do
  case $opt in
    b) BIRDSEYE="$OPTARG" ;;
    p) POV="$OPTARG" ;;
    o) OUT="$OPTARG" ;;
    r) RAW=1 ;;
    h) sed -n '2,20p' "$0"; exit 0 ;;
    *) exit 2 ;;
  esac
done

if [ "$RAW" -eq 1 ]; then
  TOPICS=("$BIRDSEYE" "$POV")
else
  TOPICS=("${BIRDSEYE}/compressed" "${POV}/compressed")
fi

echo "=== pre-flight ==="

# use_sim_time must be true, or the header stamps are wall-clock epoch numbers
# and the whole point of this approach (a shared simulation clock) is lost.
SIMTIME=$(rosparam get /use_sim_time 2>/dev/null || echo "unset")
echo "  /use_sim_time = $SIMTIME"
if [ "$SIMTIME" != "true" ]; then
  echo "  !! use_sim_time is not true. Header stamps will be wall-clock time."
  echo "     Fix with:  rosparam set /use_sim_time true"
  echo "     and restart the nodes that publish these topics."
  read -r -p "  continue anyway? [y/N] " a; [ "$a" = "y" ] || exit 1
fi

for t in "${TOPICS[@]}"; do
  if rostopic info "$t" >/dev/null 2>&1; then
    echo "  OK   $t"
  else
    echo "  !!   $t  is not advertised."
    if [[ "$t" == *"/compressed" ]]; then
      echo "       Compressed transports appear only once something subscribes."
      echo "       Open it once:  rosrun image_view image_view image:=${t%/compressed} _image_transport:=compressed"
      echo "       or record raw with -r."
    fi
    exit 1
  fi
done

# Confirm the stamps really are simulation time. A stamp above ~1e9 is a Unix
# epoch, i.e. wall-clock, which means use_sim_time did not take effect.
STAMP=$(rostopic echo -n1 "${TOPICS[0]}/header/stamp/secs" 2>/dev/null | head -1 || echo "")
if [ -n "$STAMP" ] && [ "$STAMP" -gt 1000000000 ] 2>/dev/null; then
  echo "  !! header stamp is $STAMP -- that is wall-clock, not simulation time."
  read -r -p "  continue anyway? [y/N] " a; [ "$a" = "y" ] || exit 1
else
  echo "  header stamp looks like simulation time ($STAMP s)"
fi

echo
echo "=== recording to ${OUT}.bag ==="
printf '  %s\n' "${TOPICS[@]}" /clock
echo "  Ctrl-C to stop."
rosbag record -O "$OUT" "${TOPICS[@]}" /clock

echo
echo "Done. Build the side-by-side video with:"
echo "  python3 bag_to_video.py ${OUT}.bag --probe"
echo "  python3 bag_to_video.py ${OUT}.bag \\"
echo "      -t ${TOPICS[0]} \\"
echo "      -t ${TOPICS[1]} \\"
echo "      --layout h --label --overlay -o presentation.mp4"
