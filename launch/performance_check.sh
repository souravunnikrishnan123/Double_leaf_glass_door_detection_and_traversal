#!/usr/bin/env bash
# =============================================================================
# performance_check.sh
#
# Launches the full door-detection pipeline, triggers detection, samples CPU and
# memory for every node in the package once a second, then prints a summary and
# plots the result.
#
# CPU is read as deltas of utime+stime from /proc/<pid>/stat. `ps -o %cpu` is
# deliberately not used: it reports total CPU time divided by process age, a
# lifetime average, so sampling it repeatedly yields a decaying curve dominated
# by start-up cost rather than the load at each moment.
#
# Usage:
#   chmod +x performance_check.sh
#   ./performance_check.sh
# =============================================================================

set -o pipefail

# ── Config ────────────────────────────────────────────────────────────────────
LAUNCH_PKG="robodog_glass_door_detection"
LAUNCH_FILE="door_detection.launch"
LAUNCH_ARGS="input_mode:=bag debug:=False reference_door_distance_m:=2.0"

SAMPLE_INTERVAL=1.0                    # seconds between samples
# Seconds of sampling before detection is triggered. Long enough that start-up
# has finished and the pipeline has settled, so the load that appears after the
# trigger is detection itself and not the tail of initialisation.
TRIGGER_DELAY_S=20
OUTPUT_DIR="$HOME/door_detection_profile"
LOG_FILE="$OUTPUT_DIR/resource_log.txt"
SUMMARY_FILE="$OUTPUT_DIR/summary.txt"
MONITOR_SCRIPT="$OUTPUT_DIR/monitor_resources.py"
PLOT_SCRIPT="$OUTPUT_DIR/plot_resources.py"
PLOT_OUTPUT="$OUTPUT_DIR/resource_plot.png"
TRIGGER_MARKER="$OUTPUT_DIR/trigger_time.txt"

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
> "$LOG_FILE"
rm -f "$TRIGGER_MARKER"

# ── Write the monitor script ──────────────────────────────────────────────────
# Written during setup, before the monitor runs: the documented way to stop this
# script is Ctrl+C, which fires the EXIT trap straight into generate_output(), so
# emitting the helpers later would leave the trap without them.
cat > "$MONITOR_SCRIPT" << 'PYEOF'
"""Sample CPU and RSS for every node in the package once per interval.

CPU percent is 100 * (delta utime+stime) / clock_ticks / delta_wallclock, so 100
means one core's worth of work. Values above 100 are expected: OpenCV and Open3D
thread internally.
"""
import os
import sys
import time

LOG_FILE = sys.argv[1]
INTERVAL = float(sys.argv[2])

HZ = os.sysconf("SC_CLK_TCK")
PAGE_KB = os.sysconf("SC_PAGESIZE") // 1024

# Matched on the script's basename, not on a directory path: catkin installs the
# nodes to devel/lib/<package>/, so at run time their command lines contain no
# "scripts/" component and a path-based pattern matches nothing at all. Comparing
# whole basenames is also what keeps main.py from matching
# passability_check_main.py, which a substring test would.
SCRIPT_TO_NODE = {
    "realsense_bag_bridge.py":      "bridge",
    "main.py":                      "detection",
    "passability_check_main.py":    "passability",
    "door_traversal_controller.py": "traversal",
    "odom_bridge.py":               "odom",
}
NAMES = ["bridge", "detection", "passability", "traversal", "odom"]


def find_pids():
    """Map each node label to the pids running its script."""
    found = {name: [] for name in NAMES}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline") as handle:
                tokens = handle.read().split("\0")
        except OSError:
            continue
        for token in tokens:
            if not token.endswith(".py"):
                continue
            name = SCRIPT_TO_NODE.get(os.path.basename(token))
            if name:
                found[name].append(entry)
                break
    return found


def sample(pid):
    """Return (cpu_jiffies, rss_kb) for one pid, or None if it is gone."""
    try:
        with open(f"/proc/{pid}/stat") as handle:
            fields = handle.read().rsplit(")", 1)[1].split()
        jiffies = int(fields[11]) + int(fields[12])
        with open(f"/proc/{pid}/statm") as handle:
            rss_kb = int(handle.read().split()[1]) * PAGE_KB
    except (OSError, IndexError, ValueError):
        return None
    return jiffies, rss_kb


header = ["ELAPSED_S", "TOTAL_CPU_PCT", "TOTAL_RSS_MB"]
for name in NAMES:
    header += [f"{name}_CPU", f"{name}_RSS"]

live = find_pids()
prev = {}
for name in NAMES:
    for pid in live[name]:
        got = sample(pid)
        if got:
            prev[pid] = got[0]
prev_t = time.time()
start_t = prev_t
ever_alive = any(live[name] for name in NAMES)
# Nodes can still be coming up when sampling starts, so keep looking for a while
# rather than exiting on an empty first scan.
GRACE_S = 30.0

with open(LOG_FILE, "w", buffering=1) as out:
    out.write(",".join(header) + "\n")
    print(",".join(header))
    try:
        while True:
            time.sleep(INTERVAL)
            now = time.time()
            dt = now - prev_t
            if dt <= 0:
                continue
            # Pick up nodes that started after the first scan.
            for name, pids in find_pids().items():
                for pid in pids:
                    if pid not in live[name]:
                        live[name].append(pid)
            row_cpu, row_rss, any_alive = {}, {}, False
            for name in NAMES:
                cpu_sum, rss_sum = 0.0, 0.0
                for pid in list(live[name]):
                    got = sample(pid)
                    if got is None:
                        live[name].remove(pid)
                        prev.pop(pid, None)
                        continue
                    any_alive = True
                    jiffies, rss_kb = got
                    if pid in prev:
                        cpu_sum += 100.0 * (jiffies - prev[pid]) / HZ / dt
                    prev[pid] = jiffies
                    rss_sum += rss_kb / 1024.0
                row_cpu[name], row_rss[name] = cpu_sum, rss_sum
            if any_alive:
                ever_alive = True
            elif ever_alive or (now - start_t) > GRACE_S:
                if not ever_alive:
                    print("No package nodes found. Is the pipeline running?",
                          file=sys.stderr)
                break
            prev_t = now
            elapsed = int(round(now - start_t))
            total_cpu = sum(row_cpu.values())
            total_rss = sum(row_rss.values())
            cells = [str(elapsed), f"{total_cpu:.1f}", f"{total_rss:.1f}"]
            for name in NAMES:
                cells += [f"{row_cpu[name]:.1f}", f"{row_rss[name]:.1f}"]
            line = ",".join(cells)
            out.write(line + "\n")
            print(line)
    except KeyboardInterrupt:
        pass
PYEOF

# ── Write the plot script ─────────────────────────────────────────────────────
cat > "$PLOT_SCRIPT" << 'PYEOF'
import sys
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log_file = sys.argv[1]
plot_file = sys.argv[2]

# Optional: elapsed seconds at which detection was triggered. Everything to the
# left of it is start-up and idle, everything to the right is detection.
trigger_at = None
if len(sys.argv) > 3:
    try:
        with open(sys.argv[3]) as handle:
            trigger_at = float(handle.read().strip())
    except (OSError, ValueError):
        trigger_at = None

NAMES = ["bridge", "detection", "passability", "traversal", "odom"]
time_s, total_cpu, total_rss = [], [], []
per_node = {n: [] for n in NAMES}

with open(log_file) as f:
    for row in csv.DictReader(f):
        try:
            time_s.append(int(row["ELAPSED_S"]))
            total_cpu.append(float(row["TOTAL_CPU_PCT"]))
            total_rss.append(float(row["TOTAL_RSS_MB"]))
            for n in NAMES:
                per_node[n].append(float(row[f"{n}_CPU"]))
        except (ValueError, KeyError):
            continue

if not time_s:
    print("No data to plot.")
    sys.exit(1)

avg_cpu, peak_cpu = sum(total_cpu) / len(total_cpu), max(total_cpu)
avg_rss, peak_rss = sum(total_rss) / len(total_rss), max(total_rss)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
fig.suptitle("Door Detection Pipeline - Resource Profile (measured)",
             fontsize=13, fontweight="bold")

ax1.plot(time_s, total_cpu, color="#2196F3", linewidth=1.5, label="Total CPU %")
ax1.fill_between(time_s, total_cpu, alpha=0.15, color="#2196F3")
ax1.axhline(avg_cpu, color="#2196F3", linestyle="--", linewidth=1, alpha=0.7,
            label=f"Avg {avg_cpu:.1f}%")
ax1.axhline(peak_cpu, color="#0d47a1", linestyle=":", linewidth=1, alpha=0.7,
            label=f"Peak {peak_cpu:.1f}%")
ax1.axhline(100, color="grey", linewidth=0.8, alpha=0.5, label="100% = one core")
ax1.set_title(f"Total CPU across all package nodes - avg {avg_cpu:.1f}%, peak {peak_cpu:.1f}%")
ax1.set_ylabel("CPU % (100 = one core)")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

ax2.stackplot(time_s, [per_node[n] for n in NAMES], labels=NAMES, alpha=0.8)
ax2.set_title("CPU per node (stacked)")
ax2.set_ylabel("CPU %")
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)

ax3.plot(time_s, total_rss, color="#4CAF50", linewidth=1.5, label="Total RSS MB")
ax3.fill_between(time_s, total_rss, alpha=0.15, color="#4CAF50")
ax3.axhline(avg_rss, color="#4CAF50", linestyle="--", linewidth=1, alpha=0.7,
            label=f"Avg {avg_rss:.1f} MB")
ax3.axhline(peak_rss, color="#1b5e20", linestyle=":", linewidth=1, alpha=0.7,
            label=f"Peak {peak_rss:.1f} MB")
ax3.set_title(f"Total memory (RSS) - avg {avg_rss:.1f} MB, peak {peak_rss:.1f} MB")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Memory (MB)")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(bottom=0)

if trigger_at is not None:
    for ax in (ax1, ax2, ax3):
        ax.axvline(trigger_at, color="#c62828", linestyle="-", linewidth=1.4,
                   alpha=0.9, zorder=5)
    ax1.annotate("detection triggered", xy=(trigger_at, ax1.get_ylim()[1]),
                 xytext=(4, -12), textcoords="offset points",
                 color="#c62828", fontsize=9, fontweight="bold")

fig.tight_layout()
plt.savefig(plot_file, dpi=150, bbox_inches="tight")
print(f"Plot saved: {plot_file}")
PYEOF

echo "============================================="
echo "  Door Detection Pipeline - Resource Profiler"
echo "============================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# ── Source ROS ────────────────────────────────────────────────────────────────
source ~/.bashrc

# ── Trap: generate output on Ctrl+C, kill, or normal exit ─────────────────────
MONITORING=false

generate_output() {
    if [ "${OUTPUT_GENERATED:-false}" = "true" ]; then return; fi
    OUTPUT_GENERATED=true
    if [ "$MONITORING" = "false" ]; then return; fi

    echo ""
    echo "[*] Generating summary and plot..."

    kill "${MONITOR_PID:-}" 2>/dev/null || true
    kill "$LAUNCH_PID" 2>/dev/null || true
    [ -n "${ROSCORE_PID:-}" ] && kill "$ROSCORE_PID" 2>/dev/null || true

    if [ -s "$LOG_FILE" ]; then
        AVG_CPU=$(tail -n +2 "$LOG_FILE"  | awk -F',' '{sum+=$2; n++} END {if(n) printf "%.1f", sum/n}')
        PEAK_CPU=$(tail -n +2 "$LOG_FILE" | awk -F',' 'BEGIN{m=0} {if($2>m) m=$2} END {printf "%.1f", m}')
        AVG_MEM=$(tail -n +2 "$LOG_FILE"  | awk -F',' '{sum+=$3; n++} END {if(n) printf "%.1f", sum/n}')
        PEAK_MEM=$(tail -n +2 "$LOG_FILE" | awk -F',' 'BEGIN{m=0} {if($3>m) m=$3} END {printf "%.1f", m}')
        DURATION=$(tail -n +2 "$LOG_FILE" | tail -1 | cut -d',' -f1)
        # Per-node average CPU: columns 4,6,8,10,12 hold each node's CPU percent.
        NODE_TABLE=$(tail -n +2 "$LOG_FILE" | awk -F',' '
            {b+=$4; d+=$6; p+=$8; t+=$10; o+=$12; n++}
            END {if(n) printf "  bridge          : %.1f%%\n  detection       : %.1f%%\n  passability     : %.1f%%\n  traversal       : %.1f%%\n  odom            : %.1f%%", b/n, d/n, p/n, t/n, o/n}')

        tee "$SUMMARY_FILE" <<EOF
=============================================
  RESOURCE USAGE SUMMARY
=============================================
Run duration      : ${DURATION}s

--- Desktop (measured) ---
  Avg CPU         : ${AVG_CPU}%
  Peak CPU        : ${PEAK_CPU}%
  Avg RAM (RSS)   : ${AVG_MEM} MB
  Peak RAM (RSS)  : ${PEAK_MEM} MB

--- Avg CPU per node ---
${NODE_TABLE}
=============================================
EOF
    fi

    echo ""
    echo "[✓] Done. Outputs:"
    echo "    Raw log  : $LOG_FILE"
    echo "    Summary  : $SUMMARY_FILE"
    if [ -f "$PLOT_SCRIPT" ] && python3 "$PLOT_SCRIPT" "$LOG_FILE" "$PLOT_OUTPUT" "$TRIGGER_MARKER" >/dev/null 2>&1; then
        echo "    Plot     : $PLOT_OUTPUT"
    else
        echo "[!] Plot generation failed. Check that matplotlib is installed:"
        echo "    pip install matplotlib --break-system-packages"
    fi
}

trap 'generate_output' EXIT INT TERM

# ── Start roscore if not already running ──────────────────────────────────────
if ! rostopic list &>/dev/null; then
    echo "[*] Starting roscore..."
    roscore &
    ROSCORE_PID=$!
    sleep 3
else
    echo "[*] roscore already running."
    ROSCORE_PID=""
fi

# ── Launch the pipeline ───────────────────────────────────────────────────────
echo "[*] Launching $LAUNCH_FILE ..."
roslaunch "$LAUNCH_PKG" "$LAUNCH_FILE" $LAUNCH_ARGS &
LAUNCH_PID=$!

echo "[*] Waiting for nodes to start..."
for i in $(seq 1 45); do
    rostopic list 2>/dev/null | grep -q glass_door_detection && break
    sleep 1
done
sleep 4

# ── Monitor, then trigger ─────────────────────────────────────────────────────
# Sampling starts BEFORE the trigger is sent. `rostopic pub -1` blocks for about
# four seconds while it registers a publisher and waits for subscriber
# connections, and the pipeline reacts to the message as soon as it lands - the
# RANSAC plane search and the whole open/closed decision are typically finished
# inside that window. Triggering first would therefore start the measurement
# after the most expensive phase was already over, leaving only post-detection
# idle in the log.
echo "[*] Sampling CPU + memory every ${SAMPLE_INTERVAL}s... (Ctrl+C to stop and generate output)"
echo ""

MONITORING=true
MONITOR_START=$(date +%s.%N)
python3 "$MONITOR_SCRIPT" "$LOG_FILE" "$SAMPLE_INTERVAL" &
MONITOR_PID=$!

# Baseline the idle pipeline first. Anything before this mark is start-up and
# settling; anything after it is detection.
echo "[*] Baselining idle pipeline for ${TRIGGER_DELAY_S}s before triggering..."
sleep "$TRIGGER_DELAY_S"

# Elapsed seconds at the moment of the trigger, matching the log's ELAPSED_S
# column so the plot can mark exactly where detection begins.
TRIGGER_AT=$(awk -v a="$MONITOR_START" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}')
echo "$TRIGGER_AT" > "$TRIGGER_MARKER"

echo "[*] Triggering door frame detection at t=${TRIGGER_AT}s ..."
# Without this the state machine stays in idle_state and the run would measure
# the pipeline not detecting anything.
rostopic pub -1 /trigger_start_door_frame_detection std_msgs/Bool "data: true" >/dev/null 2>&1

wait "$MONITOR_PID"

# script ends here — generate_output() is called automatically via trap
