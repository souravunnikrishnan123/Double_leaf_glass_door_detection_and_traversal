#!/usr/bin/env bash
# =============================================================================
# profile_door_detection.sh
#
# Launches the door detection node, monitors CPU + memory every second,
# prints a summary, and plots CPU and memory over time when the bag finishes.
#
# Usage:
#   chmod +x profile_door_detection.sh
#   ./profile_door_detection.sh
# =============================================================================

set -o pipefail

# ── Config ────────────────────────────────────────────────────────────────────
LAUNCH_PKG="robodog_glass_door_detection"
LAUNCH_FILE="door_detection.launch"
LAUNCH_ARGS="input_mode:=bag debug:=False reference_door_distance_m:=2.0"

# Matched against the full command line. Anchored on the directory separator
# because a bare "main.py" also matches passability_check_main.py, which would
# silently profile the passability node instead of this one.
NODE_SEARCH_TERM="main.py"
SAMPLE_INTERVAL=1                           # seconds between samples
OUTPUT_DIR="$HOME/door_detection_profile"
LOG_FILE="$OUTPUT_DIR/resource_log.txt"
SUMMARY_FILE="$OUTPUT_DIR/summary.txt"
PLOT_SCRIPT="$OUTPUT_DIR/plot_resources.py"
PLOT_OUTPUT="$OUTPUT_DIR/resource_plot.png"

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
> "$LOG_FILE"   # clear previous log

# ── Write the plot script ─────────────────────────────────────────────────────
# Written during setup, before the monitor loop: the documented way to stop
# this script is Ctrl+C, which fires the EXIT trap straight into
# generate_output(), so emitting it after the loop would leave the trap with
# no plot script on exactly the path people actually use.
cat > "$PLOT_SCRIPT" << 'PYEOF'
import sys
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log_file  = sys.argv[1]
plot_file = sys.argv[2]

time_s, cpu, mem_mb = [], [], []

with open(log_file) as f:
    for row in csv.DictReader(f):
        try:
            time_s.append(int(row["ELAPSED_S"]))
            cpu.append(float(row["CPU_PCT"]))
            mem_mb.append(float(row["RSS_MB"]))
        except (ValueError, KeyError):
            continue

if not time_s:
    print("No data to plot.")
    sys.exit(1)

avg_cpu, peak_cpu = sum(cpu) / len(cpu), max(cpu)
avg_mem, peak_mem = sum(mem_mb) / len(mem_mb), max(mem_mb)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
fig.suptitle("Door Detection Node - Resource Profile (measured)",
             fontsize=13, fontweight="bold")

# CPU. Values above 100 percent mean more than one core's worth of work, which
# is expected for a node whose OpenCV and Open3D calls thread internally.
ax1.plot(time_s, cpu, color="#2196F3", linewidth=1.5, label="CPU %")
ax1.fill_between(time_s, cpu, alpha=0.15, color="#2196F3")
ax1.axhline(avg_cpu, color="#2196F3", linestyle="--", linewidth=1, alpha=0.7,
            label=f"Avg {avg_cpu:.1f}%")
ax1.axhline(peak_cpu, color="#0d47a1", linestyle=":", linewidth=1, alpha=0.7,
            label=f"Peak {peak_cpu:.1f}%")
ax1.axhline(100, color="grey", linestyle="-", linewidth=0.8, alpha=0.5,
            label="100% = one core")
ax1.set_title(f"CPU usage - avg {avg_cpu:.1f}%, peak {peak_cpu:.1f}%")
ax1.set_ylabel("CPU % (100 = one core)")
ax1.legend(fontsize=8, loc="best")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

# Memory.
ax2.plot(time_s, mem_mb, color="#4CAF50", linewidth=1.5, label="RSS MB")
ax2.fill_between(time_s, mem_mb, alpha=0.15, color="#4CAF50")
ax2.axhline(avg_mem, color="#4CAF50", linestyle="--", linewidth=1, alpha=0.7,
            label=f"Avg {avg_mem:.1f} MB")
ax2.axhline(peak_mem, color="#1b5e20", linestyle=":", linewidth=1, alpha=0.7,
            label=f"Peak {peak_mem:.1f} MB")
ax2.set_title(f"Memory (RSS) - avg {avg_mem:.1f} MB, peak {peak_mem:.1f} MB")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Memory (MB)")
ax2.legend(fontsize=8, loc="best")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)

fig.tight_layout()
plt.savefig(plot_file, dpi=150, bbox_inches="tight")
print(f"Plot saved: {plot_file}")
PYEOF


echo "============================================="
echo "  Door Detection Node - Resource Profiler"
echo "============================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# ── Source ROS ────────────────────────────────────────────────────────────────
source ~/.bashrc

# ── Trap: generate output on Ctrl+C, kill, or normal exit ─────────────────────
MONITORING=false   # flag: only generate output if monitoring actually started

generate_output() {
    # Avoid running twice (trap fires on EXIT too)
    if [ "${OUTPUT_GENERATED:-false}" = "true" ]; then return; fi
    OUTPUT_GENERATED=true

    if [ "$MONITORING" = "false" ]; then return; fi

    echo ""
    echo "[*] Generating summary and plot..."

    kill "$LAUNCH_PID" 2>/dev/null || true
    [ -n "$ROSCORE_PID" ] && kill "$ROSCORE_PID" 2>/dev/null || true

    # ── Generate summary ──────────────────────────────────────────────────────
    if [ -s "$LOG_FILE" ]; then
        AVG_CPU=$(tail -n +2 "$LOG_FILE" | awk -F',' '{sum+=$2; n++} END {printf "%.1f", sum/n}')
        PEAK_CPU=$(tail -n +2 "$LOG_FILE" | awk -F',' 'BEGIN{max=0} {if($2>max) max=$2} END {printf "%.1f", max}')
        AVG_MEM=$(tail -n +2 "$LOG_FILE" | awk -F',' '{sum+=$4; n++} END {printf "%.1f", sum/n}')
        PEAK_MEM=$(tail -n +2 "$LOG_FILE" | awk -F',' 'BEGIN{max=0} {if($4>max) max=$4} END {printf "%.1f", max}')
        DURATION=$(tail -n +2 "$LOG_FILE" | tail -1 | cut -d',' -f1)

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
=============================================
EOF
    fi

    # ── Plot CPU and memory over time ─────────────────────────────────────────
    echo ""
    echo "[✓] Done. Outputs:"
    echo "    Raw log  : $LOG_FILE"
    echo "    Summary  : $SUMMARY_FILE"
    if [ -f "$PLOT_SCRIPT" ] && python3 "$PLOT_SCRIPT" "$LOG_FILE" "$PLOT_OUTPUT" >/dev/null 2>&1; then
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

# ── Launch the node ───────────────────────────────────────────────────────────
echo "[*] Launching $LAUNCH_FILE ..."
roslaunch "$LAUNCH_PKG" "$LAUNCH_FILE" $LAUNCH_ARGS &
LAUNCH_PID=$!

echo "[*] Waiting for node to start..."
sleep 10

# ── Find node PID ─────────────────────────────────────────────────────────────
NODE_PID=$(pgrep -f "$NODE_SEARCH_TERM" | head -1)

if [ -z "$NODE_PID" ]; then
    echo "[ERROR] Could not find node process matching '$NODE_SEARCH_TERM'."
    echo "        Check that the node is running with: rosnode list"
    kill "$LAUNCH_PID" 2>/dev/null
    [ -n "$ROSCORE_PID" ] && kill "$ROSCORE_PID" 2>/dev/null
    exit 1
fi

echo "[*] Found node PID: $NODE_PID"
echo "[*] Sampling CPU + memory every ${SAMPLE_INTERVAL}s... (Ctrl+C to stop and generate output)"
echo ""
echo "ELAPSED_S,CPU_PCT,MEM_PCT,RSS_MB" | tee "$LOG_FILE"

# ── Monitor loop ──────────────────────────────────────────────────────────────
MONITORING=true
ELAPSED=0
while kill -0 "$NODE_PID" 2>/dev/null; do
    STATS=$(ps -p "$NODE_PID" -o %cpu,%mem,rss --no-headers 2>/dev/null)

    if [ -z "$STATS" ]; then
        break
    fi

    CPU=$(echo "$STATS" | awk '{print $1}')
    MEM=$(echo "$STATS" | awk '{print $2}')
    RSS=$(echo "$STATS" | awk '{printf "%.1f", $3/1024}')   # KB -> MB

    echo "$ELAPSED,$CPU,$MEM,$RSS" | tee -a "$LOG_FILE"

    sleep "$SAMPLE_INTERVAL"
    ELAPSED=$((ELAPSED + SAMPLE_INTERVAL))
done

# script ends here — generate_output() is called automatically via trap