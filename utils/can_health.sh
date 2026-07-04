#!/bin/bash
# Show health of every CAN interface and (optionally) recover broken ones.
#
# Diagnoses the "SendCanMessage(SEND_MESSAGE_FAILED)" / bus-off problems:
# prints link state, CAN error state, error counters and tx queue length.
#
# Usage:
#   bash utils/can_health.sh                 # just report
#   bash utils/can_health.sh --fix           # restart any bus-off iface + tune txqueuelen/restart-ms
#   bash utils/can_health.sh --fix can_master right_follower   # only these
#   bash utils/can_health.sh --wait left_follower right_follower   # block until ALIVE (launch guard)
#   bash utils/can_health.sh --wait --timeout 60 ...              # give up after 60s (exit 1)
#
# --fix  applies to bus-off interfaces (down/up re-init) and sets a larger
#        txqueuelen(1000) + restart-ms(100) so a hiccup auto-recovers instead
#        of wedging the bus.
# --wait polls until every named arm is transmitting feedback (RX growing),
#        then prints the report and exits 0. Times out (default 30s) -> exit 1.
#        Put it at the top of launch scripts so recording/teleop only starts
#        once the arms are really alive.

BITRATE=1000000
FIX=0
WAIT=0
TIMEOUT=30
IFACES=()

while [ $# -gt 0 ]; do
    case "$1" in
        --fix) FIX=1 ;;
        --wait) WAIT=1 ;;
        --timeout) shift; TIMEOUT="$1" ;;
        *) IFACES+=("$1") ;;
    esac
    shift
done

# Default to all CAN interfaces if none were named.
if [ ${#IFACES[@]} -eq 0 ]; then
    mapfile -t IFACES < <(ip -br link show type can 2>/dev/null | awk '{print $1}')
fi

if [ ${#IFACES[@]} -eq 0 ]; then
    echo "No CAN interfaces found. Activate them first (see README section 3)."
    exit 1
fi

# Is interface $1 receiving arm feedback? (RX packets grow over a short window)
rx_alive() {
    local iface="$1"
    ip link show "$iface" >/dev/null 2>&1 || return 1
    local a b
    a=$(cat /sys/class/net/"$iface"/statistics/rx_packets 2>/dev/null || echo 0)
    sleep 0.3
    b=$(cat /sys/class/net/"$iface"/statistics/rx_packets 2>/dev/null || echo 0)
    [ "$b" -gt "$a" ]
}

# --wait: block until every named arm is transmitting, or time out.
if [ "$WAIT" -eq 1 ]; then
    echo "Waiting up to ${TIMEOUT}s for arms to come ALIVE: ${IFACES[*]}"
    elapsed=0
    while true; do
        all_alive=1
        for i in "${IFACES[@]}"; do
            rx_alive "$i" || { all_alive=0; break; }
        done
        if [ "$all_alive" -eq 1 ]; then
            echo "All arms ALIVE after ~${elapsed}s."
            break
        fi
        if [ "$elapsed" -ge "$TIMEOUT" ]; then
            echo "TIMEOUT after ${TIMEOUT}s: not all arms are transmitting (power on / release e-stop / check cables)."
            bash "$0" "${IFACES[@]}"
            exit 1
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
fi

# Sample RX packets over a short window: a live, powered arm continuously
# broadcasts feedback frames, so a growing RX count is the real "arm alive"
# signal. can_state=ERROR-ACTIVE alone is just the controller's default and
# does NOT prove an arm is connected.
declare -A RX0
for i in "${IFACES[@]}"; do
    ip link show "$i" >/dev/null 2>&1 && RX0[$i]=$(cat /sys/class/net/"$i"/statistics/rx_packets 2>/dev/null)
done
sleep 0.5

for i in "${IFACES[@]}"; do
    if ! ip link show "$i" >/dev/null 2>&1; then
        echo "=== $i === NOT FOUND"
        continue
    fi

    LINK_STATE=$(ip -br link show "$i" | awk '{print $2}')
    CAN_STATE=$(ip -details link show "$i" | grep -oP 'can state \K[A-Z-]+')
    BITRATE_NOW=$(ip -details link show "$i" | grep -oP 'bitrate \K\d+')
    QLEN=$(ip -details link show "$i" | grep -oP 'qlen \K\d+')
    # error counter line (values under: bus-errors error-warn error-pass bus-off ...)
    COUNTERS=$(ip -details -statistics link show "$i" | grep -A1 'bus-off' | tail -n1 | awk '{print $1,$2,$3,$4,$5,$6}')

    RX_NOW=$(cat /sys/class/net/"$i"/statistics/rx_packets 2>/dev/null)
    RX_DELTA=$(( ${RX_NOW:-0} - ${RX0[$i]:-0} ))

    echo "=== $i ==="
    echo "  link=$LINK_STATE  can_state=${CAN_STATE:-?}  bitrate=${BITRATE_NOW:-?}  qlen=${QLEN:-?}  rx_in_0.5s=$RX_DELTA"
    [ -n "$COUNTERS" ] && echo "  counters(restarts bus-errs arb-lost err-warn err-pass bus-off):$COUNTERS"

    # Verdict combines controller state with actual arm traffic.
    if [ "$CAN_STATE" = "BUS-OFF" ]; then
        echo "  -> BUS-OFF (interface wedged; needs restart)"
    elif [ "$RX_DELTA" -le 0 ]; then
        echo "  -> SILENT (controller ok but NO arm feedback: arm off / e-stop / cable unplugged)"
    elif [ "$CAN_STATE" = "ERROR-WARNING" ] || [ "$CAN_STATE" = "ERROR-PASSIVE" ]; then
        echo "  -> DEGRADED (arm talking but errors accumulating: check cable / termination)"
    else
        echo "  -> ALIVE (arm transmitting, no errors)"
    fi

    if [ "$FIX" -eq 1 ]; then
        if [ "$CAN_STATE" = "BUS-OFF" ] || [ "$LINK_STATE" != "UP" ]; then
            echo "  [fix] restarting $i ..."
            sudo ip link set "$i" down
            sudo ip link set "$i" type can bitrate "$BITRATE" restart-ms 100
            sudo ip link set "$i" txqueuelen 1000
            sudo ip link set "$i" up
        else
            # healthy: just make sure it can self-recover next time
            sudo ip link set "$i" txqueuelen 1000 2>/dev/null
        fi
        NEW=$(ip -details link show "$i" | grep -oP 'can state \K[A-Z-]+')
        echo "  [fix] $i now: $NEW"
    fi
done
