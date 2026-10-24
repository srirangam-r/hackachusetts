#!/usr/bin/env bash
# Gather the real footage and stills the video uses into public/footage/ (git-ignored).
# Needs the recorded runs under ../runs and, for the demo strips, the live server on $URL serving live_184743.
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS=${RUNS:-../runs}
URL=${URL:-http://localhost:8790}
EP=$RUNS/live/live_184743/ep_002            # the hero run: drawer, butter, close, orange juice (one recovery)
OUT=public/footage
mkdir -p "$OUT"

# cold open: the raw sim camera of the hero run (open drawer, pick butter, place it in the drawer), 20 fps = 2x
ffmpeg -v error -y -framerate 20 -start_number 0 -i "$EP/rec/%06d.jpg" -frames:v 195 \
  -c:v libx264 -pix_fmt yuv420p -crf 18 "$OUT/sim_hero.mp4"
# what Kiro sees on its first look, and a raw frame for the three-layer diagram
cp "$EP/frames/call01_agentview.png" "$OUT/marked_front.png"
cp "$EP/frames/call01_wrist.png" "$OUT/marked_wrist.png"
cp "$EP/rec/000100.jpg" "$OUT/sim_frame.jpg"
# the butter grasp's source demo (LIBERO-Object demo 26), as served to the page
curl -sf -o "$OUT/demo_butter.jpg" \
  "$URL/demo?seg=libero_object/pick_up_the_butter_and_place_it_in_the_basket/demo_26/grasp0"
# UR5e: "put the wine bottle on top of the cabinet", scripted program + SAM 3 + Panda demos (runner --scripted --record)
ffmpeg -v error -y -framerate 20 -i "$RUNS/ur5e_clip_wine/ep_000/rec/%06d.jpg" -vf "tpad=stop_mode=clone:stop_duration=4" \
  -c:v libx264 -pix_fmt yuv420p -crf 18 "$OUT/ur5e_wine.mp4"
ls -la "$OUT"
