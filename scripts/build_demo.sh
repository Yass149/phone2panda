#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root_dir"

ffmpeg_bin="$(.venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/phone2panda-demo.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT

common=(-an -sn -dn -c:v libx264 -preset medium -crf 24 -pix_fmt yuv420p -r 30 \
  -map_metadata -1 -map_chapters -1 -fflags +bitexact -flags:v +bitexact)
font="/System/Library/Fonts/Supplemental/Arial.ttf"
bold="/System/Library/Fonts/Supplemental/Arial Bold.ttf"

render_card() {
  local output="$1" duration="$2" title="$3" subtitle="$4"
  "$ffmpeg_bin" -loglevel error -y -f lavfi -i "color=c=0x0f172a:s=960x540:d=${duration}:r=30" \
    -vf "drawtext=fontfile='${bold}':text='${title}':fontcolor=white:fontsize=46:x=(w-text_w)/2:y=190,drawtext=fontfile='${font}':text='${subtitle}':fontcolor=0x94a3b8:fontsize=25:x=(w-text_w)/2:y=265" \
    "${common[@]}" "$output"
}

render_video() {
  local input="$1" output="$2" duration="$3" title="$4" subtitle="$5"
  "$ffmpeg_bin" -loglevel error -y -i "$input" -t "$duration" \
    -vf "scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:0x0f172a,tpad=stop_mode=clone:stop_duration=6,drawbox=x=0:y=0:w=iw:h=72:color=0x0f172a@0.86:t=fill,drawtext=fontfile='${bold}':text='${title}':fontcolor=white:fontsize=28:x=28:y=12,drawtext=fontfile='${font}':text='${subtitle}':fontcolor=0xcbd5e1:fontsize=18:x=28:y=45" \
    "${common[@]}" "$output"
}

render_image() {
  local input="$1" output="$2" duration="$3" title="$4" subtitle="$5"
  "$ffmpeg_bin" -loglevel error -y -loop 1 -i "$input" -t "$duration" \
    -vf "scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:white,drawbox=x=0:y=0:w=iw:h=72:color=0x0f172a@0.90:t=fill,drawtext=fontfile='${bold}':text='${title}':fontcolor=white:fontsize=28:x=28:y=12,drawtext=fontfile='${font}':text='${subtitle}':fontcolor=0xcbd5e1:fontsize=18:x=28:y=45" \
    "${common[@]}" "$output"
}

render_card "$work_dir/00.mp4" 5 "Phone2Panda" "Can one phone demonstration shape safe robot motion?"
render_video media/phone_demo_sanitized.mp4 "$work_dir/01.mp4" 9 \
  "1  Phone demonstration" "Silent, metadata-stripped, privacy-reviewed derived clip"
render_image results/dataset_quality/plots/ep_001.png "$work_dir/02.mp4" 9 \
  "2  Detected + calibrated trajectory" "Homography-normalized left route with footprint clearance"
render_video results/phase4e/representative_calibrated_success.mp4 "$work_dir/03.mp4" 9 \
  "3  Human-derived DMP" "Route/confidence teacher · calibrated 40 mm obstacle"
render_video results/phase5b/representative_success.mp4 "$work_dir/04.mp4" 9 \
  "4  Compact GRU" "24 hidden units · distilled from human-derived DMP actions"
render_video results/phase4e/representative_calibrated_failure.mp4 "$work_dir/05.mp4" 9 \
  "5  Straight-line failure" "Expected baseline collision · no human route geometry"
render_image results/phase6/ablation_plot.png "$work_dir/06.mp4" 9 \
  "6  Fixed-seed ablations" "50 calibrated scenarios per condition · dashed line = 90 percent gate"
render_card "$work_dir/07.mp4" 11 \
  "DMP 50/50   ·   GRU 50/50   ·   straight 0/50" \
  "Human path + calibration + route filtering preserved the safety margin"

for segment in "$work_dir"/*.mp4; do
  printf "file '%s'\n" "$segment"
done > "$work_dir/concat.txt"

"$ffmpeg_bin" -loglevel error -y -f concat -safe 0 -i "$work_dir/concat.txt" -map 0:v:0 \
  -an -sn -dn -c copy -movflags +faststart -map_metadata -1 -map_chapters -1 \
  -fflags +bitexact \
  media/phone2panda_demo.mp4

.venv/bin/python scripts/check_submission.py --demo-only
