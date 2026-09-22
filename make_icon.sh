#!/bin/bash
# Build drift_correct.icns from any image in this folder.
# Usage: ./make_icon.sh my_icon.png
set -euo pipefail

source_image="${1:?Usage: ./make_icon.sh <image>}"
work="$(mktemp -d)"
iconset="$work/drift_correct.iconset"
mkdir -p "$iconset"

# Pad the source out to a square before scaling, so a non-square image keeps
# its proportions instead of being stretched into the icon grid.
width=$(sips -g pixelWidth "$source_image" | awk '/pixelWidth/{print $2}')
height=$(sips -g pixelHeight "$source_image" | awk '/pixelHeight/{print $2}')
side=$(( width > height ? width : height ))
sips -p "$side" "$side" --padColor FFFFFF "$source_image" \
    --out "$work/square.png" >/dev/null

# macOS expects each size at 1x and 2x inside the iconset folder.
for size in 16 32 128 256 512; do
    sips -z "$size" "$size" "$work/square.png" \
        --out "$iconset/icon_${size}x${size}.png" >/dev/null
    sips -z "$((size * 2))" "$((size * 2))" "$work/square.png" \
        --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
done

iconutil --convert icns "$iconset" --output drift_correct.icns
rm -rf "$work"
echo "Wrote drift_correct.icns from $source_image (${width}x${height} padded to ${side}x${side})"
