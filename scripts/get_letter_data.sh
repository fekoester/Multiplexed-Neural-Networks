#!/usr/bin/env sh
# Download the UCI letter recognition data set (20000 rows, 16 features, 26 letters) into the current directory.
set -e
url="https://archive.ics.uci.edu/static/public/59/letter+recognition.zip"
tmp=$(mktemp -d)
curl -fsSL -o "$tmp/letter.zip" "$url"
unzip -o -q "$tmp/letter.zip" letter-recognition.data -d .
rm -rf "$tmp"
echo "letter-recognition.data: $(wc -l < letter-recognition.data) rows"
