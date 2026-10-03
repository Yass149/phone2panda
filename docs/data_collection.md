# Data collection

## Recording layout

- Fix an iPhone above a flat, evenly lit canvas; do not move it during a take.
- Keep the four custom L markers visible in the opening and closing hands-free
  windows, labelled TL, TR, BL and BR in decoded-frame orientation.
- Put the red marker on the transported black box, the blue obstacle near the
  centre, three start zones along the near edge and the target opposite them.
- Record a left or right route around the obstacle and finish with the entire
  box footprint inside the target.
- Use landscape 1920×1080 at approximately 30 fps. Finder rotation is not
  authoritative; decoded marker positions are.

## Naming and metadata

`data/metadata.csv` is the source of episode ID, filename, split, start region
and expected route. Replacement files receive a suffix and the metadata row is
updated without deleting the superseded source. Raw files are opened read-only
and hashed before and after processing.

## Quality gate

Every accepted recording must decode without dropped frames, have upright
marker order, at least 95% simultaneous four-marker visibility in both
calibration windows, at most 3 px marker jitter and 5 px opening-to-closing
drift, at least 95% usable red-marker tracking, the labelled start/route, safe
footprint clearance and final target placement. Whole-video corner visibility
is diagnostic because the shoulder may temporarily hide BL after placement.

## Privacy and publication

Raw MOV files are private and ignored by Git. Their AAC audio, phone metadata
and timed metadata are never copied into published media. Public video is
re-encoded video-only with metadata stripped; repository checks reject audio,
location/device fields, raw MOV extensions, absolute home paths and oversized
artifacts. Review frames before publication for faces, documents, reflections
or other identifying background content.
