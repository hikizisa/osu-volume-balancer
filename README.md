# osu! Hitsound Volume Balancer

Gemini 3.8 Flash wrote below cuz I don't have budget

---

A fast, lightweight, and **100% AI-free / machine-learning-free** tool that automatically balances hitsound volumes across sections and measures for osu!standard beatmaps (`.osu` and `.osz`).

---

## Why AI-Free?

Unlike hitsound rhythm and sample assignment (which involves complex musical pattern classification), **hitsound volume balancing is fundamentally a psychoacoustic acoustics problem**.

Human ear perception and hitsound audibility follow well-established auditory masking and psychoacoustic laws:
1. **Auditory Masking Bandpass (350 Hz – 6,500 Hz)**:
   Hitsound transients (clicks, claps, whistles, finishes) are masked primarily by mid-high frequency energy—where vocals, guitars, synth chords, piano, and snare body reside. Sub-bass (<250 Hz) vibrates speakers but does **not** mask high-frequency hitsound transients. This tool applies a 4th-order Butterworth bandpass filter across the song audio to measure true acoustic masking power over time.
2. **Stevens' Power Law of Perceived Loudness**:
   Human subjective loudness perception is non-linear. To maintain an equal perceived hitsound balance across dynamic shifts (from quiet piano verses to explosive choruses), hitsound volume must scale logarithmically (~`0.52 dB` hitsound shift per `1.0 dB` music change).
3. **Custom Sample Kit RMS Calibration**:
   Beatmaps with loud ("hot") custom samples need lower volume ceilings (e.g., 78%–80%) to avoid deafening players, while maps with quiet custom samples need higher ceilings (88%–90%). The tool inspects `.wav` files in the beatmap folder and automatically calibrates the chorus ceiling.
4. **Zero Neural Networks / Zero Weights**:
   - Runs in ~1–2 seconds for a full 5-minute song.
   - Minimal dependencies: `numpy`, `scipy`, and `soundfile` only.
   - No PyTorch, no ONNX, no CUDA, no model checkpoints, no GPU required.

---

## Strict osu! Timing Safeguards

The tool is designed specifically around official osu! ranking criteria and mapper workflows:
- **Preserves all Slider Velocities (SV)**: Inherited green lines retain their exact original SV multiplier (`beatLength`).
- **Preserves all Red Lines**: Never touches uninherited timing points (BPM, meter, time signatures).
- **Preserves Sample Sets & Indices**: Custom sample sets (`sampleSet`) and indices (`sampleIndex`) are strictly preserved.
- **Preserves Kiai Flags & Effects**: Kiai modes and barline omits are preserved; Kiai sections automatically receive a calibrated `+1.2 dB` audibility lift.
- **Measure-Grid Alignment**: Section volume changes are placed strictly on measure downbeats (bars) calculated from the parent BPM, avoiding off-beat volume glitches.
- **Hysteresis (Anti-Chatter)**: Requires sustained volume shifts (>= 10% held for at least 2 bars) before inserting a green line, completely preventing rapid volume toggling or spam.
- **Automatic Backups**: Creates `.bak` files before in-place overwriting by default.

---

## Installation

Requires Python 3.9+:

```bash
git clone https://github.com/hikizisa/osu-volume-balancer.git
cd osu-volume-balancer
pip install -r requirements.txt
```

*(Optional: `ffmpeg` can be installed on your system PATH as an automatic fallback for rare audio codecs).*

---

## Usage

### 1. Single `.osu` Beatmap
```bash
python osu_volume_balancer.py "path/to/difficulty.osu"
```

### 2. Entire Beatmapset Directory (All Difficulties)
```bash
python osu_volume_balancer.py "path/to/Songs/123456 Artist - Title"
```

### 3. `.osz` Archive
```bash
python osu_volume_balancer.py "path/to/beatmap.osz" --out "path/to/balanced.osz"
```

### 4. Interactive HTML Loudness Curve Preview
Export an interactive HTML5 chart to visualize the music's masking curve alongside the resulting hitsound volume curve across the entire song:
```bash
python osu_volume_balancer.py "path/to/difficulty.osu" --plot "preview.html"
```

---

## Command-Line Options

| Option | Default | Description |
|---|---|---|
| `path` | *(required)* | Path to `.osu` file, beatmap folder, or `.osz` package |
| `-o`, `--out` | `None` | Destination file or folder (default: modifies in place) |
| `--min-vol` | `35` | Minimum hitsound volume for quiet intros/bridges (%) |
| `--max-vol` | `90` | Maximum hitsound volume cap (%) |
| `--step` | `5` | Volume quantization step (osu! mappers typically use 5% steps) |
| `--kiai-boost` | `1.2` | Additional audibility boost during Kiai sections (dB) |
| `--slope` | `0.52` | Stevens' law transfer slope (dB hitsound per dB music) |
| `--threshold` | `10` | Minimum sustained volume difference required to insert a green line (%) |
| `--no-insert` | `False` | Only adjust existing timing points; do not insert new measure lines |
| `--no-backup` | `False` | Do not create `.bak` backup files |
| `--dry-run` | `False` | Analyze and print statistics without writing changes |
| `--plot` | `None` | Path to export interactive HTML5 volume curve visualizer |

---

## Running Tests

Run the test suite using Python's standard `unittest`:

```bash
python -m unittest discover -s tests
```

---

## License

MIT License. Feel free to use in your mapping, hitsounding, or modding tools!
