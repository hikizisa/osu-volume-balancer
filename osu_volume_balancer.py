#!/usr/bin/env python3
"""osu! Standard Hitsound Volume Balancer.

A standalone, AI-free, machine-learning-free tool that automatically balances
osu!standard hitsound volumes across song sections and measures using psychoacoustic
masking and human auditory perception principles.

Key principles:
1. Psychoacoustic masking: Transients of hitsounds (claps, whistles, finishes, clicks)
   are primarily masked by mid/high-frequency energy (350 Hz - 6,500 Hz), where vocals,
   guitars, snare crack, synth chords, and cymbals reside. Sub-bass (<250 Hz) does not
   mask crisp hitsounds in the human ear.
2. Stevens' Power Law (Audibility Transfer): To keep hitsounds at an equal perceived
   subjective prominence across dynamics, hitsound volume scales logarithmically
   (~0.52 dB hitsound change per 1.0 dB music masking change).
3. Sample Kit Calibration: Automatically measures custom hitsound samples in the beatmap
   folder to calibrate the chorus volume ceiling (78% for hot samples, 85% typical, 90% soft).
4. Strictly Safe Timing Rewriting: Updates [TimingPoints] volumes and inserts measure-aligned
   inherited green lines on sustained section transitions without ever altering BPM,
   meter, slider velocity (SV), custom sample sets, sample indices, or Kiai flags.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple
import zipfile

import numpy as np
import scipy.signal as signal
import soundfile as sf


@dataclass(frozen=True)
class AudioLoudnessProfile:
    """Represents the psychoacoustic masking loudness curve over time."""
    db_curve: np.ndarray      # Smoothed masking band loudness in dBFS per frame
    hop_ms: float             # Frame step in milliseconds (e.g., 20.0 ms)
    ref_chorus_db: float      # Reference chorus loudness (e.g. 88th percentile of active music)
    peak_db: float            # Maximum active loudness
    duration_ms: float        # Total song duration in ms


@dataclass
class TimingPoint:
    """Represents a single line in an osu! [TimingPoints] section."""
    time: float
    beat_length: float
    meter: int
    sample_set: int
    sample_index: int
    volume: int
    uninherited: int          # 1 = uninherited (red line / BPM), 0 = inherited (green line / SV)
    effects: int              # bitmask: 1 = kiai mode, 8 = omit first bar line

    @property
    def is_kiai(self) -> bool:
        return bool(self.effects & 1)

    @property
    def is_red(self) -> bool:
        return self.uninherited == 1

    def to_line(self) -> str:
        # Preserve integer formatting for time when integer-valued
        t_str = str(int(round(self.time))) if abs(self.time - round(self.time)) < 1e-4 else f"{self.time:.3f}"
        bl_str = str(int(round(self.beat_length))) if abs(self.beat_length - round(self.beat_length)) < 1e-4 else f"{self.beat_length:.12g}"
        return f"{t_str},{bl_str},{self.meter},{self.sample_set},{self.sample_index},{int(self.volume)},{self.uninherited},{self.effects}"


def read_audio_mono(audio_path: Path, target_sr: int = 44100) -> Tuple[np.ndarray, int]:
    """Read an audio file (.mp3, .ogg, .wav, .flac) as a 1D float32 mono numpy array."""
    try:
        y, sr = sf.read(str(audio_path), dtype="float32")
        mono = np.mean(y, axis=1) if y.ndim > 1 else y
        return mono.astype(np.float32), sr
    except Exception:
        # Fallback to ffmpeg if soundfile fails on non-standard formats
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                f"Failed to read audio file '{audio_path}' with soundfile, and ffmpeg was not found. "
                f"Please ensure soundfile>=0.12.0 is installed or ffmpeg is on your system PATH."
            )
        cmd = [
            ffmpeg, "-nostdin", "-threads", "0", "-i", str(audio_path),
            "-f", "f32le", "-ac", "1", "-ar", str(target_sr), "-"
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        mono = np.frombuffer(proc.stdout, dtype=np.float32)
        return mono, target_sr


def compute_masking_curve(
    audio_path: Path,
    hop_ms: float = 20.0,
    smoothing_sec: float = 1.2,
    low_cutoff: float = 350.0,
    high_cutoff: float = 6500.0,
) -> AudioLoudnessProfile:
    """Compute psychoacoustic mid-high masking loudness curve from song audio."""
    mono, sr = read_audio_mono(audio_path)
    if len(mono) == 0:
        return AudioLoudnessProfile(
            db_curve=np.array([-30.0], dtype=np.float32),
            hop_ms=hop_ms,
            ref_chorus_db=-15.0,
            peak_db=-10.0,
            duration_ms=0.0,
        )

    # 4th-order Butterworth bandpass filter isolating 350 - 6500 Hz masking band
    sos = signal.butter(4, [low_cutoff, high_cutoff], btype="bandpass", fs=sr, output="sos")
    filtered = signal.sosfilt(sos, mono)

    hop_samples = max(1, int(round(sr * (hop_ms / 1000.0))))
    n_frames = len(filtered) // hop_samples
    if n_frames == 0:
        return AudioLoudnessProfile(
            db_curve=np.array([-30.0], dtype=np.float32),
            hop_ms=hop_ms,
            ref_chorus_db=-15.0,
            peak_db=-10.0,
            duration_ms=len(mono) * 1000.0 / sr,
        )

    framed = filtered[:n_frames * hop_samples].reshape(n_frames, hop_samples)
    power = np.mean(framed ** 2, axis=1).astype(np.float64)

    # Smooth with a Hann window (default 1.2s) to capture section dynamics without single-beat jitter
    win_frames = max(3, int(round((smoothing_sec * 1000.0) / hop_ms)))
    kernel = np.hanning(win_frames)
    kernel /= max(np.sum(kernel), 1e-9)
    smooth_power = np.convolve(power, kernel, mode="same")

    db_curve = (10.0 * np.log10(np.maximum(smooth_power, 1e-10))).astype(np.float32)

    # Identify active musical frames (within 26 dB of 99th percentile peak)
    pk_db = float(np.percentile(db_curve, 99))
    active = db_curve[db_curve >= pk_db - 26.0]
    ref_chorus_db = float(np.percentile(active, 88)) if len(active) >= 10 else pk_db

    duration_ms = len(mono) * 1000.0 / sr
    return AudioLoudnessProfile(
        db_curve=db_curve,
        hop_ms=hop_ms,
        ref_chorus_db=ref_chorus_db,
        peak_db=pk_db,
        duration_ms=duration_ms,
    )


def estimate_sample_kit_ceiling(folder_path: Optional[Path], active_indices: Optional[Sequence[int]] = None) -> int:
    """Calibrate chorus volume ceiling (78..90%) based on the RMS of custom sample files in the beatmap directory."""
    if folder_path is None or not folder_path.is_dir():
        return 85

    idx_set = {i for i in active_indices if i > 0} if active_indices else {1}
    if not idx_set:
        idx_set = {1}

    rms_vals: List[float] = []
    prefixes = (
        "normal-hitnormal", "soft-hitclap", "normal-hitclap",
        "soft-hitwhistle", "normal-hitwhistle", "soft-hitfinish", "normal-hitfinish"
    )
    for idx in idx_set:
        suf = "" if idx == 1 else str(idx)
        for prefix in prefixes:
            for ext in (".wav", ".ogg", ".mp3"):
                cand = folder_path / f"{prefix}{suf}{ext}"
                if idx == 1 and not cand.is_file():
                    cand = folder_path / f"{prefix}1{ext}"
                if cand.is_file():
                    try:
                        y, sr = sf.read(str(cand), dtype="float32")
                        mono = np.mean(y, axis=1) if y.ndim > 1 else y
                        pk = float(np.max(np.abs(mono))) if len(mono) else 0.0
                        if pk < 0.02:
                            continue
                        onset = int(np.argmax(np.abs(mono) >= 0.15 * pk))
                        seg = mono[onset: min(len(mono), onset + max(256, int(0.12 * sr)))]
                        if len(seg) >= 64:
                            rms_vals.append(float(np.sqrt(np.mean(seg ** 2))))
                    except Exception:
                        pass
                    break

    if not rms_vals:
        return 85

    med_rms = float(np.median(rms_vals))
    # Hot kits (RMS >= 0.25) -> 78-80%; typical kits (0.12..0.24) -> 82-85%; soft kits (<0.12) -> 90%
    if med_rms >= 0.25:
        return 78
    if med_rms >= 0.19:
        return 80
    if med_rms >= 0.12:
        return 85
    return 90


def query_target_volume(
    time_ms: float,
    span_end_ms: float,
    is_kiai: bool,
    profile: AudioLoudnessProfile,
    ceiling_vol: int = 85,
    min_vol: int = 35,
    max_vol: int = 90,
    step: int = 5,
    slope: float = 0.52,
    kiai_boost_db: float = 1.2,
) -> int:
    """Calculate target hitsound volume using Stevens' equal-audibility power law."""
    db_curve = profile.db_curve
    hop = max(profile.hop_ms, 1.0)
    i0 = max(0, min(len(db_curve) - 1, int(round(time_ms / hop))))
    i1 = max(i0 + 1, min(len(db_curve), int(round(span_end_ms / hop))))

    # 65th percentile across the span: sustained accompaniment sets baseline, transient peaks lift it
    seg_db = float(np.percentile(db_curve[i0:i1], 65))
    delta_db = float(np.clip(seg_db - profile.ref_chorus_db, -24.0, 3.0))

    # Stevens' power law transfer (~0.52 dB hitsound per 1.0 dB music masking) + Kiai boost
    hs_db = slope * delta_db + (kiai_boost_db if is_kiai else 0.0)
    raw_vol = float(ceiling_vol) * (10.0 ** (hs_db / 20.0))

    # Quantize to step
    quantized = int(step * round(np.clip(raw_vol, float(min_vol), float(max_vol)) / step))
    return max(min_vol, min(max_vol, quantized))


def parse_timing_points(raw_lines: Sequence[str]) -> Tuple[List[TimingPoint], List[TimingPoint]]:
    """Parse raw lines of [TimingPoints] into (all_timing_points, uninherited_red_points)."""
    all_tps: List[TimingPoint] = []
    red_tps: List[TimingPoint] = []
    for line in raw_lines:
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        parts = s.split(",")
        if len(parts) < 8:
            continue
        try:
            t = float(parts[0])
            bl = float(parts[1])
            meter = int(parts[2])
            sset = int(parts[3])
            sidx = int(parts[4])
            vol = int(parts[5])
            uninh = int(parts[6])
            eff = int(parts[7])
            tp = TimingPoint(
                time=t,
                beat_length=bl,
                meter=meter,
                sample_set=sset,
                sample_index=sidx,
                volume=vol,
                uninherited=uninh,
                effects=eff,
            )
            all_tps.append(tp)
            if uninh == 1:
                red_tps.append(tp)
        except ValueError:
            continue

    all_tps.sort(key=lambda p: (p.time, -p.uninherited))
    red_tps.sort(key=lambda p: p.time)
    return all_tps, red_tps


def find_active_timing(time_ms: float, all_tps: Sequence[TimingPoint], red_tps: Sequence[TimingPoint]) -> Tuple[TimingPoint, TimingPoint]:
    """Find (active_inherited_or_uninherited, active_parent_red) at given time_ms."""
    red = red_tps[0]
    for r in red_tps:
        if r.time <= time_ms + 2.0:
            red = r
        else:
            break

    act = all_tps[0]
    for tp in all_tps:
        if tp.time <= time_ms + 2.0:
            act = tp
        else:
            break
    return act, red


def balance_timing_volumes(
    osu_text: str,
    profile: AudioLoudnessProfile,
    samples_dir: Optional[Path] = None,
    min_vol: int = 35,
    max_vol: int = 90,
    step: int = 5,
    slope: float = 0.52,
    kiai_boost_db: float = 1.2,
    shift_threshold: int = 10,
    allow_insert: bool = True,
) -> Tuple[str, Dict[str, object]]:
    """Update [TimingPoints] volumes in osu_text based on audio loudness profile."""
    lines = osu_text.splitlines(keepends=True)
    eol = "\r\n" if "\r\n" in osu_text else "\n"

    sec = ""
    tp_start = tp_end = -1
    tp_raw_lines: List[str] = []
    obj_times: List[float] = []

    for idx, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("[") and s.endswith("]"):
            if sec == "TimingPoints":
                tp_end = idx
            sec = s[1:-1]
            if sec == "TimingPoints":
                tp_start = idx + 1
            continue
        if sec == "TimingPoints" and s and not s.startswith("//"):
            tp_raw_lines.append(s)
        elif sec == "HitObjects" and s and not s.startswith("//"):
            parts = s.split(",")
            if len(parts) >= 3:
                try:
                    obj_times.append(float(parts[2]))
                except ValueError:
                    pass

    if tp_start < 0:
        return osu_text, {"changed": False, "reason": "No [TimingPoints] section found"}
    if tp_end < 0:
        tp_end = len(lines)

    all_tps, red_tps = parse_timing_points(tp_raw_lines)
    if not all_tps or not red_tps:
        return osu_text, {"changed": False, "reason": "No valid timing points found"}

    active_indices = sorted({tp.sample_index for tp in all_tps})
    ceiling_vol = estimate_sample_kit_ceiling(samples_dir, active_indices)

    first_obj_ms = min(obj_times) if obj_times else 0.0
    last_obj_ms = max(obj_times) if obj_times else float(profile.duration_ms)

    # Parse raw rows to retain exact original text tokens
    parsed_rows: List[List[str]] = []
    existing_times: List[float] = []
    for raw in tp_raw_lines:
        p = raw.split(",")
        if len(p) >= 8:
            parsed_rows.append(p[:8])
            existing_times.append(float(p[0]))

    # Step 1: Update volumes on existing timing points
    for i, row in enumerate(parsed_rows):
        t_ms = float(row[0])
        t_next = max(t_ms + 1500.0, float(parsed_rows[i + 1][0]) if i + 1 < len(parsed_rows) else t_ms + 2500.0)
        t_next = min(t_next, t_ms + 4000.0)
        kiai = bool(int(row[7]) & 1)
        vol = query_target_volume(
            t_ms, t_next, kiai, profile,
            ceiling_vol=ceiling_vol, min_vol=min_vol, max_vol=max_vol,
            step=step, slope=slope, kiai_boost_db=kiai_boost_db
        )
        row[5] = str(vol)

    inserted_rows: List[List[str]] = []
    working_rows = sorted(parsed_rows, key=lambda r: (float(r[0]), -int(r[6])))

    # Step 2: Measure-grid downbeat insertion (if enabled)
    if allow_insert:
        measure_grid: List[float] = []
        for r_i, r_tp in enumerate(red_tps):
            t_cur = float(r_tp.time)
            t_lim = float(red_tps[r_i + 1].time) - 10.0 if r_i + 1 < len(red_tps) else (last_obj_ms + 500.0)
            bar_ms = max(250.0, float(r_tp.beat_length) * max(1, int(r_tp.meter)))
            while t_cur < first_obj_ms - bar_ms:
                t_cur += bar_ms
            while t_cur <= t_lim:
                if first_obj_ms - bar_ms <= t_cur <= last_obj_ms + 200.0:
                    if not any(abs(t_cur - et) <= 35.0 for et in existing_times):
                        measure_grid.append(round(t_cur))
                t_cur += bar_ms

        def _active_row_at(t_ms: float, rows: List[List[str]]) -> List[str]:
            best = rows[0]
            for r in rows:
                if float(r[0]) <= t_ms + 2.0:
                    best = r
                else:
                    break
            return best

        last_insert_ms = -1e9
        for m_ms in measure_grid:
            act = _active_row_at(m_ms, working_rows)
            cur_vol = int(act[5])
            kiai = bool(int(act[7]) & 1)
            _, red = find_active_timing(m_ms, all_tps, red_tps)
            bar_ms = max(400.0, float(red.beat_length) * max(1, int(red.meter))) if red else 1500.0

            # Guard against rapid toggling (require at least 1.8 bars between inserted changes)
            if m_ms - last_insert_ms < 1.8 * bar_ms:
                continue

            v_now = query_target_volume(
                m_ms, m_ms + bar_ms, kiai, profile,
                ceiling_vol=ceiling_vol, min_vol=min_vol, max_vol=max_vol,
                step=step, slope=slope, kiai_boost_db=kiai_boost_db
            )
            v_sust = query_target_volume(
                m_ms + bar_ms, m_ms + 2.0 * bar_ms, kiai, profile,
                ceiling_vol=ceiling_vol, min_vol=min_vol, max_vol=max_vol,
                step=step, slope=slope, kiai_boost_db=kiai_boost_db
            )

            # Require sustained volume shift >= shift_threshold
            diff_now = v_now - cur_vol
            diff_sust = v_sust - cur_vol
            if abs(diff_now) >= shift_threshold and abs(diff_sust) >= shift_threshold and (diff_now * diff_sust > 0):
                inh_sv = act[1] if int(act[6]) == 0 else "-100"
                t_str = str(int(round(m_ms)))
                # Format: time, beatLength(SV), meter, sampleSet, sampleIndex, volume, uninherited=0, effects
                new_r = [t_str, inh_sv, act[2], act[3], act[4], str(v_now), "0", act[7]]
                inserted_rows.append(new_r)
                working_rows.append(new_r)
                working_rows.sort(key=lambda r: (float(r[0]), -int(r[6])))
                last_insert_ms = m_ms

    final_tp_lines = [",".join(r) + eol for r in working_rows]
    comment_lines = [ln for ln in lines[tp_start:tp_end] if ln.strip().startswith("//")]
    has_trailing_blank = tp_end > tp_start and lines[tp_end - 1].strip() == ""
    replacement = comment_lines + final_tp_lines + ([eol] if has_trailing_blank else [])
    new_text = "".join(lines[:tp_start] + replacement + lines[tp_end:])

    vols_out = [int(r[5]) for r in working_rows]
    summary = {
        "changed": True,
        "ceiling_vol": ceiling_vol,
        "min_vol": min(vols_out) if vols_out else min_vol,
        "median_vol": int(np.median(vols_out)) if vols_out else ceiling_vol,
        "max_vol": max(vols_out) if vols_out else max_vol,
        "existing_tps_updated": len(parsed_rows),
        "measure_tps_inserted": len(inserted_rows),
        "total_tps": len(working_rows),
    }
    return new_text, summary


def get_audio_filename_from_osu(osu_text: str) -> Optional[str]:
    """Extract AudioFilename from [General] section of .osu text."""
    for line in osu_text.splitlines():
        s = line.strip()
        if s.startswith("AudioFilename:"):
            return s.split(":", 1)[1].strip()
    return None


def process_osu_file(
    osu_path: Path,
    out_path: Optional[Path] = None,
    audio_path_override: Optional[Path] = None,
    backup: bool = True,
    dry_run: bool = False,
    min_vol: int = 35,
    max_vol: int = 90,
    step: int = 5,
    slope: float = 0.52,
    kiai_boost_db: float = 1.2,
    shift_threshold: int = 10,
    allow_insert: bool = True,
) -> Dict[str, object]:
    """Process a single .osu beatmap file."""
    osu_text = osu_path.read_text(encoding="utf-8-sig", errors="replace")

    # Locate song audio
    audio_file = audio_path_override
    if audio_file is None:
        audio_name = get_audio_filename_from_osu(osu_text)
        if audio_name:
            cand = osu_path.parent / audio_name
            if cand.is_file():
                audio_file = cand

    if audio_file is None or not audio_file.is_file():
        # Look for any audio file in folder
        for ext in (".mp3", ".ogg", ".wav"):
            for cand in osu_path.parent.glob(f"*{ext}"):
                if cand.is_file() and not cand.name.startswith("virtual_"):
                    audio_file = cand
                    break
            if audio_file:
                break

    if audio_file is None or not audio_file.is_file():
        raise FileNotFoundError(f"Could not locate song audio for '{osu_path.name}'.")

    # Compute masking curve
    profile = compute_masking_curve(audio_file)

    # Balance timing volumes
    new_text, summary = balance_timing_volumes(
        osu_text=osu_text,
        profile=profile,
        samples_dir=osu_path.parent,
        min_vol=min_vol,
        max_vol=max_vol,
        step=step,
        slope=slope,
        kiai_boost_db=kiai_boost_db,
        shift_threshold=shift_threshold,
        allow_insert=allow_insert,
    )

    summary["file"] = osu_path.name
    summary["audio_file"] = audio_file.name

    if dry_run:
        summary["dry_run"] = True
        return summary

    target_out = out_path if out_path is not None else osu_path
    if target_out == osu_path and backup:
        bak_path = osu_path.with_suffix(osu_path.suffix + ".bak")
        if not bak_path.exists():
            shutil.copy2(osu_path, bak_path)
            summary["backup"] = str(bak_path.name)

    target_out.write_text(new_text, encoding="utf-8-sig")
    summary["written_to"] = str(target_out)
    return summary


def process_directory(
    dir_path: Path,
    out_dir: Optional[Path] = None,
    backup: bool = True,
    dry_run: bool = False,
    min_vol: int = 35,
    max_vol: int = 90,
    step: int = 5,
    slope: float = 0.52,
    kiai_boost_db: float = 1.2,
    shift_threshold: int = 10,
    allow_insert: bool = True,
) -> List[Dict[str, object]]:
    """Process all .osu files in a directory."""
    osu_files = sorted(dir_path.glob("*.osu"))
    if not osu_files:
        raise FileNotFoundError(f"No .osu beatmap files found in '{dir_path}'.")

    results: List[Dict[str, object]] = []
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    for osu_file in osu_files:
        target_out = (out_dir / osu_file.name) if out_dir is not None else None
        res = process_osu_file(
            osu_path=osu_file,
            out_path=target_out,
            backup=backup,
            dry_run=dry_run,
            min_vol=min_vol,
            max_vol=max_vol,
            step=step,
            slope=slope,
            kiai_boost_db=kiai_boost_db,
            shift_threshold=shift_threshold,
            allow_insert=allow_insert,
        )
        results.append(res)
    return results


def process_osz(
    osz_path: Path,
    out_path: Optional[Path] = None,
    backup: bool = True,
    dry_run: bool = False,
    min_vol: int = 35,
    max_vol: int = 90,
    step: int = 5,
    slope: float = 0.52,
    kiai_boost_db: float = 1.2,
    shift_threshold: int = 10,
    allow_insert: bool = True,
) -> Dict[str, object]:
    """Process an osu! .osz package."""
    with tempfile.TemporaryDirectory(prefix="osu_vol_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        with zipfile.ZipFile(osz_path, "r") as zf:
            zf.extractall(tmp_path)

        results = process_directory(
            dir_path=tmp_path,
            out_dir=None,
            backup=False,
            dry_run=dry_run,
            min_vol=min_vol,
            max_vol=max_vol,
            step=step,
            slope=slope,
            kiai_boost_db=kiai_boost_db,
            shift_threshold=shift_threshold,
            allow_insert=allow_insert,
        )

        if not dry_run:
            target_out = out_path if out_path is not None else osz_path
            if target_out == osz_path and backup:
                bak_path = osz_path.with_suffix(osz_path.suffix + ".bak")
                if not bak_path.exists():
                    shutil.copy2(osz_path, bak_path)

            temp_zip = tmp_path.parent / (osz_path.stem + "_packed.zip")
            with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf_out:
                for root, _, files in os.walk(tmp_path):
                    for file in files:
                        full_f = Path(root) / file
                        rel_f = full_f.relative_to(tmp_path)
                        zf_out.write(full_f, str(rel_f))
            shutil.move(str(temp_zip), str(target_out))

    return {
        "osz": osz_path.name,
        "maps_processed": len(results),
        "results": results,
    }


def export_plot_html(
    profile: AudioLoudnessProfile,
    osu_path: Path,
    out_html: Path,
    title: str = "Hitsound Volume Curve Profile",
) -> None:
    """Generate an interactive HTML5 preview comparing music loudness and hitsound volume curves."""
    osu_text = osu_path.read_text(encoding="utf-8-sig", errors="replace")
    _, raw_summary = balance_timing_volumes(osu_text, profile, samples_dir=osu_path.parent)

    lines = osu_text.splitlines()
    all_tps, red_tps = parse_timing_points([ln for ln in lines if ln.strip() and not ln.strip().startswith("//")])

    # Sample timeline at 500ms steps
    times_sec = []
    music_db_vals = []
    vol_vals = []
    kiai_vals = []

    dur_sec = profile.duration_ms / 1000.0
    for t_s in np.arange(0.0, dur_sec, 0.5):
        t_ms = float(t_s * 1000.0)
        idx = max(0, min(len(profile.db_curve) - 1, int(round(t_ms / profile.hop_ms))))
        music_db = float(profile.db_curve[idx])
        act, _ = find_active_timing(t_ms, all_tps, red_tps) if all_tps else (None, None)
        vol = act.volume if act else 70
        is_k = 1 if (act and act.is_kiai) else 0

        times_sec.append(round(t_s, 2))
        music_db_vals.append(round(music_db, 1))
        vol_vals.append(int(vol))
        kiai_vals.append(is_k)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #12141a; color: #e1e4ea; margin: 0; padding: 24px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ margin-top: 0; font-size: 24px; color: #fff; }}
  .chart-box {{ background: #1c1f2b; padding: 20px; border-radius: 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.4); margin-bottom: 24px; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .stat-card {{ background: #1e2230; padding: 16px; border-radius: 8px; border-left: 4px solid #6366f1; }}
  .stat-card .val {{ font-size: 22px; font-weight: 700; color: #a5b4fc; }}
  .stat-card .lbl {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; margin-top: 4px; }}
</style>
</head>
<body>
<div class="container">
  <h1>{title}</h1>
  <p style="color: #94a3b8;">Map: <code>{osu_path.name}</code> | Song Duration: {dur_sec:.1f}s</p>

  <div class="stats-grid">
    <div class="stat-card" style="border-color: #3b82f6;">
      <div class="val">{raw_summary.get('min_vol', 35)}% - {raw_summary.get('max_vol', 90)}%</div>
      <div class="lbl">Dynamic Volume Range</div>
    </div>
    <div class="stat-card" style="border-color: #10b981;">
      <div class="val">{raw_summary.get('median_vol', 75)}%</div>
      <div class="lbl">Median Section Volume</div>
    </div>
    <div class="stat-card" style="border-color: #ec4899;">
      <div class="val">{raw_summary.get('ceiling_vol', 85)}%</div>
      <div class="lbl">Kit Chorus Ceiling</div>
    </div>
    <div class="stat-card" style="border-color: #f59e0b;">
      <div class="val">{raw_summary.get('measure_tps_inserted', 0)}</div>
      <div class="lbl">Measure Green Lines Inserted</div>
    </div>
  </div>

  <div class="chart-box">
    <canvas id="volChart" height="90"></canvas>
  </div>
</div>

<script>
const times = {times_sec};
const musicDbs = {music_db_vals};
const hsVols = {vol_vals};

const ctx = document.getElementById('volChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: times,
    datasets: [
      {{
        label: 'Hitsound Volume (%)',
        data: hsVols,
        borderColor: '#60a5fa',
        backgroundColor: 'rgba(96, 165, 250, 0.15)',
        fill: true,
        yAxisID: 'yVol',
        stepped: true,
        borderWidth: 2,
        pointRadius: 0
      }},
      {{
        label: 'Music Masking Loudness (dBFS)',
        data: musicDbs,
        borderColor: '#f43f5e',
        backgroundColor: 'transparent',
        yAxisID: 'yDb',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.2
      }}
    ]
  }},
  options: {{
    responsive: true,
    interaction: {{ mode: 'index', intersect: false }},
    scales: {{
      x: {{
        title: {{ display: true, text: 'Time (seconds)', color: '#94a3b8' }},
        grid: {{ color: '#2d3345' }},
        ticks: {{ color: '#94a3b8' }}
      }},
      yVol: {{
        type: 'linear',
        position: 'left',
        min: 20,
        max: 100,
        title: {{ display: true, text: 'Hitsound Volume (%)', color: '#60a5fa' }},
        grid: {{ color: '#2d3345' }},
        ticks: {{ color: '#60a5fa' }}
      }},
      yDb: {{
        type: 'linear',
        position: 'right',
        min: -35,
        max: 0,
        title: {{ display: true, text: 'Music Masking Band (dBFS)', color: '#f43f5e' }},
        grid: {{ drawOnChartArea: false }},
        ticks: {{ color: '#f43f5e' }}
      }}
    }},
    plugins: {{
      legend: {{ labels: {{ color: '#e1e4ea' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""
    out_html.write_text(html_content, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="osu! Standard Hitsound Volume Balancer (AI-Free / Model-Free)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", type=str, help="Path to a .osu file, a beatmapset folder, or a .osz archive.")
    parser.add_argument("-o", "--out", type=str, default=None, help="Output destination (file or directory). Default: in-place.")
    parser.add_argument("--min-vol", type=int, default=35, help="Minimum volume percentage (calm intros/bridges).")
    parser.add_argument("--max-vol", type=int, default=90, help="Maximum volume percentage.")
    parser.add_argument("--step", type=int, default=5, help="Volume quantization step (typically 5%%).")
    parser.add_argument("--kiai-boost", type=float, default=1.2, help="Kiai time dB audibility boost.")
    parser.add_argument("--slope", type=float, default=0.52, help="Stevens' power law audibility slope (dB hitsound per dB music).")
    parser.add_argument("--threshold", type=int, default=10, help="Minimum sustained volume shift required to insert measure green line.")
    parser.add_argument("--no-insert", action="store_true", help="Only adjust existing timing point volumes; do not insert measure lines.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak backup files when overwriting in place.")
    parser.add_argument("--dry-run", action="store_true", help="Analyze and display proposed volume adjustments without writing files.")
    parser.add_argument("--plot", type=str, default=None, help="Export an interactive HTML volume curve preview chart.")

    args = parser.parse_args(argv)
    target = Path(args.path)
    if not target.exists():
        print(f"Error: Path '{target}' does not exist.", file=sys.stderr)
        return 1

    out_p = Path(args.out) if args.out else None
    allow_insert = not args.no_insert
    backup = not args.no_backup

    try:
        if target.is_file() and target.suffix.lower() == ".osu":
            res = process_osu_file(
                osu_path=target,
                out_path=out_p,
                backup=backup,
                dry_run=args.dry_run,
                min_vol=args.min_vol,
                max_vol=args.max_vol,
                step=args.step,
                slope=args.slope,
                kiai_boost_db=args.kiai_boost,
                shift_threshold=args.threshold,
                allow_insert=allow_insert,
            )
            print(f"[Done] {res['file']}: Dynamic range {res['min_vol']}% - {res['max_vol']}% (median {res['median_vol']}%, kit ceiling {res['ceiling_vol']}%)")
            print(f"       Updated {res['existing_tps_updated']} existing timing points, inserted {res['measure_tps_inserted']} measure green lines.")
            if args.plot:
                profile = compute_masking_curve(target.parent / str(res["audio_file"]))
                export_plot_html(profile, target, Path(args.plot))
                print(f"[Plot] Exported HTML preview to '{args.plot}'")

        elif target.is_file() and target.suffix.lower() == ".osz":
            res = process_osz(
                osz_path=target,
                out_path=out_p,
                backup=backup,
                dry_run=args.dry_run,
                min_vol=args.min_vol,
                max_vol=args.max_vol,
                step=args.step,
                slope=args.slope,
                kiai_boost_db=args.kiai_boost,
                shift_threshold=args.threshold,
                allow_insert=allow_insert,
            )
            print(f"[Done] Processed {res['maps_processed']} beatmaps in '{res['osz']}'.")

        elif target.is_dir():
            results = process_directory(
                dir_path=target,
                out_dir=out_p,
                backup=backup,
                dry_run=args.dry_run,
                min_vol=args.min_vol,
                max_vol=args.max_vol,
                step=args.step,
                slope=args.slope,
                kiai_boost_db=args.kiai_boost,
                shift_threshold=args.threshold,
                allow_insert=allow_insert,
            )
            print(f"[Done] Processed {len(results)} beatmaps in '{target.name}'.")
            for r in results:
                print(f"  - {r['file']}: {r['min_vol']}%..{r['max_vol']}% (median {r['median_vol']}%, +{r['measure_tps_inserted']} measure lines)")
            if args.plot and results:
                first_osu = target / str(results[0]["file"])
                profile = compute_masking_curve(target / str(results[0]["audio_file"]))
                export_plot_html(profile, first_osu, Path(args.plot))
                print(f"[Plot] Exported HTML preview to '{args.plot}'")

        else:
            print(f"Error: Unsupported file format '{target.name}'. Supported: .osu, .osz, or beatmap folder.", file=sys.stderr)
            return 1

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
