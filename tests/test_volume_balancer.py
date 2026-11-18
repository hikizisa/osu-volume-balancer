import tempfile
from pathlib import Path
import unittest

import numpy as np
import scipy.io.wavfile as wavfile

from osu_volume_balancer import (
    AudioLoudnessProfile,
    compute_masking_curve,
    estimate_sample_kit_ceiling,
    query_target_volume,
    parse_timing_points,
    balance_timing_volumes,
    get_audio_filename_from_osu,
)


class TestVolumeBalancer(unittest.TestCase):
    def test_query_target_volume_scaling(self):
        # Create a synthetic loudness curve: frame 0 = quiet (-25 dB), frame 50 = chorus (-10 dB)
        db_curve = np.full(100, -25.0, dtype=np.float32)
        db_curve[40:80] = -10.0
        profile = AudioLoudnessProfile(
            db_curve=db_curve,
            hop_ms=20.0,
            ref_chorus_db=-10.0,
            peak_db=-9.0,
            duration_ms=2000.0,
        )

        # Quiet section: delta = -15 dB -> 85 * 10^(0.52 * -15 / 20) = 85 * 10^(-0.39) = 34.6 -> clamped 35%
        vol_quiet = query_target_volume(100.0, 300.0, False, profile, ceiling_vol=85, min_vol=35, max_vol=90)
        self.assertEqual(vol_quiet, 35)

        # Chorus section: delta = 0 dB -> 85 * 10^0 = 85%
        vol_chorus = query_target_volume(1000.0, 1200.0, False, profile, ceiling_vol=85, min_vol=35, max_vol=90)
        self.assertEqual(vol_chorus, 85)

        # Chorus + Kiai boost (+1.2 dB) -> 85 * 10^(1.2/20) = 97 -> clamped to max_vol (90%)
        vol_kiai = query_target_volume(1000.0, 1200.0, True, profile, ceiling_vol=85, min_vol=35, max_vol=90)
        self.assertEqual(vol_kiai, 90)

    def test_timing_points_parsing_and_invariants(self):
        raw_lines = [
            "// Timing Points Section",
            "1000,333.333333333333,4,2,0,70,1,0",     # 180 BPM red line, 4/4 meter, soft sampleSet
            "1500,-100,4,2,0,70,0,0",                # 1.0x SV green line
            "2000,-50,4,2,1,80,0,1",                 # 2.0x SV green line, sampleIndex=1, Kiai=1
        ]
        all_tps, red_tps = parse_timing_points(raw_lines)
        self.assertEqual(len(all_tps), 3)
        self.assertEqual(len(red_tps), 1)

        self.assertTrue(red_tps[0].is_red)
        self.assertAlmostEqual(red_tps[0].beat_length, 333.333333333333)
        self.assertEqual(red_tps[0].meter, 4)

        self.assertFalse(all_tps[1].is_red)
        self.assertEqual(all_tps[1].beat_length, -100.0)

        self.assertTrue(all_tps[2].is_kiai)
        self.assertEqual(all_tps[2].sample_index, 1)

    def test_balance_timing_volumes_preserves_sv_and_inserts_at_measure(self):
        sample_osu = """osu file format v14

[General]
AudioFilename: audio.mp3
Mode: 0

[TimingPoints]
1000,500.0,4,2,1,50,1,0
1500,-66.666666666667,4,2,1,50,0,0
// Midpoint comment
6000,-100,4,2,1,50,0,1

[HitObjects]
256,192,1000,1,0,0:0:0:0:
256,192,8000,1,0,0:0:0:0:
"""
        # Create a mock profile where 1000-4000ms is quiet (-20 dB) and 4000-9000ms is loud (-8 dB)
        db_curve = np.full(500, -22.0, dtype=np.float32)
        db_curve[200:] = -8.0  # 4000ms onwards
        profile = AudioLoudnessProfile(
            db_curve=db_curve,
            hop_ms=20.0,
            ref_chorus_db=-8.0,
            peak_db=-7.0,
            duration_ms=10000.0,
        )

        new_text, summary = balance_timing_volumes(
            osu_text=sample_osu,
            profile=profile,
            min_vol=35,
            max_vol=90,
            allow_insert=True,
            shift_threshold=10,
        )

        self.assertTrue(summary["changed"])
        self.assertEqual(summary["existing_tps_updated"], 3)

        # Verify comments are preserved
        self.assertIn("// Midpoint comment", new_text)

        # Parse output timing points
        new_lines = new_text.splitlines()
        tp_lines = [ln.strip() for ln in new_lines if ln.strip() and not ln.strip().startswith("//") and "," in ln]
        out_all, out_red = parse_timing_points(tp_lines)

        # Red line at 1000ms must still have beatLength=500.0, meter=4, uninherited=1
        self.assertEqual(out_red[0].time, 1000.0)
        self.assertEqual(out_red[0].beat_length, 500.0)
        self.assertEqual(out_red[0].meter, 4)
        self.assertEqual(out_red[0].uninherited, 1)

        # Green line at 1500ms must preserve SV -66.666666666667
        sv_tp = [tp for tp in out_all if abs(tp.time - 1500.0) < 1e-3][0]
        self.assertAlmostEqual(sv_tp.beat_length, -66.666666666667, places=4)
        self.assertEqual(sv_tp.uninherited, 0)

        # Green line at 6000ms must preserve Kiai flag
        kiai_tp = [tp for tp in out_all if abs(tp.time - 6000.0) < 1e-3][0]
        self.assertTrue(kiai_tp.is_kiai)
        self.assertGreaterEqual(kiai_tp.volume, 80)

    def test_audio_masking_bandpass_computation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test_tone.wav"
            sr = 44100
            # 2 seconds of 1 kHz tone (in the masking passband)
            t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
            sine = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
            wavfile.write(str(wav_path), sr, sine)

            profile = compute_masking_curve(wav_path, hop_ms=20.0)
            self.assertGreater(profile.duration_ms, 1900.0)
            self.assertGreater(profile.peak_db, -20.0)
            self.assertGreater(len(profile.db_curve), 50)


if __name__ == "__main__":
    unittest.main()
