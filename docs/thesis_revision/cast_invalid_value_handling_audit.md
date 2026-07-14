# CAST invalid-value handling audit

Audit date: 2026-07-14

## Question

This audit determines whether the 14 negative values in `CAST.mat/Zc` were historically masked, replaced, clipped, interpolated, or passed into the severity and FFT labels used by the thesis.

## Raw-data findings

`Zc` is a finite `float32` array with shape `180 x 24750`:

- finite values: 4,455,000;
- NaN: 0;
- Inf: 0;
- negative values: 14;
- minimum: -6.0561199;
- maximum: 18.3743401.

The positions below use zero-based matrix indices. “Relative bin angle” is `azimuth_index x 2 degrees`; it is not an absolute geographic or tool-reference azimuth.

| azimuth index | relative bin angle | depth index | depth (ft) | Zc |
|---:|---:|---:|---:|---:|
| 4 | 8 deg | 6936 | 4898.053333 | -6.0561199 |
| 121 | 242 deg | 24186 | 2470.436667 | -0.0599586 |
| 121 | 242 deg | 24187 | 2470.303333 | -0.0439226 |
| 122 | 244 deg | 23944 | 2504.845000 | -0.1599333 |
| 122 | 244 deg | 24187 | 2470.303333 | -0.0870927 |
| 123 | 246 deg | 24185 | 2470.578333 | -0.0203372 |
| 123 | 246 deg | 24189 | 2470.053333 | -0.0387477 |
| 123 | 246 deg | 24197 | 2469.003333 | -0.0887403 |
| 124 | 248 deg | 23944 | 2504.845000 | -0.8556783 |
| 124 | 248 deg | 24185 | 2470.578333 | -0.0903720 |
| 124 | 248 deg | 24187 | 2470.303333 | -0.0185556 |
| 128 | 256 deg | 24141 | 2476.411667 | -0.0246200 |
| 129 | 258 deg | 23944 | 2504.845000 | -0.6318564 |
| 132 | 264 deg | 23944 | 2504.845000 | -0.0725139 |

All 14 values lie outside the thesis's 2732--4132 ft main interval.

## Historical code behavior

The label-producing branch is `origin/percentage_label+FFT` at commit `7ba021cfa6eacd148247258ee28b8527dbbc6c92`.

`src/data_processing/main_preprocess.py`:

1. reads the complete raw `Zc` array;
2. sorts it by CAST depth;
3. performs linear interpolation onto `np.arange(2732, 4132, 0.1)`;
4. writes depth-window slices into the ground-truth HDF5 database.

The code contains no negative-value mask, replacement, clip, or validity test before interpolation.

`src/data_processing/create_tfrecords.py` then computes, for each selected depth row:

```text
severity_map = maximum(0, 2.5 - zc_row)
fft_result = fft(severity_map)
magnitude = abs(fft_result)
label = log(1 + magnitude[:30])
```

The label function also contains no negative-value mask.

## Impact on thesis labels

Although the historical code did not mask negative values, the target interval contains no raw negative `Zc` value. Reproducing the historical interpolation on the 0.1 ft target grid produced:

- interpolated grid shape: `14000 x 180`;
- interpolated negative cells: 0;
- interpolated minimum: 0.5635798;
- ring-03 unique sonic samples checked: 2,846;
- path windows containing a negative cell: 0;
- FFT-label entries affected by a negative cell: 0.

Therefore, the 14 negative raw values did **not** enter the one-dimensional percentage labels, severity maps, FFT labels, or training records used for the thesis. No experiment metric needs correction, and no training was rerun.

## Thesis wording rule

The manuscript must not claim that the historical pipeline masked the negative values. The accurate statement is:

> Historical code did not explicitly mask the 14 invalid negative values, but all of them lie outside the 2732--4132 ft main interval and therefore did not enter the labels used in this study.

If future work expands the depth interval, negative values must be explicitly masked before interpolation and label construction, with a documented replacement or exclusion policy.
