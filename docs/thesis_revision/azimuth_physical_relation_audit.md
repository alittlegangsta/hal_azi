# Azimuth physical-relation audit

Audit date: 2026-07-14

## Source arrays and project-confirmed geometry

`D2_XSI_RelBearing_Inclination.mat` contains 13,508 depth, inclination, and relative-bearing records. The main 2732--4132 ft interval contains 5,549 records and no NaN in these arrays.

CAST contains 180 azimuth samples at 2-degree intervals, covering the full 360-degree circumference. This confirms relative bin spacing and full circumferential coverage, but it does not define the CAST azimuth-zero direction relative to the XSI tool reference axis.

## Inclination

Within 2732--4132 ft, inclination ranges from 0.1349 to 2.2879 degrees, with mean 0.5218 degrees and median 0.4705 degrees. The interval is therefore near vertical overall. Inclination describes the magnitude of wellbore/tool-axis tilt; it does not encode rotation about that axis and cannot be substituted for an XSI-to-CAST rotation angle.

## Relative Bearing

Within 2732--4132 ft, circular absolute step changes have median 5.156 degrees, 95th percentile 28.639 degrees, 99th percentile 64.611 degrees, and maximum 178.563 degrees. There are 111 steps greater than 45 degrees and 23 greater than 90 degrees. The changes are frequent and sometimes severe in the near-vertical main interval.

The current evidence does not establish whether the stored reference is magnetic north, borehole high side, another tool face, or a processed convention. The project-confirmed behavior is sufficient to show that Relative Bearing is not a stable absolute correction angle in the main near-vertical interval.

## Depth sampling

- XSI ring-03 raw depth: median unique step about 0.493 ft in the main interval.
- Orientation metadata: nominal 0.25 ft descending sampling with local reorderings or gaps.
- CAST raw depth: irregular descending sampling with target median about 0.142 ft and 9,884 records in the main interval.
- The historical project processing maps CAST to a derived 0.1 ft grid.

Depth interpolation can establish a common depth coordinate, but it cannot supply a missing common azimuth zero.

## Physical conclusion and label implication

Instrument azimuth, receiver number, borehole high side, Relative Bearing, Inclination, CAST image zero, and XSI side-channel zero are distinct quantities. Receiver geometry supplies relative 45-degree XSI spacing; CAST supplies 2-degree circumferential spacing. Neither fact supplies a unique transform between the tools' absolute azimuth zeros.

Three observations jointly motivate the label design:

1. the main interval is near vertical, with median Inclination about 0.4705 degrees;
2. Relative Bearing changes frequently and sharply in that interval;
3. no metadata establish a unique XSI-to-CAST absolute azimuth-zero transform.

The thesis therefore does not use absolute CAST azimuth position as supervision. It treats a rotation of the 180-point CAST sequence as a circular shift and uses the DFT magnitude to reduce sensitivity to that shift. The representation preserves the frequency composition of circumferential structure but discards phase and absolute azimuth position. It cannot recover the absolute channeling direction.
