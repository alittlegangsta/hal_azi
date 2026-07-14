# Azimuth physical-relation audit

## Source arrays

`D2_XSI_RelBearing_Inclination.mat` contains 13,508 depth, inclination, and relative-bearing records. The target interval 2732--4132 ft contains 5,549 records. No NaN occurs in the three arrays.

## Inclination

Within the target interval, inclination ranges from 0.1349 to 2.2879 degrees, with median 0.4705 degrees. It describes the magnitude of wellbore/tool-axis tilt and is not an azimuthal direction. It therefore cannot be substituted for an XSI-to-CAST rotation angle.

## Relative Bearing

Within the target interval, circular absolute step changes have median 5.156 degrees, 95th percentile 28.639 degrees, 99th percentile 64.611 degrees, and maximum 178.563 degrees; 111 steps exceed 45 degrees and 23 exceed 90 degrees. In the 4132--5600 ft inclined interval the reported relative bearing is much smoother. This agrees with the project slide note that relative bearing becomes unreliable near vertical while waveform quality is better there.

The current evidence does not establish whether the stored reference is magnetic north, borehole high side, another tool face, or a processed convention. No CAST azimuth-zero metadata was found.

## Depth sampling

- XSI group-03 raw depth: median unique step about 0.493 ft in the target interval.
- Orientation metadata: nominal 0.25 ft descending sampling with local reorderings/gaps.
- CAST raw depth: irregular descending sampling with target median about 0.142 ft.
- The project processing aligns data on a derived 0.1 ft grid.

## Conclusion

Instrument azimuth, receiver number, borehole high side, relative bearing, inclination, CAST image zero, and XSI side-channel zero are distinct quantities. Receiver geometry supplies relative 45-degree spacing but not a shared absolute origin. The available metadata cannot establish a unique transform between the two tools' absolute azimuth zeros. Direct bin-to-channel supervision could therefore introduce an unknown cyclic offset.

The physically defensible response is to treat an azimuthal rotation of the discrete CAST sequence as a circular shift and use the DFT magnitude, while explicitly accepting the loss of absolute orientation. No unsupported absolute-angle formula is introduced.
