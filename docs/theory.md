# Theory

## Beamforming in one line

Beamforming applies complex weights across an array so signals add constructively in a desired direction and destructively elsewhere.

## Main ideas

- `theta` and `phi` define the observation or steering direction
- `d/lambda` controls electrical spacing between elements
- complex weights set amplitude and phase per element
- tapering trades narrower beams for lower sidelobes
- spacing above `lambda/2` can create grating lobes

## Models used here

- Far-field: incoming waves are treated as plane waves
- Near-field: the phase depends on element-to-focus distance
- Wideband: fixed phase-shifter weights cause beam squint away from the center frequency
- Impaired arrays: element patterns and coupling distort the ideal response

## Practical takeaway

For most scans, start with:

- `spacing_lambda = 0.5`
- uniform or Hamming taper
- far-field model unless focusing at a finite range
