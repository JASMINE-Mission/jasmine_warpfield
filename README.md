# JASMINE image-warp demonstration code
An experimental code to simulate the image warp function.


## Installation
Clone this.

### Dependencies

- numpy
- scipy
- astropy
- astroquery


## Usage

- Input
    - source list
    - optics parameters
    - distortion function
- Output
    - source list (ra, dec, mag)
    - detector center (x, y)


## Structure

- Source => astropy.coordinate.SkyCoord
- Optics
    - ra
    - dec
    - pa
    - focal_length  = 0.4
    - diameter      = 7.3
    - distortion    = None
- Detector
    - dimension     = 4096x4096
    - pixel_scale   = 10 (um)
    - field of view = 0.6 deg
    - displacement  = None
