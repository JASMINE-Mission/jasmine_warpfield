# JASMINE warpfield demonstration code
An experimental code to simulate the image warp function.


## Installation
Clone this repository and try the command below:

``` bash
python setup.py install
```

The module `warpfield` will be installed in your system.


### Dependencies

- numpy
- scipy
- astropy
- astroquery


## Usage

- Input
    - list of astronomical sources
    - optics parameters
        - pointing
        - position angle
        - focal length
        - distortion function
    - detector parameters
        - number of detectors
        - offsets
        - dimension
        - pixel displacement
- Output
    - object positions on the focal plane.
