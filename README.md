# JASMINE warpfield demonstration code
An experimental code to simulate the image warp function.


## Installation

The package is available using the command below:

``` bash
pip install git+https://github.com/xr0038/jasmine_warpfield.git
```

Otherwise clone this repository and try the command below:

``` bash
python setup.py install
```

The module `jasmine_warpfield` will be installed in your system. A simple example is described below.

``` python
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
import warpfield as w

pointing = SkyCoord(0.0*u.deg, 0.0*u.deg, frame="galactic")
position_angle = Angle(5.0*u.deg)

jasmine = w.Telescope(pointing, position_angle)
gaia_sources = w.retrieve_gaia_sources(pointing, radius=0.4*u.deg)
position = jasmine.observe(gaia_sources)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(position[0].x, position[0].y, marker='x')
ax.set_xlabel('focal plane position (um)', fontsize=14)
ax.set_ylabel('focal plane position (um)', fontsize=14)
plt.show()
```


### Dependencies

- numpy
- scipy
- astropy
- astroquery
- matplotlib


## Basic Design

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
