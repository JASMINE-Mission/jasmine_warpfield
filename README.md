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
ax.set_aspect(1.0)
ax.scatter(position[0].x, position[0].y, marker='x')
ax.set_xlabel('focal plane position (um)', fontsize=14)
ax.set_ylabel('focal plane position (um)', fontsize=14)
fig.tight_layout()
plt.show()
```


### Dependencies

- `numpy>=1.20`
- `scipy>=1.6`
- `pandas>=1.1`
- `astropy>=4.2`
- `astroquery>=0.4`
- `matplotlib>=3.3`
