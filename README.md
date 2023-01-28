[![test](https://github.com/JASMINE-Mission/jasmine_warpfield/actions/workflows/validation.yml/badge.svg?branch=develop)](https://github.com/JASMINE-Mission/jasmine_warpfield/actions/workflows/validation.yml)
[![build](https://github.com/JASMINE-Mission/jasmine_warpfield/actions/workflows/build.yml/badge.svg?branch=develop)](https://github.com/JASMINE-Mission/jasmine_warpfield/actions/workflows/build.yml)
[![Maintainability](https://api.codeclimate.com/v1/badges/92fcbdadd8b118238161/maintainability)](https://codeclimate.com/github/JASMINE-Mission/jasmine_warpfield/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/92fcbdadd8b118238161/test_coverage)](https://codeclimate.com/github/JASMINE-Mission/jasmine_warpfield/test_coverage)

# JASMINE warpfield demonstration code
An experimental code to simulate the image warp function.


## Installation

The package is available using the command below:

``` console
$ pip install git+https://github.com/JASMINE_Mission/jasmine_warpfield.git
```

Otherwise clone this repository and try the command below:

``` console
$ python setup.py install
```

The module `jasmine_warpfield` will be installed in your system. A simple example is described below.

``` python
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
import warpfield as w

pointing = SkyCoord(0.0*u.deg, 0.0*u.deg, frame="galactic")
position_angle = Angle(5.0*u.deg)

jasmine = w.Telescope(pointing, position_angle)
source_table = w.retrieve_gaia_sources(pointing, radius=0.4*u.deg)
position = jasmine.observe(source_table.skycoord)

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
