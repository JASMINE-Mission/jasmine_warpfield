[![unittest](https://github.com/JASMINE-Mission/jasmine_warpfield/actions/workflows/unittest.yml/badge.svg)](https://github.com/JASMINE-Mission/jasmine_warpfield/actions/workflows/unittest.yml)
[![validation](https://github.com/JASMINE-Mission/jasmine_warpfield/actions/workflows/validation.yml/badge.svg?branch=develop)](https://github.com/JASMINE-Mission/jasmine_warpfield/actions/workflows/validation.yml)
[![Maintainability](https://qlty.sh/badges/b197d600-fe17-4e94-a00a-caced39bdada/maintainability.svg)](https://qlty.sh/gh/JASMINE-Mission/projects/jasmine_warpfield)
[![Code Coverage](https://qlty.sh/badges/b197d600-fe17-4e94-a00a-caced39bdada/test_coverage.svg)](https://qlty.sh/gh/JASMINE-Mission/projects/jasmine_warpfield)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/JASMINE-Mission/jasmine_warpfield)

# warpfield: Plate Analysis demonstration code

An experimental code to demonstrate Plate Analysis, an algorithm for precise relative astrometric analysis. The code consists of two modules:

- `telescope`: A code to generate mock measurements.
- `analysis`: A code to estimate an astrometric solution.


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

jasmine = w.telescope.Telescope(pointing, position_angle)
source_table = w.telescope.retrieve_gaia_sources(pointing, radius=0.4*u.deg)
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
