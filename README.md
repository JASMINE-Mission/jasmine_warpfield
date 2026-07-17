[![unittest](https://github.com/astronasutarou/warpfield/actions/workflows/unittest.yml/badge.svg)](https://github.com/astronasutarou/warpfield/actions/workflows/unittest.yml)
[![validation](https://github.com/astronasutarou/warpfield/actions/workflows/validation.yml/badge.svg?branch=develop)](https://github.com/astronasutarou/warpfield/actions/workflows/validation.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/astronasutarou/warpfield)

# warpfield: Differentiable astrometric analysis

An experimental package for precise relative astrometric analysis and mock
measurement generation with JAX.

## Installation

The package is available using the command below:

```console
pip install git+https://github.com/astronasutarou/warpfield.git
```

Otherwise clone this repository and try the command below:

```console
pip install .
```

The module `warpfield` will be installed in your system. A simple simulation
example is described below.

```python
from warpfield import Exposure, Pointing, SourceCatalog
from warpfield.calibration import IdentityCalibration
from warpfield.instrument.jasmine import get_jasmine

telescope = get_jasmine(simulator=True)
source = SourceCatalog(
    ra=[266.4, 266.5],
    dec=[-29.0, -29.1],
)
pointing = Pointing.from_coord(
    frame='galactic',
    lon=0.0,
    lat=0.0,
    pa=5.0,
)
exposure = Exposure(pointing, IdentityCalibration())
measurement = telescope.observe(source, exposure)
```

## References

If you use this package in your research, please cite the following
publication:

- Ohsawa, R., Kawata, D., Kamizuka, T., Yamada, Y., Löffler, W., Biermann,
  M., 2025. Demonstration of plate analysis, an algorithm for precise
  relative astrometry. *Journal of Astronomical Telescopes, Instruments, and
  Systems* 11, 028001.
  [https://doi.org/10.1117/1.JATIS.11.2.028001](https://doi.org/10.1117/1.JATIS.11.2.028001)
