# Jasmine Astrometry Challenge
## Case 3
The data of the astrometry challenge Case 3: In this challenge, the telescope is affected by distortion. Artificial sources tiled in a grid pattern are generated. Lists of source positions on the focal plane and original ICRS coordinates are provided. Solve the position angle and distortion parameters. Solve the field and the distortion parameters. The answers are found as keywords in the meta section in the tables.


- [case3_challenge_00.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case3/case3_challenge_00.txt)
- [case3_challenge_01.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case3/case3_challenge_01.txt)
- [case3_challenge_02.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case3/case3_challenge_02.txt)
- [case3_challenge_03.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case3/case3_challenge_03.txt)
- [case3_challenge_04.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case3/case3_challenge_04.txt)


A different distortion function is used in the following data tables. The positions on the focal plane are distorted as

$$
\delta{x} = x + \sum_{m,n} c_{m,n}x^{m}y^{n}, \quad
\delta{y} = y + \sum_{m,n} d_{m,n}x^{m}y^{n},
$$

where $c_{0,0} = c_{1,0} = c_{0,1} = 0$ and $d_{0,0} = d_{1,0} = d_{0,1} = 0$.

- [case3_challenge_05.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case3/case3_challenge_05.txt)
- [case3_challenge_06.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case3/case3_challenge_06.txt)
- [case3_challenge_07.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case3/case3_challenge_07.txt)
- [case3_challenge_08.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case3/case3_challenge_08.txt)
- [case3_challenge_09.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case3/case3_challenge_09.txt)
