# Jasmine Astrometry Challenge
## Case 5
The data of the astrometry challenge Case 5: In this challenge, the telescope is affected by distortion. Astronomical sources are fetched from the Gaia EDR3 catalog. The observation fields are randomly selected from the sky. The number of sources in each exposure is smaller than in Case-4. Lists of source positions on the focal plane and original ICRS coordinates are provided. Solve the field and the distortion parameters. The field parameters are available in separate tables. The distortion parameters are found as keywords in the meta section in the tables.


- [case5_challenge_00.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case5/case5_challenge_00.txt) / [case5_challenge_00_pointing.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case5/case5_challenge_00_pointing.txt)
- [case5_challenge_01.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case5/case5_challenge_01.txt) / [case5_challenge_01_pointing.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case5/case5_challenge_01_pointing.txt)
- [case5_challenge_02.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case5/case5_challenge_02.txt) / [case5_challenge_02_pointing.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case5/case5_challenge_02_pointing.txt)
- [case5_challenge_03.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case5/case5_challenge_03.txt) / [case5_challenge_03_pointing.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case5/case5_challenge_03_pointing.txt)
- [case5_challenge_04.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case5/case5_challenge_04.txt) / [case5_challenge_04_pointing.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case5/case5_challenge_04_pointing.txt)


### Table Explanation
#### Exercises

|Column|Unit|Explanation|
|------|-----|-----|
|x|&mu;m|x-positions on the focal plane|
|y|&mu;m|y-positions on the focal plane|
|ra|degree|right ascension of the sources (ICRS)|
|dec|degree|declination of the sources (ICRS)|
|field||the index of the observation field|


#### Pointing Table

|Column|Unit|Explanation|
|------|-----|-----|
|field||the index of the observation field|
|ra|degree|right ascension of the pointing center (ICRS)|
|dec|degree|declination of the pointing center (ICRS)|
|pa|degree|position angle of the telescope|
