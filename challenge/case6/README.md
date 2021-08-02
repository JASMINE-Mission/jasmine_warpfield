# Jasmine Astrometry Challenge
## Case 6
The data of the astrometry challenge Case 6: In this challenge, the telescope is free from distortion. Astronomical sources are fetched from the Gaia EDR3 catalog. The Galactic center region is selected as a test field. The sources are stored in the source list file. Observation fields are randomly selected from the test field. Lists of source positions on the focal plane and unique source ID numbers are provided. Each table contains the data of 25 fields obtained in a 5&times;5 tiling pattern. The angles after the file names are the strides of the tiling pattern. Solve the relative positions of the fields and the plate scale without the IRCS coordinates of the sources. The field parameters are available in separate tables. The telescope pointing coordinates of the central frame are found as keywords in the meta section in the tables.


- [case6_source_list.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case6/case6_source_list.txt)
- [case6_challenge_00.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case6/case6_challenge_00.txt) / [case6_challenge_00_pointing.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case6/case6_challenge_00_pointing.txt) (0.05&deg;)
- [case6_challenge_01.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case6/case6_challenge_01.txt) / [case6_challenge_01_pointing.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case6/case6_challenge_01_pointing.txt) (0.10&deg;)
- [case6_challenge_02.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case6/case6_challenge_02.txt) / [case6_challenge_02_pointing.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case6/case6_challenge_02_pointing.txt) (0.15&deg;)
- [case6_challenge_03.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case6/case6_challenge_03.txt) / [case6_challenge_03_pointing.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case6/case6_challenge_03_pointing.txt) (0.20&deg;)
- [case6_challenge_04.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case6/case6_challenge_04.txt) / [case6_challenge_04_pointing.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case6/case6_challenge_04_pointing.txt) (0.25&deg;)


### Table Explanation
#### Exercises

|Column|Unit|Explanation|
|------|-----|-----|
|x|&mu;m|x-positions on the focal plane|
|y|&mu;m|y-positions on the focal plane|
|catalog_id||the index of the source in the source list|
|ra|degree|right ascension of the sources (ICRS)|
|dec|degree|declination of the sources (ICRS)|
|field||the index of the observation field|


#### Source List

|Column|Unit|Explanation|
|------|-----|-----|
|catalog_id||the index of the source in the source list|
|ra|degree|right ascension of the sources (ICRS)|
|dec|degree|declination of the sources (ICRS)|


#### Pointing Table

|Column|Unit|Explanation|
|------|-----|-----|
|field||the index of the observation field|
|ra|degree|right ascension of the pointing center (ICRS)|
|dec|degree|declination of the pointing center (ICRS)|
|pa|degree|position angle of the telescope|
