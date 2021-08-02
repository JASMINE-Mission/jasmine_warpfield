# Jasmine Astrometry Challenge
## Case 8
The data of the astrometry challenge Case 8: In this challenge, the telescope is subject to distortion. Not as in Case7, the distortion is defined in the SIP convention. Astronomical sources are fetched from the Gaia EDR3 catalog. The Galactic center region is selected as a test field. The sources are stored in the source list file. Observation fields are randomly selected from the test field. Lists of source positions on the focal plane and unique source ID numbers are provided. Each table contains the data of 4 fields obtained in a 2&times;2 tiling pattern. The stride angle is fixed to 4 arcmins, approximately half of the field-of-view. Solve the relative positions of the fields and the plate scale without the IRCS coordinates of the sources. The field parameters are available in separate tables. The center of the tiling pattern and the distortion parameters are presented in the meta section in the tables.


- [case8_source_list.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case8/case8_source_list.txt)
- [case8_challenge_00.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case8/case8_challenge_00.txt) / [case8_challenge_00_pointing.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case8/case8_challenge_00_pointing.txt)
- [case8_challenge_01.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case8/case8_challenge_01.txt) / [case8_challenge_01_pointing.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case8/case8_challenge_01_pointing.txt)
- [case8_challenge_02.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case8/case8_challenge_02.txt) / [case8_challenge_02_pointing.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case8/case8_challenge_02_pointing.txt)
- [case8_challenge_03.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case8/case8_challenge_03.txt) / [case8_challenge_03_pointing.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case8/case8_challenge_03_pointing.txt)
- [case8_challenge_04.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case8/case8_challenge_04.txt) / [case8_challenge_04_pointing.txt](https://github.com/xr0038/jasmine_warpfield/raw/master/challenge/case8/case8_challenge_04_pointing.txt)


### Table Explanation
#### Exercises

|Column|Unit|Explanation|
|------|-----|-----|
|x|&mu;m|x-positions on the focal plane with distortion|
|y|&mu;m|y-positions on the focal plane with distortion|
|catalog_id||the index of the source in the source list|
|x_orig|&mu;m|x-positions on the focal plane w/o distortion|
|y_orig|&mu;m|y-positions on the focal plane w/o distortion|
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
