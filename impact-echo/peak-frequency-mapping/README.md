# IE-Peak-Frequency-Mapping

Conventional IE data analysis in the form of peak frequency mapping serves as a reference, against which we compare the performance of other classification algorithms. To create peak frequency maps, the frequency corresponding to the maximum peak amplitude in each IE frequency spectrum is identified and mapped on to the location of IE test point on the specimen. This algorithm is written to read and process impact echo (.txt, .dat) files.


## Requirements
- [`pandas`](https://pandas.pydata.org/)
- [`numpy`](http://www.numpy.org/)
- [`scipy`](https://www.scipy.org/)
- [`matplotlib`](https://matplotlib.org/)
- [`seaborn`](https://seaborn.pydata.org/)
- [`natsort`](https://pypi.org/project/natsort/)



## Instructions to run the scripts

From a command line:

```bash
python peakFrequencyMapping.py -i input -o output -f frequency -a annotation
```
```
required flags:
     OPTION       |      ARGUMENT       |       FUNCTIONALITY
-i, --input       | folder: /data       |  input folder of IE files
-o, --output      | string, eg. "map"   |  output file name for the map
-f, --frequency   | positive integer    |  sampling frequency in kHz
-a, --annotation  | 1 | 0               |  1 for annotated map and 0 for unnannotated map

```
Example
```bash
python peakFrequencyMapping.py -i /data -o map -f 200 -a 1
```
Output

![Example Radargram](https://github.com/TFHRCFASTNDElab/IE-Peak-Frequency-Mapping/blob/main/map.png)