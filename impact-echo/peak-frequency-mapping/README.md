# IE-Peak-Frequency-Mapping

Conventional IE data analysis in the form of peak frequency mapping serves as a reference, against which we compare the performance of other classification algorithms. To create peak frequency maps, the frequency corresponding to the maximum peak amplitude in each IE frequency spectrum is identified and mapped on to the location of IE test point on the specimen. This algorithm is written to read and process impact echo (.txt, .dat) files.


## Requirements

We strongly recommend installing  [`charisma-env`](https://github.com/TFHRCFASTNDElab/CHARISMA/blob/main/environment) via conda.

or install the following dependencies individually. 

- [`pandas`](https://pandas.pydata.org/)
- [`numpy`](http://www.numpy.org/)
- [`plotly`](https://plotly.com/python/getting-started/)
- [`scipy`](https://www.scipy.org/)
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
-f, --frequency   | positive integer    |  sampling frequency in kHz
-a, --annotation  | 1 | 0               |  1 for unannotated map and 2 for annotated map

```
Example
```bash
python peakFrequencyMapping.py -i /data -f 200 -a 2
```
Output

![Example Radargram](https://github.com/TFHRCFASTNDElab/CHARISMA/blob/main/impact-echo/peak-frequency-mapping/output.png)
