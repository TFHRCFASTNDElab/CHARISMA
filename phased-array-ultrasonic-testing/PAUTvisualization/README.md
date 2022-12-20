# PAUT Visualization

The algorithm aims to reproduce the A-scans for all the swept angles recorded during PAUT data acquisition for given length and construct B, C, and corrected S-scans from it. B-scans are the stacked A-scans arranged one after the other with the amplitude of the signals presented in color coded images. C-scan in sectorial scans provides a top view image where the position of the discontinuities is plotted according to the focal law sequence generating different angles. The s-scan displays the range of the swept angles as defined in the PAUT test setup, using a fixed aperture.

## Requirements

We strongly recommend installing  [`charisma-env`](https://github.com/TFHRCFASTNDElab/CHARISMA/blob/main/environment) via conda.

or install the following dependencies individually. 

- [`pandas`](https://pandas.pydata.org/)
- [`numpy`](http://www.numpy.org/)
- [`plotly`](https://plotly.com/python/getting-started/)



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
-a, --annotation  | 1 | 0               |  1 for annotated map and 0 for unnannotated map

```
Example
```bash
python peakFrequencyMapping.py -i /data -f 200 -a 1
```
Output
