# PAUT Visualization

The algorithm aims to reproduce the A-scans for all the swept angles recorded during PAUT data acquisition for a given length and construct B, C, and corrected S-scans from it. B-scans are the stacked A-scans arranged one after the other with the amplitude of the signals presented in color-coded images. C-scan provides a top-view image where the position of the discontinuities is plotted according to the focal law sequence generating different angles. The s-scan displays the range of the swept angles as defined in the PAUT test setup.


This algorithm works for the following format of PAUT data from Olympus Omniscan MX2 Equipment. Take a look at the [`sample file`] for additional information about the data format.(https://github.com/TFHRCFASTNDElab/CHARISMA/blob/main/phased-array-ultrasonic-testing/PAUTvisualization/TP1-NEG1.63.xlsx) for additional information about the data format.

## Requirements

We strongly recommend installing  [`charisma-env`](https://github.com/TFHRCFASTNDElab/CHARISMA/blob/main/environment) via conda.

or install the following dependencies individually. 

- [`pandas`](https://pandas.pydata.org/)
- [`numpy`](http://www.numpy.org/)
- [`plotly`](https://plotly.com/python/getting-started/)



## Instructions to run the scripts

From a command line:

```bash
python PAUTvisualization.py -i input
```
```
required flags:
     OPTION       |      ARGUMENT       |       FUNCTIONALITY
-i, --input       | file: /data         |  input file

```
Example
```bash
python PAUTvisualization.py -i TP1-NEG1.63.xlxs
```
Output

C-Scan

![Example Radargram](https://github.com/TFHRCFASTNDElab/CHARISMA/blob/main/phased-array-ultrasonic-testing/PAUTvisualization/cscan.png)

S-Scan

![Example Radargram](https://github.com/TFHRCFASTNDElab/CHARISMA/blob/main/phased-array-ultrasonic-testing/PAUTvisualization/sscan.png)
