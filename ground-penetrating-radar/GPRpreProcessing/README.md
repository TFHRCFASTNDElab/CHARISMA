# GPRpreProcessing

Conventional GPR data analysis consists of creating attenuation maps. The initial steps in this process are time zero correction and migration. Time Zero correction is a very important aspect and an essential factor in order to the position the subsurface tragets, especially those located at shallow depths, at their true position in the depth. Migration is the transformation of the unfocused space-time GPR image to a focused om showing the object's true location and size with corresponding EM reflctivity. This algorithm is written to read and process gpr .DZT files.


## Requirements

We strongly recommend installing  [`charisma-env`](https://github.com/TFHRCFASTNDElab/CHARISMA/blob/main/environment/) via conda.

or install the following dependencies individually. 

- [`pandas`](https://pandas.pydata.org/)
- [`numpy`](http://www.numpy.org/)
- [`scipy`](https://www.scipy.org/)
- [`sklearn`](https://scikit-learn.org/stable/)
- [`plotly`](https://plotly.com/python/getting-started/)



## Instructions to run the scripts

From a command line:

```bash
python GPRpreProcessing.py -i input -f flag
```
```
required flags:
     OPTION       |      ARGUMENT       |       FUNCTIONALITY
-i, --input       | file: /file.DZT     |  input DZT file
-f, --flag        | positive integer    |  1 for TimeZero Correction and 2 for TimeZero correction and Migration(Stolt's)

```
Example
```bash
python GPRpreProcessing.py -i FILE____488.DZT -f 2
```
Output

![Example Radargram](https://github.com/TFHRCFASTNDElab/CHARISMA/blob/main/ground-penetrating-radar/GPRpreProcessing/output.png)
