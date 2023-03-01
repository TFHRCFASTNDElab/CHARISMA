# HCP-Mapping

The half-cell potential (HCP) method can be used to identify corrosion activity of steel reinforcement in reinforced concrete structures. However, the method cannot directly measure the degree of corrosion. To create condition maps, the HCP measurement is mapped on to the corresponding location of ER test point on the specimen. This algorithm is written to read and process half cell potential (.xls, .json) files.


## Requirements

We strongly recommend installing  [`charisma-env`](https://github.com/TFHRCFASTNDElab/CHARISMA/blob/main/environment) via conda.

or install the following dependencies individually. 

- [`pandas`](https://pandas.pydata.org/)
- [`plotly`](https://plotly.com/python/getting-started/)


## Instructions to run the scripts

From a command line:

```bash
python HCPMapping.py -i input
```
```
required flags:
     OPTION       |      ARGUMENT       |       FUNCTIONALITY
-i, --input       | folder: /data       |  input folder of HCP files

```
Example
```bash
python HCPMapping.py -i /data
```
Output

![Example Radargram](https://github.com/TFHRCFASTNDElab/CHARISMA/blob/main/half-cell-potential/HCPMapping/output.png)
