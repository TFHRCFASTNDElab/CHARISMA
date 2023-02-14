# ER-Mapping

Conventional ER data analysis in the form of condition mapping serves as a reference, against which we compare the performance of other classification algorithms. To create condition maps, the resistivity value is mapped on to the corresponding location of ER test point on the specimen. This algorithm is written to read and process electrical resistivity (.txt, .csv, .json) files.


## Requirements

We strongly recommend installing  [`charisma-env`](https://github.com/TFHRCFASTNDElab/CHARISMA/blob/main/environment) via conda.

or install the following dependencies individually. 

- [`pandas`](https://pandas.pydata.org/)
- [`plotly`](https://plotly.com/python/getting-started/)


## Instructions to run the scripts

From a command line:

```bash
python ERMapping.py -i input
```
```
required flags:
     OPTION       |      ARGUMENT       |       FUNCTIONALITY
-i, --input       | folder: /data       |  input folder of ER files

```
Example
```bash
python ERMapping.py -i /data
```
Output

![Example Radargram](https://github.com/TFHRCFASTNDElab/CHARISMA/blob/main/electrical-resistivity/ERMapping/output.png)
