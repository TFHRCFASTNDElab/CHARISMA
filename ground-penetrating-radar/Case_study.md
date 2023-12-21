# Case Study: An application of GPR for rebar identification in reinforced concrete bridge decks

## Chapter 1. Introduction

#### Physical Principal

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ground Penetrating Radar (GPR) is a non-destructive evaluation technique to investigate objects or structures buried underground. The basic principle of GPR involves the transmission of electromagnetic (EM) waves with high frequencies ranging from 10 MHz to 2.6 GHz into the test subject. The EM will reflect when reaching subsurface objects that have dielectric properties different from the surrounding materials. Table 1 shows the dielectric of common materials. For concrete bridge decks, concrete has a dielectric normally around 8, while metallic rebars have a theoretically infinite dielectric. Thus, large reflections will be observed when using GPR to scan concrete bridge decks. The GPR system records the reflected signals and uses this information to identify rebars. 

Table 1. Relative Dielectric Permittivity for Different Materials.

| Material           | Dielectric Constant |
|--------------------|---------------------|
| Air                | 1                   |
| Clay               | 25-40               |
| Concrete           | 8-10                |
| Crushed base       | 6-8                 |
| Gravel             | 4-7                 |
| Hot mix asphalt    | 4-8                 |
| Ice                | 4                   |
| Insulation board   | 2-2.5               |
| Sand               | 4-6                 |
| Silt               | 16-30               |
| Silty sand         | 7-10                |
| Water (fresh)      | 81                  |

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;While the antenna is traveling the bridge horizontally (along the x-axis), the transmitter (Tx) of the antenna emits the electromagnetic wavelet and it penetrates through the test subject (z-axis). The reflected signal from the object in the subsurface is recorded into the receiver (Rx) of the antenna. Thus, the recorded data contains multiple reflected wavelets with respect to the distance along x. Each reflected wavelet as a function of discrete time is called an “A-scan”. “B-scan” is obtained by stitching multiple A-scans together along the survey line (x) and assigning a color (black to white) to the wave amplitude. Since the B-scan provides 2D vertical information underground, it is commonly used to analyze the system or structure. “C-scan” refers to the horizontal view of the specific horizontal layer of the subsurface. Figure 1 helps us understand the definition of each type of scan from GPR.

<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/70f1acb4-2a15-4eae-aef6-498ec6c014fe" alt="image">
</p>

Figure 1. The definition of A-, B-, and C-scan.[1]

<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/bde0cf0b-773f-4793-8996-d585ccd7e38f" alt="image">
</p>

Figure 2. GPR data acquisition on the bridge (Left) and how GPR B-scan data looks like with respect to the actual rebar configuration (Right). The horizontal axis (x-axis) of the data corresponds to the distance along the survey line (moving direction of the GPR machine), and the vertical axis (z-axis) is the depth of the ground. 

#### Objectives of the Case Study

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This case study aims to provide a Python-based solution to processing GPR data for rebar identification in concrete bridge decks. The case study will give a tutorial with detailed step-by-step on how to use CHARISMA for this task. The case study will first demonstrate the use of CHARISMA with GPR data collected in the FHWA NDE Laboratory. The study will then extend the use of CHARISMA to process GPR data collected in the field, a bridge located in Mississippi. 

## Chapter 2. Setup CHARISMA for GPR Data Analysis

#### Installing CHARISMA in Python

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;We recommend the use of Anaconda to simplify the environment setup. Anaconda is commonly used for managing Python packages in user-defined environments. In other words, it allows you to create isolated environments for different projects, each with its own set of dependencies, without interfering with the system-wide Python installation. This is particularly useful in data science and scientific computing where projects may have different requirements and dependencies. Here in First, download Anaconda in the provided URL.
URL to download conda: https://www.anaconda.com/download

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;We recommend installing our conda environment `charisma-env` to run the code properly. This environment has all the dependencies from our entire code. Download the `charisma-env.yml` file from our CHARISMA Github first, open your Anaconda Prompt, and go to your download directory by typing the following command.

`cd C:/your_download_path/`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To create and activate `charisma-env` with conda, run the following command:

`conda env create -f charisma-env.yml`

`conda activate charisma-env`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;After creating and activating the `charisma-env` environment, specify the environment and install (or launch) Jupyter Notebook from Anaconda Navigator to use our CHARISMA Python package. 

<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/65c52a82-8f65-4a72-aebb-020334b98795" alt="image">
</p>

Figure 3. Screenshot of launching the Jupyter Notebook under a specific environment.

#### Open GPR data in DZT
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;We provide Python code to allow users to read GPR data in DZT format. To simplify the data analysis process, we convert the GPR data from DZT format into CSV format, which is more friendly to readers. The `readdzt` function in our Python code `GPR_locate_rebars.py` is in charge of opening, reading, and configuring the DZT data into the Pandas DataFrame, and the `save_to_csv` function exports the DataFrame into CSV format. Here the readers need to define the directory path individually to save the CSV files in your storage system. Here we are saving two CSV files for DataFrame1 `df1` and DataFrame2 `df2` from one DZT file. The `df1` is for collected GPR data and `df2` is for the configuration settings of GPR.

## Chapter 3. GPR Data from FHWA NDE Lab Specimen

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This chapter focuses on elucidating the procedures employed for processing Ground Penetrating Radar (GPR) data obtained from our laboratory specimen.[3] The structure under examination within our laboratory setting replicates a section of a concrete bridge with embedded reinforcement bars (rebar). Our GPR data processing involves implementing advanced techniques for time-zero correction and F-K migration, ensuring a precise representation of the rebar configuration within the specimen.

#### Step 1. Read the saved CSV files
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;We read the saved CSV files to process further. Let’s use the read_csv function to define two Pandas DataFrames. After that, we set each row of configuration data `df_2` as a local variable in Python, using the `config_to_variable` function.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Let’s investigate the CSV data in detail. The DataFrame1 `df_1` is actual GPR data (512 Rows × 332 Cols). The 512 rows are the “depth” within the investigated underground subsurface (also considered as wavelet traveling time), and 332 columns are the number of scans, corresponding to a distinct scan instance where a radar wavelet is emitted and recorded by the antenna while the GPR machine traverses along the survey line.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The DataFrame2 `df_2` is a configuration setting of GPR machine (24 Rows × 2 Cols). Here we discuss some of the important parameters among them, but readers are redirected to the GPR manufacturer webpage for further details: GSSI SIR 3000 Manual https://www.geophysical.com/wp-content/uploads/2017/10/GSSI-SIR-3000-Manual.pdf. Here we discuss 7 important parameters in the configuration setting: `rh_nsamp`, `rhf_sps`, `rhf_spm`, `rhf_position`, `rhf_range`, `rhf_espr`, and `rhf_depth`.
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The parameter `rh_nsamp` represents the number of samples in the depth dimension (equivalent to the row dimension of GPR data 512). Imagine a grid interface of the underground subsurface to save the wavelet in discrete form (to save as data). The higher number of samples provide higher resolution, but larger size of the data. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The `rhf_sps` and `rhf_spm` refer to “scans-per-second” and “scans-per-meter”. Here scans mean the column dimension of the GPR data (332 in our data). Both of parameters indicate how many scans are obtained per unit time or distance. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The `rhf_position` and `rhf_range` are parameters for indicating position and range of the GPR scans in nano seconds (ns). For example, if `rhf_position = 0`, this means that the starting point for measuring positions in your GPR data is at the initial time when the radar pulse is sent. In other words, the measurement of positions starts from the moment the radar pulse is emitted. Similarly, if `rhf_range = 8`, it indicates the time it takes for the radar signals to travel to the subsurface and return takes 8 ns. By knowing the speed of electromagnetic waves and average dielectric constant, this time can be converted to a distance measurement.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The `rhf_espr` is average dielectric constant. It accounts for the varying properties of the subsurface materials. Different materials have different dielectric constants, and accurate knowledge of this parameter is crucial for correctly interpreting the travel time of radar signals and converting it into depth. Table 1 shows the relative dielectric constant for various materials. The `rhf_depth` is depth in meters, indicating how deep into the subsurface the radar signals penetrate underground.

#### Step 2. Time-zero correction

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Time-zero correction aligns multiple A-scans vertically. When the reflected signal is recorded in the receiver, several factors (thermal drift, electronic instability, cable length differences, or variations in antenna airgap) can cause inconsistent wavelet arrival time.[3] Time-zero correction provides a more accurate depth calculation because it sets the top of the scan to a close approximation of the ground surface.[4] We assume the first positive peak of each A-scan represents the 0 m depth of the ground surface, which is called as “first positive peak” method.[5][6]

