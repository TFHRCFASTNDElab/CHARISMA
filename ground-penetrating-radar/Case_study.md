# Case Study: An application of GPR for rebar identification in reinforced concrete bridge decks

<img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/c25b5bc0-6c21-424c-b90f-135e052cdff8" width="1000" />

***★ Please click the red box on the upper right side of GitHub to open the outlines***

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

Figure 1. The definition of A-, B-, and C-scan.(Merkle, Frey, and Reiterer 2021)

<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/bde0cf0b-773f-4793-8996-d585ccd7e38f" alt="image">
</p>

Figure 2. GPR data acquisition on the bridge (Left) and how GPR B-scan data looks like with respect to the actual rebar configuration (Right). The horizontal axis (x-axis) of the data corresponds to the distance along the survey line (moving direction of the GPR machine), and the vertical axis (z-axis) is the depth of the ground. 

#### Objectives of the Case Study

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This case study aims to provide a Python-based solution to processing GPR data for rebar identification in concrete bridge decks. The case study will give a tutorial with detailed step-by-step on how to use CHARISMA for this task. The case study will first demonstrate the use of CHARISMA with GPR data collected in the Federal Highway Administration (FHWA) Non-Destructive Evaluation (NDE) Laboratory. The study will then extend the use of CHARISMA to process GPR data collected in the field, a bridge located in Mississippi. 

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

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This chapter focuses on elucidating the procedures employed for processing Ground Penetrating Radar (GPR) data obtained from our laboratory specimen.(Lin et al. 2018) The structure under examination within our laboratory setting replicates a section of a concrete bridge with embedded reinforcement bars (rebar). Our GPR data processing involves implementing advanced techniques for time-zero correction and F-K migration, ensuring a precise representation of the rebar configuration within the specimen.

#### Step 1. Read the saved CSV files
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;We read the saved CSV files to process further. Let’s use the read_csv function to define two Pandas DataFrames. After that, we set each row of configuration data `df_2` as a local variable in Python, using the `config_to_variable` function.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Let’s investigate the CSV data in detail. The DataFrame1 `df_1` is actual GPR data (512 Rows × 332 Cols). The 512 rows are the “depth” within the investigated underground subsurface (also considered as wavelet traveling time), and 332 columns are the number of scans, corresponding to a distinct scan instance where a radar wavelet is emitted and recorded by the antenna while the GPR machine traverses along the survey line.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The DataFrame2 `df_2` is a configuration setting of GPR machine (24 Rows × 2 Cols). Here we discuss some of the important parameters among them, but readers are redirected to the GPR manufacturer webpage for further details: GSSI SIR 3000 Manual https://www.geophysical.com/wp-content/uploads/2017/10/GSSI-SIR-3000-Manual.pdf. Here we discuss 7 important parameters in the configuration setting: `rh_nsamp`, `rhf_sps`, `rhf_spm`, `rhf_position`, `rhf_range`, `rhf_espr`, and `rhf_depth`.
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The parameter `rh_nsamp` represents the number of samples in the depth dimension (equivalent to the row dimension of GPR data 512). Imagine a grid interface of the underground subsurface to save the wavelet in discrete form (to save as data). The higher number of samples provide higher resolution, but larger size of the data. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The `rhf_sps` and `rhf_spm` refer to “scans-per-second” and “scans-per-meter”. Here scans mean the column dimension of the GPR data (332 in our data). Both of parameters indicate how many scans are obtained per unit time or distance. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The `rhf_position` and `rhf_range` are parameters for indicating position and range of the GPR scans in nano seconds (ns). For example, if `rhf_position = 0`, this means that the starting point for measuring positions in your GPR data is at the initial time when the radar pulse is sent. In other words, the measurement of positions starts from the moment the radar pulse is emitted. Similarly, if `rhf_range = 8`, it indicates the time it takes for the radar signals to travel to the subsurface and return takes 8 ns. By knowing the speed of electromagnetic waves and average dielectric constant, this time can be converted to a distance measurement.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The `rhf_espr` is average dielectric constant. It accounts for the varying properties of the subsurface materials. Different materials have different dielectric constants, and accurate knowledge of this parameter is crucial for correctly interpreting the travel time of radar signals and converting it into depth. Table 1 shows the relative dielectric constant for various materials. The `rhf_depth` is depth in meters, indicating how deep into the subsurface the radar signals penetrate underground.

#### Step 2. Time-zero correction

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Time-zero correction aligns multiple A-scans vertically. When the reflected signal is recorded in the receiver, several factors (thermal drift, electronic instability, cable length differences, or variations in antenna airgap) can cause inconsistent wavelet arrival time.(Jol 2008) Time-zero correction provides a more accurate depth calculation because it sets the top of the scan to a close approximation of the ground surface.(GSSI Inc. 2017) We assume the first positive peak of each A-scan represents the 0 m depth of the ground surface, which is called the “first positive peak” method.(Cook et al. 2022)(Yelf 2004)

<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/558763e5-ff93-41f4-a056-f9bcf913e8d3" alt="image">
</p>

Figure 4. Comparison of (a) before time-zero correction and (b) after scan-by-scan time-zero correction on our GPR data.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;There are a couple of methods of time-zero correction, one method is calculating the mean or median value of the first peak’s arrival time for the entire A-scans and cut out the data before the mean or median time.(Cook et al. 2022) However, this method is not robust since the time-zero position is not perfectly aligned with all the 1st positive peaks of the A-scans (see Figure 5). This method does not correspond with our assumption that the first positive peak of each A-scan represents the 0 m depth of the ground surface.

<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/03a6088f-0b02-47e2-819b-d9de0ecd277e" alt="image">
</p>

Figure 5. Plots of multiple A-scans with mean value method. The red vertical line shows the time-zero index based on the mean value. Some of the 1st positive peaks of A-scans align with the time-zero, but some A-scans do not.   

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The other method is “scan-by-scan” which is more reasonable and robust than the mean or median strategy. This method detects the 1st positive peaks of A-scans and processes them individually. However, since the location of the 1st peak is different from each other, the data length is also changed. For example, one A-scan has 1st positive peak at the 127th depth index, the other has the 130th index, and if we align the data based on the time-zero index, the starting and ending points of A-scans mismatch each other (See Figure 6). Thus, we cut out the data indices that are not in the common range. For example, if one of the A-scan ranges [-125, 386] and the other ranges [-135, 376], we are taking the data only in common so that the range of data indices becomes [-125, 376]. Figure 7 shows the result of our scan-by-scan time-zero correction after the index cut out.

<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/05d38f89-1285-40c2-a681-a90ea5357318" alt="image">
</p>

Figure 6. Plots of multiple A-scans with scan-by-scan method. The red vertical line shows the time-zero index, and the 1st positive peak is aligned to the red line. This method results in misalignment at the starting and ending points of the A-scan profiles.


<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/bb5e629b-58c2-49ad-8a06-d7ddea478369" alt="image">
</p>

Figure 7. Plots containing multiple A-scans using a scan-by-scan approach, with the exclusion of data outside the common range. The red vertical line shows the time-zero index, and the 1st positive peak is aligned to the red line. The misalignment issue at the starting and the ending points of the A-scan profiles is solved.

#### Step 3. Migration

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Migration process converts the hyperbolic signal into spatial locations of the subsurface object. The hyperbolas in the GPR B-scan data stem from the nature of its method. GPR uses electromagnetic wave pulse to detect buried objects, and the antenna is traversing over the survey line. When the antenna is directly above the object, the distance is at its minimum. As the antenna moves, the distance is getting longer. This changing distance results in a distance-time plot that resembles a hyperbola (Figure 8).

<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/de5d0530-65b3-4939-8a1e-60822e756e15" alt="image">
</p>

Figure 8. Schematic of GPR data acquisition process and hyperbola profile formation on the distance-time plot. Note that the distance is minimal when the object and antenna are vertically aligned.(Poluha et al. 2017) 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The hyperbola profiles in B-scan can lead to distortions in the radar image. Migration algorithms help correct these distortions, relocating the reflected signals to their correct positions in the subsurface, resulting in a more accurate representation of the buried features. Here we specifically introduce Frequency-Wavenumber (F-K or Stolt) migration.(Stolt 1978) This method has proven to be working well for the constant-velocity propagation media,(Xu, Miller, and Rappaport 2003)(Özdemir et al. 2014) which also fits well with our objective (bridge rebar configuration).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The F-K migration transforms the GPR B-scan into an artificially created wave map. In other words, this method specifically locates the object by reconstructing the waveforms at object locations. This is done by the Fourier Transform, converting the waves from the time-space domain to the frequency-wavenumber domain. To understand how this process works, we need the wave propagation equation in a media,

$$
\left( \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial z^2} - \frac{1}{v^2} \frac{\partial^2}{\partial t^2} \right) \phi(x, z, t) = 0,
$$

where $\phi\$ is the wave function (apparently GPR signals are electromagnetic waves), $x$ is the axis along the GPR survey line, $z$ is the axis along the depth, and $t$ is the time. If you are not familiar with the wave propagation equation in a media, you are redirected to the following YouTube video, which derives the wave equation from scratch with the guitar (acoustic wave): [https://www.youtube.com/watch?v=UXqUXYaRyGU&t=1684s&ab_channel=SteveBrunton](https://www.youtube.com/watch?v=UXqUXYaRyGU&t=1684s&ab_channel=SteveBrunton)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The next step is applying the Fourier transform to the wave function $\phi(x, z, t)\$,

$$
\phi(x, z, t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} E(k_x, \omega) e^{-j(k_x x + k_z z - \omega t)} \ dk_x \ d\omega,
$$

where $E(k_x, \omega)$ is the Fourier domain for every possible combination of wave number $k_x$ and frequency $\omega$. The physical meaning of the equation is that the wave function can be expressed as a summation of various plane waves with different wavenumbers and frequencies. It is noteworthy that the $E(k_x, \omega)$ is time-independent, which will be used to relocate the waveform at the specific location of the object underground (The meaning of “migration” comes from this aspect).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;After the Fourier transform, we correlate our actual GPR data into the equation. Note that we receive GPR signal at $z = 0$, where the antenna locations are at the surface. Then the wave function becomes,

$$
\phi(x, z=0, t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} E(k_x, \omega) e^{-j(k_x x - \omega t)} \ dk_x \ d\omega.
$$

This equation can transform the GPR signals into the frequency-wavenumber domain. To reconstruct the wave function, we need to know the $E(k_x, \omega)$. $E(k_x, \omega)$ is obtained by inverse Fourier transform and our GPR signal,

$$
E(k_x, \omega) = \frac{1}{2\pi} \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} \phi(x, z=0, t) e^{-j(k_x x - \omega t)} \ dx \ dt.
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Now, the equation is formulated to construct our parameter $E(k_x, \omega)$ based on the GPR data. We will employ the Exploding Source Modeling (ESM), which straightforwardly assumes that the reflected GPR wave originates directly from the object itself. This assumption allows us to express our reconstructed signal (the migrated image) as $\phi(x, z, t = 0)\$, where $t = 0$ signifies the initial waveform at the object. At this point, we successfully correlated the GPR signal $\phi(x, z = 0, t)\$ with the Fourier domain $E(k_x, \omega)$, and this $E(k_x, \omega)$ is again correlated to the wavefunction at the object $\phi(x, z, t = 0)\$. This is the main logic of the F-K migration.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;It's crucial to emphasize that this assumption holds true only when the data is time-zeroed, as our consideration of the time frame extends from the object to the surface. As previously mentioned, the time-zero correction assumes that we interpret the first positive peak as the surface reflection, and without time-zero, the time frame extends beyond the object to the surface when setting $t = 0$ at the object.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Another crucial aspect of ESM is the elimination of the $-\omega t$  term in the equation. This modification enables us to employ the Fast Fourier Transform for wavefunction reconstruction. By removing the time-dependent term from the exponential part, the equation becomes more numerically manageable. Our final mathematical representation of F-K migration is as follows,

$$
\phi(x, z, t=0) = \frac{1}{2\pi} \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} E(k_x, \omega) e^{-j(k_x x + k_z z)} \ dk_x \ d\omega.
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In our Python code, we discretize the wave function at the surface (GPR signal) $\phi(x, z = 0, t)$ as a matrix, and then solve the equations above numerically (through fast Fourier transform) to reconstruct the wave function at $t = 0$, $\phi(x, z, t = 0)$ based on $E(k_x, \omega)$. Figure 9 shows the migration results from the mean, and scan-by-scan time zero correction, respectively. Instead of the hyperbola profiles in the raw GPR B-scan data, there are some points with high amplitude (white dots in Figure 9). These points indicate the rebar locations underground. 

<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/a5e65925-b51b-46a7-b3bf-60ecbdfc9c24" alt="image">
</p>

Figure 9. F-K migration results from the (a) mean time-zero correction and (b) scan-by-scan time-zero correction. 

#### Step 4. Pinpoint Rebars
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;It is noteworthy that the size of the white points in Figure 9 are too large compared to the actual rebar diameter, so we estimate the rebar location with the K-means clustering method. K-means clustering is a popular unsupervised machine learning algorithm used for partitioning a dataset into distinct, non-overlapping clusters. The goal is to group similar data points together and separate dissimilar ones. Interesting features in K-means clustering are that 1) we can specify the number of clusters we want to observe, and 2) we can point out the centroid of each cluster. We take advantage of these features to pinpoint the rebar location.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Our code normalizes the signal amplitude of the migrated data frame from -1 to 1, gets all the data points where the amplitudes are above 0.7. This allows us to get all the white data points to form a cluster around the white locations. Since we know the number of white points, we apply the K-means clustering algorithm to identify the $(x, z)$ coordinates of each centroid of the cluster. In this case study, we want to compare the located results between the two time-zero correction methods. Figure 10 shows the estimated rebar location from the mean, and scan-by-scan time zero correction, respectively. Figure 11 shows the differences in these two cases, with the root mean squared error (RMSE) value of 0.101 inches. We confirmed that the scan-by-scan method is more accurate than the mean time-zero correction for our rebar locating algorithm since the latter method slightly overestimates the depth of the rebar.

<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/b820aff2-86d0-4815-9371-42acfb1ecd7f" alt="image">
</p>

Figure 10. Estimated rebar location from the (a) mean and (b) scan-by-scan time-zero correction.

<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/869d90f1-9c93-4ceb-8a58-2b5945ea5ce4" alt="image">
</p>

Figure 11. Rebar location difference between the mean time-zero and scan-by-scan time-zero correction.

#### Step 5. Discussion 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;With the precise rebar configuration identified in our lab specimen through time-zero correction and F-K migration techniques, we have established a solid foundation for our data analysis. The remarkably clean nature of the lab specimen data has allowed us to bypass the need for additional processing steps like gain or dewow adjustments.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;As we move forward, a crucial step in validating the robustness of our methodology is to apply it to GPR data acquired from an actual bridge. Real-world scenarios often present unique challenges that may not be fully replicated in a controlled laboratory environment. To ensure the reliability and applicability of our method, the next chapter shows how we process GPR data collected from the bridge structure.

## Chapter 4. GPR Data from FHWA InfoBridge™

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;FHWA InfoBridge™ provides field data collections using various NDE technologies, including GPR (https://infobridge.fhwa.dot.gov/Data). We selected a bridge from Mississippi to illustrate the use of CHARIMSA for rebar identification (Structure number: 11000100240460B). The bridge name is I-10 over CEDAR LAKE ROAD, built-in 1970. The bridge type is a Prestressed Concrete Girder/Beam. The length of the bridge is 242.50 and the width is 59.40 ft, respectively.

<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/fed8bd45-d4e1-41e9-ab38-65e1876b3bed" alt="image">
</p>

Figure 12. Screenshot of obtaining GPR data from FHWA InfoBridge™. Follow the URL, select the bridge, and scroll down to click LTBP. Then you can select the Download files to download the NDE data.

<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/9e7d0b1b-6db5-4f10-a470-4ff4dc72808b" alt="image">
</p>

Figure 13. Bridge location map of I-10 over CEDAR LAKE ROAD in Mississippi.

#### Step 1. Outlier control via interquartile range (IQR) method

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;With the same `read_csv` function from the previous chapter, we are making 2 Pandas DataFrames: GPR data and configuration. We noticed that there are some outlier values on the GPR data, so we removed them by the interquartile range (IQR) method. The IQR method is a statistical technique used to identify and remove outliers from a dataset. It involves calculating the range between the first quartile (Q1) and the third quartile (Q3) of the data distribution. Outliers are then identified and removed based on a specified multiplier of the IQR. This method is effective in addressing skewed or non-normally distributed data by focusing on the central portion of the dataset. A more detailed explanation is in our code description of the `Interquartile_Range`.


<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/841a406e-69d9-4f67-a6f4-0f3b8acf393a" alt="image">
</p>

Figure 14. Outlier control with IQR method. The `df_1` is the raw data and the `IQR_df_1` is the processed data. The red boxs show how the outlier value changed.

#### Step 2. Gain

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;We observed that the GPR signal isn't sufficiently clear for processing (see Figure 16 (a) and (c)), likely because the first peak amplitude significantly outweighs other signals. This disparity could be attributed to the GPR settings or signal attenuation. To address this issue, we employ a gain function to better highlight the reflected signal. We introduce two methods, namely power gain and exponential gain,(Huber and Hans 2018) to enhance the clarity of the reflected signal. The power gain function is defined as follows,

$$
x_g(t) = x(t) \cdot t^ \alpha.
$$

The signal $x(t)$ is multiplied by $t^ \alpha$, where $\alpha$ is the exponent. In the default case, $\alpha$ is set to 1. The effect of the power gain is to amplify the signal based on a power-law relationship with time $t$. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The exponential gain function is defined as,

$$
x_g(t) = x(t) \cdot \exp(\alpha \cdot t).
$$

Here, $\exp(\alpha \cdot t)$ represents the exponential function with $\alpha$ as the exponent. The signal $x(t)$ is multiplied by this exponential term. The exponential gain introduces a time-dependent exponential factor to the signal. Adjusting the $\alpha$ parameter allows control over the rate of amplification, impacting the signal characteristics. We applied the power gain function with $\alpha = 0.2$ only after the initial positive and negative peaks for improved visibility and processing.

<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/ebda4d59-3b52-4dbb-90b7-9349ce6cd697" alt="image">
</p>

Figure 15. A-scan of the GPR data (a) before gain and (b) after gain. The first peak in (a) is much larger than the other signals, making the B-scan image blur. Note that the base line of the signal in (b) is going upward, due to the amplification.

<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/d9f75034-493e-44a8-9766-813fcdaddb99" alt="image">
</p>

Figure 16. B-scan of the GPR data (a) & (c) before gain and (b) & (d) after gain.

#### Step 3. Dewow

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dewow is used to mitigate the effects of low-frequency variations, or "wows," in the GPR signal. The terminology is derived from the nature of the low-frequency variations or oscillations that it aims to mitigate—resembling a slow, undulating motion, akin to the exclamation "wow." Wows can result from various factors, such as uneven ground surfaces or fluctuations in the system. Dewow processing involves filtering or removing these low-frequency components from the GPR signal to enhance the clarity and resolution of subsurface features. This technique helps improve the overall quality of GPR data by reducing unwanted variations. Here we used trinomial dewow(Nesbitt et al. 2022) to correct the baseline of each A-scan.

<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/b4795435-f0d5-424a-b121-d7a201f7b7df" alt="image">
</p>

Figure 17. A-scan of the GPR data (a) before dewow and (b) after dewow. The baseline of the A-scan is corrected.

<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/4bb46b37-8888-4d32-9704-3c9fe2633e1c" alt="image">
</p>

Figure 18. B-scan of the GPR data (a) & (c) before dewow and (b) & (d) after dewow.

#### Step 4. Time-Zero Correction
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;With the processed data, we are applying the scan-by-scan time-zero correction. 

<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/7cfbbbdb-abe3-425d-b71c-495e20cc7fe5" alt="image">
</p>

Figure 19. The results of the scan-by-scan time-zero correction.

#### Step 5. Migration
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;We apply F-K migration based on the time-zeroed data.

<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/a4bb2ef6-57a0-4023-a41c-f76a2034fc5d" alt="image">
</p>

Figure 20. The results of the F-K migration.

#### Step 6. Pinpoint Rebars

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;We use the K-means clustering algorithm used in the previous chapter to pinpoint rebar locations.

<p align="center">
  <img src="https://github.com/TFHRCFASTNDElab/CHARISMA/assets/154364860/0e6e54d1-e78c-4555-887b-3b9c8d7b62f4" alt="image">
</p>

Figure 21. The estimated rebar location.

#### Step 7. Discussion

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compared to Lab specimens, we encountered a few challenges: firstly, the GPR data had some outliers with very high negative values (400 times larger than the 1st negative peak). Secondly, the amplitude of the 1st positive and negative peak outweighs the reflected signal, showing very unclear results. Lastly, the dielectric constant was not inputted correctly when the data was collected. This resulted in inaccurate migration results, so we estimated the dielectric based on the migration results.

**How do we use CHARISMA to solve the problem?**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;We successfully processed the actual concrete bridge GPR data by processing the outliers with IQR, applying gain to amplify the reflection signals, adjusting the A-scan baseline with dewow, leveraging the scan-by-scan time-zero correction, F-K migration, and K-means clustering algorithm to pinpoint the rebar locations. It is noteworthy that GPR data processing requires a deep understanding of each data processing method, and the workers should be able to adjust or apply variables or types of the function with respect to the data characteristics.

**What limitations have been reminded of?**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The limitation to our work lies in the F-K migration. It requires the dielectric to be constant for all media, which is hard to assume. Notably, the approximate dielectric value used in F-K migration is from the GPR configuration settings, which are defined by the user. This means if the actual data collector sets the value as default, the migration results can be significantly distorted or underestimated. We are currently working on how to automate to determine the estimated dielectric based only on the migration results.

## Chapter 5. References
Cook, Samantha N et al. 2022. “Automated Ground-Penetrating-Radar Post-Processing Software in R Programming.”

Huber, Emanuel, and Guillaume Hans. 2018. “RGPR—An Open-Source Package to Process and Visualize GPR Data.” In 2018 17th International Conference on Ground Penetrating Radar (GPR), IEEE, 1–4.

GSSI (Geophysical Survey Systems). Inc. 2017. “RADAN 7 Manual.”

Jol, Harry M. 2008. Ground Penetrating Radar Theory and Applications. elsevier.

Lin, Shibin et al. 2018. “Laboratory Assessment of Nine Methods for Nondestructive Evaluation of Concrete Bridge Decks with Overlays.” Construction and Building Materials 188: 966–82.

Merkle, Dominik, Carsten Frey, and Alexander Reiterer. 2021. “Fusion of Ground Penetrating Radar and Laser Scanning for Infrastructure Mapping.” Journal of Applied Geodesy 15(1): 31–45.

Nesbitt, Ian et al. 2022. “Readgssi: An Open-Source Tool to Read and Plot GSSI Ground-Penetrating Radar Data.” https://doi.org/10.5281/zenodo.5932420.

Özdemir, Caner, Şevket Demirci, Enes Yiǧit, and Betül Yilmaz. 2014. “A Review on Migration Methods in B-Scan Ground Penetrating Radar Imaging.” Mathematical Problems in Engineering 2014.

Poluha, Bruno et al. 2017. “Depth Estimates of Buried Utility Systems Using the GPR Method: Studies at the IAG/USP Geophysics Test Site.” International Journal of Geosciences 08(05): 726–42.

Stolt, R. H. 1978. “Migration By Fourier Transform.” Geophysics 43(1): 23–48.

Xu, Xiaoyin, Eric L. Miller, and Carey M. Rappaport. 2003. “Minimum Entropy Regularization in Frequency-Wavenumber Migration to Localize Subsurface Objects.” IEEE Transactions on Geoscience and Remote Sensing 41(8): 1804–12.

Yelf, Richard. 2004. “Where Is True Time Zero?” Proceedings of the Tenth International Conference Ground Penetrating Radar, GPR 2004 1(February 2004): 279–82.





