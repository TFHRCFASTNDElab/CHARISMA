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

