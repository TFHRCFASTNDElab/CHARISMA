Introduction to Rebar Cover Depth
=======================================

In concrete bridge construction, the rebar cover depth (referred to as "concrete cover" by AASHTO) is the specified distance between the surface of the reinforcing bars and the surface of the concrete. (AASHTO, 2024b) This depth is crucial for maintaining structural integrity because it serves as a protective barrier against rebar corrosion. (AASHTO, 2024a) Additionally, it helps distribute and transfer loads, reducing the likelihood of concrete cracking or spalling. Hence, ASTM and AASHTO have established standard criteria for the minimum rebar cover depth required for various types of concrete bridges. (AASHTO, 2024b, 2024a)

.. image:: ../../_static/GPR_RCD/def.png
   :width: 1000
   :alt: figure 1
   :align: center

Figure 1. Bridge deck sample located at Turner Fairbank Highway Research Center (TFHRC). The rebar cover depth is defined as the distance between the concrete surface and rebar surface.

Bridge structures are exposed to harsh environmental and physical conditions. The moisture with chemicals (salt or acid) can cause corrosion on rebars and damage the concrete cover. Also, the vibration, abrasion, or impact from the vehicles on the bridge affects the concrete. (Aljalawi et al., 2016) Thus, it is crucial to monitor the rebar cover depth on the concrete bridge over time in terms of structural health. The most simple and intuitive way is ‘coring’, meaning drilling cylindrical holes in the structure and measuring the cover depth from the core sample. This method directly damages the structures and leaves weak points on the spot, resulting in poor structural condition.

The alternative way is using the Non-Destructive Evaluation (NDE) technique, and the most efficient method for the rebar cover depth measurement is using Ground Penetrating Radar (GPR). It emits electromagnetic waves underground and records the reflected waves to investigate the internal structure. For detailed physical principles, readers are redirected to :code:`Ground Penetrating Radar - Physical principle` section.

Objective of the Case Study
=======================================
This case study aims to provide a Python-based solution to process raw GPR data to measure the rebar cover depth on concrete bridges and output the 2D contour plots. It will give a high-level summarized report, followed by a detailed explanation about how we processed the data. Here we leverage actual GPR data from the concrete bridge located at Mississippi I-10. The NDE data is downloadable at FHWA LTBP InfoBridge™. 

.. image:: ../../_static/GPR_RM/LTBP_Bridge_thumbnail.png
   :width: 300
   :target: https://infobridge.fhwa.dot.gov/Data
   :alt: infobridge_logo
   
Prerequisites
=======================================

Readers are redirected to the :code:`USING CHARISMA` section to install the CHARISMA environment. To make our data processing transparent, we convert the data format from DZT to CSV. This part is explained in the :code:`Ground Penetrating Radar - Data Format Conversion` section.  

GPR Data from FHWA InfoBridge™

FHWA InfoBridge™ provides field data collections using various NDE technologies. We selected a bridge from Mississippi to illustrate the use of CHARIMSA for rebar identification (Structure number: 11000100240460B). The bridge name is I-10 over CEDAR LAKE ROAD, built-in 1970. The bridge type is a Prestressed Concrete Girder/Beam. The length of the bridge is 242.50 and the width is 59.40 ft, respectively.

.. image:: ../../_static/GPR_RM/infobridge.png
   :width: 1000
   :alt: figure 2
   :align: center
   
Figure 2. Selecting Mississippi I-10 bridge to download the NDE data from LTBP Infobridge™

In :code:`Downloaded files - GPR` tab in InfoBridge™, there are multiple ZIP files along with one XML file. The ZIP files are the GPR scan data, and the XML file contains the actual location of GPR scanning lines with specific file names. CHARISMA reads the XML file and automatically creates a visual plot of the GPR scan area (see Figure 3). Also, it automatically creates a nested CSV directory within each :code:`Region` directory, organizing all the CSV files with numbering according to the DZT file order.

.. image:: ../../_static/GPR_RCD/figure3.png
   :width: 1000
   :alt: figure 3
   :align: center
   
Figure 3. Visualized XML file downloaded from InfoBridge™.

Incorporating with all the converted CSV files, CHARISMA identifies the cartesian coordinates of rebar locations, measures the cover depth, and visualizes the result with a 2D contour figure. The entire process is automated, requiring proper input from the user. CHARISMA handles multiple CSV files within each region, breaking down the entire B-scan data into smaller segments along the x-axis survey line. Subsequently, the code automatically applies various adjustments such as time-zero correction, background removal, gain, dewow, migration, and rebar mapping to each segmented B-scan. Finally, a grid space is established covering the entire bridge to accommodate the calculated rebar locations from the four regions. We leverage the GPR location coordinates from the XML file to locate each rebar point on the bridge and interpolate among the points to create a 2D contour of the grid.

Rebar Cover Depth Results
=======================================

After configuring the input variables, executing the code will produce split GPR B-scans, scatter plots, and interpolated 2D contour plots for each lane. Figure 4(a) displays the detected rebars on the split GPR B-scans. The code automatically saves the coordinates of these rebar points as a list. Following this, the code concatenates the split B-scans and generates 2D scatter plots for the entire data, and splits the concatenated B-scans again to plot the 2D scatter plots along the actual GPR scan lines (Figure 4(b)). Lastly, the saved rebar point lists and the grid space defined along the GPR scan coordinates are utilized to create 2D contour plots (Figure 4(c)).

.. image:: ../../_static/GPR_RCD/figure4.png
   :width: 1000
   :alt: figure 4
   :align: center
   
Figure 4. CHARISMA outputs from Mississippi I-10 Region 01 rebar cover depth for the first 4 GPR scan lines. (a) The rebar mapping result (3 examples among 20 split B-scans), (b) the rebar coordinates, and (c) the interpolated 2D contour map. 

Figure 5 shows the final output of processing GPR data with CHARISMA, which is the interpolated 2D contour from the entire lanes in Zone 01. We repeat this process for the other regions to gather all the rebar points on the entire bridge. 

.. image:: ../../_static/GPR_RCD/figure5.png
   :width: 1000
   :alt: figure 5
   :align: center
   
Figure 5. Rebar cover depth result from Mississippi I-10 Region 01.

Once we have obtained the rebar points from all regions, CHARISMA generates the rebar cover depth contour map for the entire bridge. It combines all the rebar point lists from individual regions and establishes a grid space based on the XML file. There are two options available: one involves plotting without interpolating the gap among regions (Figure 6(a)), while the other interpolates the entire bridge based on the gathered rebar points (Figure 6(b)). Since we use the linear interpolation method, there is no available value for the edges, which is shown in grey color.

.. image:: ../../_static/GPR_RCD/figure6.png
   :width: 1000
   :alt: figure 5
   :align: center
   
Figure 6. CHARISMA output: 2D contour plot of rebar cover depth on Mississippi I-10 concrete bridge without interpolating the gap among regions (a), and with interpolating the entire bridge (b). 

Discussion
=======================================

We encountered some challenges in measuring the rebar cover depth: firstly, the data had some horizontal noises and rebar depth varies in a wide range. These characteristics constrained the amplification (gain) of the rebar signal, making data processing difficult. Secondly, the GPR scan distance calculated from the GPR configuration and the data frame from the XML file were mismatched, which caused significant spatial distortion on the 2D contour plot. 

**How do we use CHARISMA to solve the problem?**

CHARISMA successfully measured the rebar cover depth on the concrete bridge at Mississippi I-10. Our approach begins by leveraging the XML file containing the actual GPR scan coordinates to correlate with the GPR data. Then, we process GPR data based on the raw data analysis, pinpoint rebar coordinates with the K-means clustering algorithm, and concatenate all the rebar points from different regions with the offset correction. Finally, we interpolate the points to populate the grid space, ensuring it matches the dimensions of the bridge. All the details are organized in Code Explanation section.

**What limitations have been reminded of?**

The limitation of our work lies in two aspects: one is from the F-K migration, and the other is from the K-means clustering algorithm. The migration requires the dielectric to be constant for all media, which yields inconsistent migration among all the B-scans. We also observed severe parabola superpositions on some B-scan areas, showing some weird migrated points. These anomalies hinder the algorithm from finding the actual rebar locations.

Also, the K-means clustering method has limitations in locating all the rebars correctly. It struggles to detect certain weakened rebar signals and occasionally misinterprets secondary positive diffracted signals or noise with positive amplitude as rebars. While these errors are mitigated to some extent by employing interpolation for all rebar points, they nonetheless impact the results and call for improved constraints or methodologies.

Comprehensively, this approach still requires manual raw data analysis and setting appropriate parameters for code input, indicating that the code is not fully automated to get the result. Our focus currently lies on utilizing machine learning tools to locate rebars without relying on migration or the K-means clustering algorithm, aiming for complete automation of rebar cover depth measurement.

