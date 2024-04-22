# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 09:30:58 2023

@author: steve.yang.ctr
"""
import numpy as np
from sklearn.preprocessing import minmax_scale
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def Plot_a_scan(data):
    '''
    Plot individual GPR scans with identified peaks.
    
    Parameters:
    - data: GPR Pandas dataframe
    '''
    for i in range(0, data.shape[1]):
        temp = data[i]
        temp = minmax_scale(temp, feature_range=(-1, 1))
        peaks, _ = find_peaks(temp, prominence=0.1, distance = 40)
        if i % 150 == 1:
            plt.plot(temp)
            plt.plot(peaks, temp[peaks], 'rx', label='Peaks')
            plt.ylabel('Value')
            plt.title('Plot of %i th scan' %i)
            plt.show()

def Plot_b_scan(data):
    '''
    Plot a GPR B-scan.
    
    Parameters:
    - data: GPR Pandas dataframe
    '''
    fig, ax = plt.subplots(figsize=(12, 4))
    heatmap = ax.imshow(data, cmap='gray', aspect='auto')
    
    ax.set_ylim(data.shape[0], 0)
    cbar = plt.colorbar(heatmap)
    plt.show()

def Plot_b_scan_advanced(data, midpoint_factor=0.4):
    '''
    Plot an advanced GPR B-scan with adjustable midpoint.
    
    Parameters:
    - data: GPR Pandas dataframe
    - midpoint_factor: Factor controlling the midpoint of the colormap.
    '''
    fig, ax = plt.subplots(figsize=(15, 6))

    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    # Calculate midpoint
    midpoint = vmin + (vmax - vmin) * midpoint_factor

    cmap = plt.cm.gray
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=midpoint, vmax=vmax)

    heatmap = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')

    cbar = plt.colorbar(heatmap, ax=ax)
    ax.set_ylim(data.shape[0], 0)
    # Set font size for legend
    cbar.ax.tick_params(labelsize=14)  # Adjust the font size as needed

    # Set font size for axis labels and ticks
    ax.tick_params(axis='both', which='major', labelsize=16)  # Adjust the font size as needed
    plt.show()

def Plot_migrated(data):
    '''
    Plot migrated GPR data.
    
    Parameters:
    - data: Migrated GPR numpy array
    
    Returns:
    None
    '''
    fig, ax = plt.subplots(figsize=(12, 4))
    
    heatmap = ax.imshow(data, cmap='Greys_r', aspect='auto')
    
    cbar = plt.colorbar(heatmap, ax=ax)

    ax.set_xlabel('GPR Survey line index')
    ax.set_ylabel('Depth index')
    plt.show()

def Plot_migrated_advanced(data, profilePos, velocity, rhf_range, rh_nsamp, midpoint_factor=0.4):
    '''
    Plot an advanced migrated GPR data with adjustable midpoint and depth calculation.
    
    Parameters:
    - data: 2D array representing migrated GPR data.
    - profilePos: x axis (survey line axis) after migration 
    - velocity: The wave speed in the media. (c)/math.sqrt(rhf_espr) * 1e-9 m/ns
    - rhf_range: The time it takes for the radar signals to travel to the subsurface and return (ns)
    - rh_nsamp: The number of rows in the GPR B-scan
    - midpoint_factor: Factor controlling the midpoint of the colormap.
    '''
    # mean time zero with 1st positive peak cut
    fig, ax = plt.subplots(figsize=(15, 5))  # Increase the height of the plot
    depth = (velocity/2) * rhf_range
    depth_per_point = depth / rh_nsamp
    depth_axis = np.linspace(0, depth_per_point * len(data) * 39.37, len(data))
    survey_line_axis = profilePos * 39.37

    vmin, vmax = data.min(), data.max()

    # Calculate the midpoint based on the provided factor
    midpoint = vmin + (vmax - vmin) * midpoint_factor

    cmap = plt.cm.gray
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=midpoint, vmax=vmax)

    heatmap = ax.imshow(data, cmap='Greys_r', extent=[survey_line_axis.min(), survey_line_axis.max(),
                                                      depth_axis.max(), depth_axis.min()], norm=norm)

    # Add colorbar for better interpretation
    cbar = plt.colorbar(heatmap, ax=ax)

    # Add the red zone between y=2.5 and y=3 inches with red color and alpha=0.5
    #ax.axhspan(2.5, 3.5, facecolor='red', alpha=0.5)

    # Set labels for axes
    ax.set_xlabel('GPR Survey line (inch)')
    ax.set_ylabel('Depth (inch)')

    # Adjust the aspect ratio to magnify the y-axis
    ax.set_aspect(5)
    ax.set_ylim(15, 0)
    # Show the plot
    return plt.show()

def Plot_migrated_rebarmaps(data, profilePos, velocity, rhf_range, rh_nsamp, midpoint_factor=0.4):
    '''
    Plot an advanced migrated GPR data with adjustable midpoint and depth calculation.
    
    Parameters:
    - data: 2D array representing migrated GPR data.
    - profilePos: x axis (survey line axis) after migration 
    - velocity: The wave speed in the media. (c)/math.sqrt(rhf_espr) * 1e-9 m/ns
    - rhf_range: The time it takes for the radar signals to travel to the subsurface and return (ns)
    - rh_nsamp: The number of rows in the GPR B-scan
    - midpoint_factor: Factor controlling the midpoint of the colormap.
    '''
    # mean time zero with 1st positive peak cut
    fig, ax = plt.subplots(figsize=(15, 12))  # Increase the height of the plot
    depth = (velocity/2) * rhf_range
    depth_per_point = depth / rh_nsamp
    depth_axis = np.linspace(0, depth_per_point * len(data) * 39.37, len(data))
    survey_line_axis = profilePos * 39.37

    vmin, vmax = data.min(), data.max()

    # Calculate the midpoint based on the provided factor
    midpoint = vmin + (vmax - vmin) * midpoint_factor

    cmap = plt.cm.gray
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=midpoint, vmax=vmax)

    heatmap = ax.imshow(data, cmap='Greys_r', extent=[survey_line_axis.min(), survey_line_axis.max(),
                                                      depth_axis.max(), depth_axis.min()], norm=norm)

    # Add colorbar for better interpretation
    cbar = plt.colorbar(heatmap, ax=ax, shrink=0.5)

    # Add the red zone between y=2.5 and y=3 inches with red color and alpha=0.5
    #ax.axhspan(2.5, 3.5, facecolor='red', alpha=0.5)

    # Set labels for axes
    ax.set_xlabel('GPR Survey line (inch)', fontsize=20)
    ax.set_ylabel('Depth (inch)', fontsize=20)
    cbar.ax.tick_params(labelsize=14)  # Adjust the font size as needed

    # Set font size for axis labels and ticks
    ax.tick_params(axis='both', which='major', labelsize=16)  # Adjust the font size as needed

    # Adjust the aspect ratio to magnify the y-axis
    ax.set_aspect(3)
    #ax.set_ylim(15, 0)
    # Show the plot
    return plt.show()