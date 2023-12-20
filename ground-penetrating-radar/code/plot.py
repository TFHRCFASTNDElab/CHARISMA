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
    for i in range(0, data.shape[1]):
        temp = data[i]
        temp = minmax_scale(temp, feature_range=(-1, 1))
        peaks, _ = find_peaks(temp, prominence=0.1, distance = 40)
        if i % 150 == 0:
            plt.plot(temp)
            plt.plot(peaks, temp[peaks], 'rx', label='Peaks')
            plt.ylabel('Value')
            plt.title('Plot of %i th scan' %i)
            plt.show()

def Plot_b_scan(data):
    fig, ax = plt.subplots(figsize=(12, 4))
    heatmap = ax.imshow(data, cmap='gray', aspect='auto')
    
    ax.set_ylim(data.shape[0], 0)
    cbar = plt.colorbar(heatmap)
    plt.show()

def Plot_b_scan_advanced(data, midpoint_factor=0.4):
    data = data.values
    fig, ax = plt.subplots(figsize=(12, 4))
    
    vmin, vmax = data.min(), data.max()
    
    midpoint = vmin + (vmax - vmin) * midpoint_factor
    
    cmap = plt.cm.gray
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=midpoint, vmax=vmax)
    heatmap = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')
    
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label('Intensity')
    
    ax.set_ylim(data.shape[0], 0)
    plt.show()
    
def Plot_migrated(data):
    fig, ax = plt.subplots(figsize=(12, 4))
    
    heatmap = ax.imshow(data, cmap='Greys_r', aspect='auto')
    
    cbar = plt.colorbar(heatmap, ax=ax)

    ax.set_xlabel('GPR Survey line index')
    ax.set_ylabel('Depth index')
    plt.show()

def Plot_migrated_advanced(data, profilePos, rhf_depth, rh_nsamp, midpoint_factor=0.4):
    fig, ax = plt.subplots(figsize=(15, 2))
    
    depth_per_point = rhf_depth/rh_nsamp
    depth_axis = np.linspace(0, depth_per_point * len(data) * 39.37, len(data))
    survey_line_axis = profilePos*39.37
    
    vmin, vmax = data.min(), data.max()
    
    midpoint = vmin + (vmax - vmin) * midpoint_factor
    
    cmap = plt.cm.gray
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=midpoint, vmax=vmax)
    
    heatmap = ax.imshow(data, cmap='Greys_r', extent=[survey_line_axis.min(), survey_line_axis.max(), depth_axis.max(), depth_axis.min()], norm=norm)
    
    cbar = plt.colorbar(heatmap, ax=ax)
    
    ax.set_xlabel('GPR Survey line (inch)')
    ax.set_ylabel('Depth (inch)')
    plt.show()


