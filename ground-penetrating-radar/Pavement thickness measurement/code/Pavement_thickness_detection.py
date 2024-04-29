"""
Created on Thursday April 18 2024

@author: Shengxin'Chauncy' Cai.ctr
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from scipy.signal import find_peaks
import sys 
import GPR_locate_rebars as gpr_lr
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
import pickle
import os
from scipy.interpolate import griddata

def next_letter_after_substring(input_string, substring):
    '''
    Find the next letter(s) after a given substring in a string.
    
    Parameters:
    - input_string: The string to search within.
    - substring: The substring to find within the input string.
    
    Returns:
    - str or None: The letter(s) immediately following the substring, or None if the substring doesn't exist or is at the end of the string.
    '''
    index = input_string.find(substring)  # Find the index of the substring in the input string
    if index != -1 and index < len(input_string) - len(substring):  # Check if substring exists and not at the end

        if input_string[index + len(substring)+1] == '-' or input_string[index + len(substring)+1] == '.' or input_string[index + len(substring)+1] == '_':
            lane_num_TBR = input_string[index + len(substring)]
        else:
            lane_num_TBR = input_string[index + len(substring)] + input_string[index + len(substring)+1]

        return lane_num_TBR  # Return the letter after the substring
    else:
        return None  # Return None if the substring doesn't exist or is at the end

def clean_A_scan(df_1, positive_pick=True):
    """
    Clean the beginning part of the A-scan. Since some of the initial segments of GPR A-scans are not physically accurate, we are replacing them with adjacent A-scans.

    Parameters:
    - df_1: A pandas DataFrame containing A-scan data.
    - positive_pick: Uses positive peaks for the processing. Default is True. It uses negative peaks if this variable is False.

    Returns:
    - df_1_cleaned: Cleaned A-scan DataFrame.
    """
    # Make a copy of df_1 to avoid modifying the original DataFrame
    df_1_copy = df_1.copy()
    
    # Convert df_1_copy to negative if positive_pick is False
    df_1_copy = -df_1_copy if not positive_pick else df_1_copy
    
    df_1_np = df_1_copy.values
    
    # Check condition for cleaning
    if (np.abs(np.average(df_1_np[0, :])) > 10 * np.abs(np.average(df_1_np[2, :])) or
        np.average(df_1_np[1, :]) > 10 * np.abs(np.average(df_1_np[2, :]))):
        
        # Set values of first two rows to the third row
        df_1_np[0, :] = df_1_np[2, :]
        df_1_np[1, :] = df_1_np[2, :]
    
    # Convert back to DataFrame
    df_1_cleaned = pd.DataFrame(df_1_copy)
    
    return df_1_cleaned

def Plot_b_scan_advanced(data, Travel_velocity, distance_, Time_TS, Time_BS, rhf_spm, rhf_range, Lane_num, ScanPass_num, midpoint_factor=0.4):
    '''
    Plot B-scan with pavement lines.

    Parameters:
    - data: Pandas DataFrame containing the GPR scan data
    - Travel_velocity: Wave velocity in a media
    - distance_: Calculated GPR scan distance
    - Time_TS: The time which represents top surface reflection
    - Time_BS: The time which represents bottom surface reflection
    - rhf_spm: GPR scans per meter
    - rhf_range: GPR wave travel time range
    - Lane_num: Lane number of the GPR scan
    - ScanPass_num: Scan pass number of the GPR scan
    - midpoint_factor: Factor determining the midpoint color in the heatmap (default is 0.4)

    '''
    depth_ = (Travel_velocity * ((np.linspace(0, data.shape[0] - 1, data.shape[0])) / data.shape[0] * rhf_range / (1e+9))) / 2 * 39.3701  # unit: inch

    depth_TS = (Travel_velocity * (Time_TS / data.shape[0] * rhf_range / (1e+9))) / 2 * 39.3701
    depth_BS = (Travel_velocity * (Time_BS / data.shape[0] * rhf_range / (1e+9))) / 2 * 39.3701

    data = data.values
    fig, ax = plt.subplots(figsize=(18, 7))

    vmin, vmax = data.min(), data.max()

    midpoint = vmin + (vmax - vmin) * midpoint_factor

    cmap = plt.cm.gray
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=midpoint, vmax=vmax)
    heatmap = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto', extent=(distance_.min(), distance_.max(), depth_.max(), depth_.min()))
    ax.plot(distance_, depth_TS, color='r', linestyle='-', linewidth=1.5, markersize=1, label='Line')
    ax.plot(distance_, depth_BS, color='r', linestyle='-', linewidth=1.5, markersize=1, label='Line')
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label('Intensity', fontsize=16)
    ax.set_ylabel('Depth (in.)', fontsize=16)
    ax.set_xlabel(f'Travel Distance in Lane {Lane_num} Pass {ScanPass_num} (ft.)', fontsize=16)
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis="x", bottom=False, top=True, labeltop=True, labelbottom=False, direction='in',
                   labelsize=15)
    ax.yaxis.set_tick_params(direction='in')
    plt.tight_layout()
    #plt.savefig(f'b_scan_advanced_GPR lane {Lane_num} Pass {ScanPass_num}.png', bbox_inches='tight')
    # plt.title(f'Lane {Lane_num} Pass {ScanPass_num}')
    plt.show()

def Collect_surface_thickness(data):
    '''
    Plot B-scan with pavement lines.

    Parameters:
    - data: Pandas DataFrame containing the GPR scan data

    Returns:
    - peaks_all: Coordinates of the GPR scan amplitude peaks
    '''
    peaks_all = np.zeros(shape=(data.shape[0], data.shape[1]))
    for i in range(0, data.shape[1]):
        temp = data[i]
        temp = minmax_scale(temp, feature_range=(-1, 1))  # normalize the data
        peaks, _ = find_peaks(temp, prominence=0.34, distance=10)  # positive peaks: 0.34 0.36 prominence= 0.4 negative peaks: prominence= 0.004 0.001 0.01  0.0000001
        peaks_all[:peaks.shape[0], i] = peaks

        # if i % 1000 == 0:
        #     plt.figure()
        #     prominences_ = peak_prominences(temp, peaks)[0]
        #     contour_heights = temp[peaks] - prominences_
        #     plt.plot(temp)
        #     plt.plot(peaks, temp[peaks], 'rx', label='Peaks')
        #     plt.vlines(x=peaks, ymin=contour_heights, ymax=temp[peaks], linestyles="dashed", colors="k")
        #     plt.ylabel('Value')
        #     plt.title('Plot of %i th scan' %i)
        #     plt.show()
        #     # plt.savefig(f'a_scan_{i}_GPR.png', bbox_inches='tight')
    return peaks_all

def IQR_outlier_removal(Thickness_, lower_quantile=0.25, upper_quantile=0.75, ratio=1.5):
    """
    Remove outliers from a given array of thickness data.

    Parameters:
    - Thickness_: Array of thickness data.
    - lower_quantile: The lower quantile for the calculation of the first quartile. Default is 0.25.
    - upper_quantile: The upper quantile for the calculation of the third quartile. Default is 0.75.
    - ratio: The ratio used to determine the whiskers of the boxplot. Default is 1.5.

    Returns:
    - Thickness_cleaned: Cleaned array of thickness data.
    - indices_within_limits: Indices of data within the defined limits.
    - indices_outside_limits: Indices of outliers.
    """
    # Finding the 1st quartile
    q1_ = np.quantile(Thickness_, lower_quantile)
    # Finding the 3rd quartile
    q3_ = np.quantile(Thickness_, upper_quantile)
    # Finding the median
    med = np.median(Thickness_)
    # Finding the IQR region
    iqr_ = q3_ - q1_

    # Finding upper and lower bounds
    upper_bound_ = q3_ + (ratio * iqr_)
    lower_bound_ = q1_ - (ratio * iqr_)

    # Find outliers
    outliers_ = Thickness_[(Thickness_ <= lower_bound_) | (Thickness_ >= upper_bound_)]
    indices_outside_limits = np.where((Thickness_ < lower_bound_) | (Thickness_ > upper_bound_))[0]

    # Remove outliers
    Thickness_cleaned = Thickness_[(Thickness_ >= lower_bound_) & (Thickness_ <= upper_bound_)]
    indices_within_limits = np.where((Thickness_ >= lower_bound_) & (Thickness_ <= upper_bound_))[0]

    return Thickness_cleaned, indices_within_limits, indices_outside_limits

def load_pavement_depth_coordinates(folder_current, with_outliers=True):
    """
    Load pavement depth coordinates from pickle files in a specified folder.

    Parameters:
    - folder_current: Path to the folder containing the pickle files.
    - with_outliers: Flag indicating whether to select data with outliers. Default is True.

    Returns:
    - Distance_X: List of distance in X-coordinate data.
    - Thickness_Z: List of pavement thickness data.
    """
    Distance_X = []
    Thickness_Z = []

    for file in os.listdir(folder_current):
        if file.endswith('.pickle'):
            Lane_num = next_letter_after_substring(file, "Lane_")
            ScanPass_num = next_letter_after_substring(file, "Pass_")
            with open(os.path.join(folder_current, f'Lane_{Lane_num}_Pass_{ScanPass_num}.pickle'), 'rb') as f3:
                input_sum_dict = pickle.load(f3)
                # Select data for contour plot (With outliers or without outliers)
                if with_outliers:
                    Distance_X.append(input_sum_dict['Distance_cleaned'])
                    Thickness_Z.append(input_sum_dict['Thickness_cleaned'])
                else:
                    Distance_X.append(input_sum_dict['Distance_original'])
                    Thickness_Z.append(input_sum_dict['Thickness_original'])

    return Distance_X, Thickness_Z

def thickness_plot_scatter(Thickness_, rhf_spm, Lane_num, ScanPass_num):
    '''
    Plots measured pavement thickness as scatter with line.

    Parameters:
    - Thickness_: Calculated pavement thickness based on the travel velocity and time
    - rhf_spm: GPR scans per meter
    - Lane_num: Lane number of the GPR scan
    - ScanPass_num: Scan pass number of the GPR scan
    '''
    fig = plt.figure(figsize=(15, 5))
    distance_ = np.linspace(0, Thickness_.shape[0]-1, Thickness_.shape[0]) / rhf_spm * 3.28084  #unit: ft
    plt.scatter(distance_, Thickness_)
    plt.plot(distance_, Thickness_, color='r', linestyle='-', linewidth=1, markersize=5, label='Line')
    plt.xlabel('distance (ft.)')
    plt.ylabel('Thickness (inch)')
    plt.ylim(0, 9)
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    plt.title(f'Surface layer thickness vs. scan distance (Lane {Lane_num} Pass {ScanPass_num})')
    plt.show()

def interpolate_thickness(Distance_X, Distance_Y, Thickness_Z):
    """
    Interpolate thickness data on a finer mesh using linear interpolation.

    Parameters:
    - Distance_X: List of arrays containing X distances.
    - Distance_Y: List of arrays containing Y distances.
    - Thickness_Z: List of arrays containing thickness values.

    Returns:
    - X_fine: X grid space from Distance_X
    - Y_fine: Y grid space from Distance_Y
    - result_interp_fine: Interpolated thickness data on a regular mesh.
    """
    # Obtain flattened X and Y with all lines included
    flattened_Distance_X = np.concatenate([arr.flatten() for arr in Distance_X])
    repeat_counts = np.concatenate([arr.shape for arr in Distance_X])
    repeat_Distance_Y = [np.repeat(val, count) for val, count in zip(Distance_Y, repeat_counts)]
    flattened_Distance_Y = np.concatenate(repeat_Distance_Y)
    flattened_Thickness_Z = np.concatenate(Thickness_Z)
    
    # Create a regular mesh for plotting
    X_fine = np.linspace(flattened_Distance_X.min(), flattened_Distance_X.max(), np.round(np.mean(repeat_counts)).astype(int))
    Y_fine = np.linspace(flattened_Distance_Y.min(), flattened_Distance_Y.max(), 300)
    X_mesh_fine, Y_mesh_fine = np.meshgrid(X_fine, Y_fine)
    
    # Interpolate the data on the finer mesh using linear interpolation
    result_interp_fine = griddata((flattened_Distance_X, flattened_Distance_Y),
                                    flattened_Thickness_Z, (X_mesh_fine, Y_mesh_fine), method='linear')
    
    return X_fine, Y_fine, result_interp_fine

def contour_plot_No_Transition(Y_fine, X_fine, result_interp_fine, colors, bounds, Lane_num):
    '''
    Create a contour plot of pavement thickness without continuous color spectrum.
    
    Parameters:
    - Y_fine: Y-grid space
    - X_fine: X-grid space
    - result_interp_fine: Interpolated pavement thickness values.
    - colors: List of colors for contour levels.
    - bounds: List of color boundaries
    - Lane_num: Lane number for which the contour plot is generated.

    '''
    plt.figure(figsize=(10, 7))
    ax = plt.subplot(1, 1, 1)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, len(colors))
    im = plt.imshow(result_interp_fine, extent=(X_fine.min(), X_fine.max(), Y_fine.min(), Y_fine.max()),
               cmap=cmap, origin='lower', norm=norm, aspect='auto')
    ax.set_xlabel('Travel Distance along eastern edge (ft.)', fontsize=12)
    ax.set_ylabel('Transverse Distance along southern edge (ft.)', fontsize=12)
    ax.set_ylim(0, 14)
    ax.tick_params(axis="both", which="both", bottom=True, top=True, right=True, labeltop=True, direction='in', labelsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(20.0))  # Major ticks every 1 unit
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(MultipleLocator(2.0))  # Major ticks every 1 unit
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.title(f'Pavement Thickness Contour: Lane {Lane_num}', fontsize=14)
    plt.gca().set_aspect(5, adjustable='box')
    cbar = plt.colorbar(im, orientation='horizontal')
    cbar.set_label('Thickness (inch)')
    plt.tight_layout()
    #plt.savefig(f'Pavement Thickness Contour_NT_Lane {Lane_num}.png', bbox_inches='tight')
    plt.show()

def contour_plot_Gradual_Transition(Y_fine, X_fine, result_interp_fine, Lane_num):
    '''
    Create a contour plot of pavement thickness with continuous color spectrum.
    
    Parameters:
    - Y_fine: Y-grid space
    - X_fine: X-grid space
    - result_interp_fine: Interpolated pavement thickness values.
    - colors: List of colors for contour levels.
    - Lane_num: Lane number for which the contour plot is generated.

    '''
    plt.figure(figsize=(10, 7))
    ax = plt.subplot(1, 1, 1)
    colors = [(0, 'red'), (0.2, 'lime'), (0.6, 'yellow'), (1, 'blue')]  # Example colors and their positions
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
    im = plt.imshow(result_interp_fine, extent=(X_fine.min(), X_fine.max(), Y_fine.min(), Y_fine.max()),
               cmap=cmap, origin='lower')
    ax.set_xlabel('Travel Distance along eastern edge (ft.)', fontsize=12)
    ax.set_ylabel('Transverse Distance along southern edge (ft.)', fontsize=12)
    ax.set_ylim(0, 14)
    ax.tick_params(axis="both", which="both", bottom=True, top=True, right=True, labeltop=True, direction='in', labelsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(20.0))  # Major ticks every 1 unit
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(MultipleLocator(2.0))  # Major ticks every 1 unit
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.title(f'Pavement Thickness Contour: Lane {Lane_num}', fontsize=14)
    im.set_clim(vmin=3.5, vmax=6)
    plt.gca().set_aspect(5, adjustable='box')
    cbar = plt.colorbar(im, orientation='horizontal')
    cbar.set_label('Thickness (inch)')
    plt.tight_layout()
    #plt.savefig(f'Pavement Thickness Contour_GT_Lane {Lane_num}.png', bbox_inches='tight')
    plt.show()

def contour_plot_default(Y_fine, X_fine, result_interp_fine, Lane_num):
    '''
    Create a contour plot (default)
    
    Parameters:
    - Y_fine: Y-grid space
    - X_fine: X-grid space
    - result_interp_fine: Interpolated pavement thickness values.
    - Lane_num: Lane number for which the contour plot is generated.

    '''
    plt.figure(figsize=(10, 7))
    ax = plt.subplot(1, 1, 1)
    im = plt.imshow(result_interp_fine, extent=(X_fine.min(), X_fine.max(), Y_fine.min(), Y_fine.max()),
               cmap='gist_rainbow', origin='lower')
    ax.set_xlabel('Travel Distance along eastern edge (ft.)', fontsize=12)
    ax.set_ylabel('Transverse Distance along southern edge (ft.)', fontsize=12)
    ax.set_ylim(0, 14)
    ax.tick_params(axis="both", which="both", bottom=True, top=True, right=True, labeltop=True, direction='in', labelsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(20.0))  # Major ticks every 1 unit
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(MultipleLocator(2.0))  # Major ticks every 1 unit
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.title(f'Pavement Thickness Contour: Lane {Lane_num}', fontsize=14)
    im.set_clim(vmin=3.5, vmax=6)
    cbar = plt.colorbar(orientation='horizontal')
    cbar.set_label('Thickness (inch)')
    plt.gca().set_aspect(5, adjustable='box')
    plt.tight_layout()
    #plt.savefig(f'Pavement Thickness Contour_default_Lane {Lane_num}.png', bbox_inches='tight')
    plt.show()

def thickness_plot_2D(Distance_X, Thickness_Z, Lane_num, num_passes):
    '''
    Create a 2D plot of pavement thickness versus travel distance for multiple passes.
    
    Parameters:
    - Distance_X: List of arrays containing travel distances for each pass.
    - Thickness_Z: List of arrays containing pavement thicknesses for each pass.
    - Lane_num (int): Lane number for which the plot is generated.
    - num_passes (int): Number of passes.
    '''
    # Plot thickness vs. distance for all passes in one figure - v2
    fig, axs = plt.subplots(5, 1, figsize=(17, 9))
    custom_ylim = (2, 8)
    custom_xlim = (0, 150)
    plt.setp(axs, xlim=custom_xlim, ylim=custom_ylim)
    passes = range(1, num_passes+1)
    
    for i, ax in enumerate(axs):
        ax.scatter(Distance_X[i], Thickness_Z[i], s=10)
        ax.plot(Distance_X[i], Thickness_Z[i], color='r', linestyle='-', linewidth=1, markersize=5, label='Line')
        ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
        ax.tick_params(axis='y', direction='in', which='both')
        ax.tick_params(axis='x', direction='in', which='both')
        ax.set_title(f'Pass {passes[i]}')
        ax.set_ylabel('Thickness (in.)')
        if i == 4:
            ax.set_xlabel(f'Travel Distance in Lane No. {Lane_num} (ft.)')
    
    plt.tight_layout()
    #plt.savefig(f'Pavement Thickness vs travel distance (Lane {Lane_num}).png', bbox_inches='tight')
    plt.show()
