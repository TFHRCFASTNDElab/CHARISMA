# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:51:55 2024

@author: steve.yang.ctr
"""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.interpolate import griddata

class ContourProcessor:
    """
    A class to process rebar points from individual zones and generate contour maps for the entire bridge.
    
    Methods:
    - contour_scan_area_only(lists): Plot contour maps for each zone separately without interpolating the gap between zones.
    - contour_interpolate_entire(lists): Plot a contour map for the entire bridge by interpolating the rebar points from all zones.
    """
    def contour_scan_area_only(self, lists):
        """
        Plot contour maps for each zone separately without interpolating the gap between zones.
        
        Parameters:
        - lists (dict): A dictionary containing lists of rebar points for all zones.

        """
        x_coords = {}
        y_coords = {}
        z_coords = {}

        # Access lists using dictionary
        for i in range(1, 5):
            x_points_list, y_points_list, z_points_list = lists[f'0{i}']
            
            # Store coordinates in dictionaries
            x_coords[f'x_coord_0{i}'] = [array for sublist in x_points_list for array in sublist]
            y_coords[f'y_coord_0{i}'] = [array for sublist in y_points_list for array in sublist]
            z_coords[f'z_coord_0{i}'] = [array for sublist in z_points_list for array in sublist]

        # Increase the resolution by creating a finer mesh for each dataset
        X_fine_01 = np.linspace(min(np.concatenate(x_coords['x_coord_01'])), max(np.concatenate(x_coords['x_coord_01'])), 300)
        Y_fine_01 = np.linspace(min(np.concatenate(y_coords['y_coord_01'])), max(np.concatenate(y_coords['y_coord_01'])), 300)
        X_mesh_fine_01, Y_mesh_fine_01 = np.meshgrid(X_fine_01, Y_fine_01)

        X_fine_02 = np.linspace(min(np.concatenate(x_coords['x_coord_02'])), max(np.concatenate(x_coords['x_coord_02'])), 300)
        Y_fine_02 = np.linspace(min(np.concatenate(y_coords['y_coord_02'])), max(np.concatenate(y_coords['y_coord_02'])), 300)
        X_mesh_fine_02, Y_mesh_fine_02 = np.meshgrid(X_fine_02, Y_fine_02)

        X_fine_03 = np.linspace(min(np.concatenate(x_coords['x_coord_03'])), max(np.concatenate(x_coords['x_coord_03'])), 300)
        Y_fine_03 = np.linspace(min(np.concatenate(y_coords['y_coord_03'])), max(np.concatenate(y_coords['y_coord_03'])), 300)
        X_mesh_fine_03, Y_mesh_fine_03 = np.meshgrid(X_fine_03, Y_fine_03)

        X_fine_04 = np.linspace(min(np.concatenate(x_coords['x_coord_04'])), max(np.concatenate(x_coords['x_coord_04'])), 300)
        Y_fine_04 = np.linspace(min(np.concatenate(y_coords['y_coord_04'])), max(np.concatenate(y_coords['y_coord_04'])), 300)
        X_mesh_fine_04, Y_mesh_fine_04 = np.meshgrid(X_fine_04, Y_fine_04)


        # Interpolate the data on the finer mesh using linear interpolation for each dataset
        result_interp_fine1 = griddata((np.concatenate(x_coords['x_coord_01']), np.concatenate(y_coords['y_coord_01'])),
                                       np.concatenate(z_coords['z_coord_01']), (X_mesh_fine_01, Y_mesh_fine_01), method='linear')

        result_interp_fine2 = griddata((np.concatenate(x_coords['x_coord_02']), np.concatenate(y_coords['y_coord_02'])),
                                       np.concatenate(z_coords['z_coord_02']), (X_mesh_fine_02, Y_mesh_fine_02), method='linear')

        result_interp_fine3 = griddata((np.concatenate(x_coords['x_coord_03']), np.concatenate(y_coords['y_coord_03'])),
                                       np.concatenate(z_coords['z_coord_03']), (X_mesh_fine_03, Y_mesh_fine_03), method='linear')

        result_interp_fine4 = griddata((np.concatenate(x_coords['x_coord_04']), np.concatenate(y_coords['y_coord_04'])),
                                       np.concatenate(z_coords['z_coord_04']), (X_mesh_fine_04, Y_mesh_fine_04), method='linear')

        # Determine the combined extent
        extent1 = (X_mesh_fine_01.min(), X_mesh_fine_01.max(), Y_mesh_fine_01.min(), Y_mesh_fine_01.max())
        extent2 = (X_mesh_fine_02.min(), X_mesh_fine_02.max(), Y_mesh_fine_02.min(), Y_mesh_fine_02.max())
        extent3 = (X_mesh_fine_03.min(), X_mesh_fine_03.max(), Y_mesh_fine_03.min(), Y_mesh_fine_03.max())
        extent4 = (X_mesh_fine_04.min(), X_mesh_fine_04.max(), Y_mesh_fine_04.min(), Y_mesh_fine_04.max())
        plt.figure(figsize=(20, 12))
        
        # Define custom colormap with specific color segments
        colors = [(0, 'red'),        # From -12 dB (data_min) to -8 dB (red to orange)
                  (1/3, 'orange'),   # From -8 dB to -6 dB (orange to yellow)
                  (1/2, 'yellow'),   # From -6 dB to 0 dB (yellow to lime)
                  (1, 'lime')]       # From 0 dB (data_max)
        
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        # Plot each zone individually using imshow
        plt.imshow(result_interp_fine1, extent=extent1, aspect='auto', cmap=cmap, origin='lower', vmin=-12, vmax=0, interpolation='bilinear', alpha=1)
        plt.imshow(result_interp_fine2, extent=extent2, aspect='auto', cmap=cmap, origin='lower', vmin=-12, vmax=0, interpolation='bilinear', alpha=1)
        plt.imshow(result_interp_fine3, extent=extent3, aspect='auto', cmap=cmap, origin='lower', vmin=-12, vmax=0, interpolation='bilinear', alpha=1)
        plt.imshow(result_interp_fine4, extent=extent4, aspect='auto', cmap=cmap, origin='lower', vmin=-12, vmax=0, interpolation='bilinear', alpha=1)

        # Set the background color of the figure
        plt.gca().set_facecolor('lightgrey')
        
        cbar = plt.colorbar(orientation='horizontal', shrink=0.8, pad=0.09, ticks=[-12, -8, -6, 0])
        # Set custom tick labels
        cbar.ax.set_xticklabels(['-12', '-8', '-6', '0'])
        # Concatenate all arrays to find the overall min and max values
        cbar.set_label('Amplitude (dB)', fontsize=17)
        cbar.ax.tick_params(labelsize=16)

        # Set the x and y limits to cover the entire extent of all zones
        x_min = min(extent1[0], extent2[0], extent3[0], extent4[0])
        x_max = max(extent1[1], extent2[1], extent3[1], extent4[1])
        y_min = min(extent1[2], extent2[2], extent3[2], extent4[2])
        y_max = max(extent1[3], extent2[3], extent3[3], extent4[3])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        # Add labels and title
        plt.xlabel('X-axis (feet)', fontsize=18)
        plt.ylabel('Y-axis (feet)', fontsize=18)
        plt.title('Reinforcing Reflection Amplitude on the entire bridge on the entire bridge', fontsize=25)

        # Add legend for untested area
        untested_patch = mpatches.Patch(color='lightgrey', label="Untested\nArea")

        plt.legend(handles=[untested_patch], loc='lower right', bbox_to_anchor=(1.05, -0.278), fontsize=16, frameon=False, handlelength=4, handleheight=4)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.gca().set_aspect(2, adjustable='box')
        
        # Show the plot
        plt.show()
        pass
    
    def contour_interpolate_entire(self, lists):
        """
        Plot a contour map for the entire bridge by interpolating the rebar points from all zones.
        
        Parameters:
        - lists (dict): A dictionary containing lists of rebar points for all zones.

        """
        x_coord = []
        y_coord = []
        z_coord = []

        # Access lists using dictionary
        for i in range(1, 5):
            x_points_list, y_points_list, z_points_list = lists[f'0{i}']

            x_coord.extend([array for sublist in x_points_list for array in sublist])
            y_coord.extend([array for sublist in y_points_list for array in sublist])
            z_coord.extend([array for sublist in z_points_list for array in sublist])

        # Increase the resolution by creating a finer mesh for each dataset
        X_fine = np.linspace(min(np.concatenate(x_coord)), max(np.concatenate(x_coord)), 300)
        Y_fine = np.linspace(min(np.concatenate(y_coord)), max(np.concatenate(y_coord)), 300)
        X_mesh_fine, Y_mesh_fine = np.meshgrid(X_fine, Y_fine)

        # Interpolate the data on the finer mesh using linear interpolation for each dataset
        result_interp_fine = griddata((np.concatenate(x_coord), np.concatenate(y_coord)),
                                       np.concatenate(z_coord), (X_mesh_fine, Y_mesh_fine), method='linear')
        
        extent = (X_mesh_fine.min(), X_mesh_fine.max(), Y_mesh_fine.min(), Y_mesh_fine.max())
        
        plt.figure(figsize=(20, 12))

        # Define custom colormap with specific color segments
        colors = [(0, 'red'),        # From -12 dB (data_min) to -8 dB (red to orange)
                  (1/3, 'orange'),   # From -8 dB to -6 dB (orange to yellow)
                  (1/2, 'yellow'),   # From -6 dB to 0 dB (yellow to lime)
                  (1, 'lime')]       # From 0 dB (data_max)
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        
        # Plot each zone individually using imshow
        plt.imshow(result_interp_fine, extent=extent, aspect='auto', cmap=cmap, interpolation='bilinear', origin='lower', alpha=1, vmin=-12, vmax=0)

        # Set the background color of the figure
        plt.gca().set_facecolor('lightgrey')
        
        cbar = plt.colorbar(orientation='horizontal', shrink=0.8, pad=0.09, ticks=[-12, -8, -6, 0])
        # Set custom tick labels
        cbar.ax.set_xticklabels(['-12', '-8', '-6', '0'])
        # Concatenate all arrays to find the overall min and max values
        cbar.set_label('Amplitude (dB)', fontsize=17)
        # Create a colorbar with adjusted size and position
        cbar.ax.tick_params(labelsize=16)

        plt.xlim(X_mesh_fine.min(), X_mesh_fine.max())
        plt.ylim(Y_mesh_fine.min(), Y_mesh_fine.max())

        # Add labels and title
        plt.xlabel('X-axis (feet)', fontsize=18)
        plt.ylabel('Y-axis (feet)', fontsize=18)
        plt.title('Reinforcing Reflection Amplitude on the entire bridge', fontsize=25)

        # Add legend for untested area
        untested_patch = mpatches.Patch(color='lightgrey', label="Untested\nArea")

        plt.legend(handles=[untested_patch], loc='lower right', bbox_to_anchor=(1.05, -0.278), fontsize=16, frameon=False, handlelength=4, handleheight=4)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.gca().set_aspect(2, adjustable='box')
        
        # Show the plot
        plt.show()
        pass
    
    def contour_scan_area_only_normal(self, lists):
        """
        Plot contour maps for each zone separately without interpolating the gap between zones.
        
        Parameters:
        - lists (dict): A dictionary containing lists of rebar points for all zones.

        """
        x_coords = {}
        y_coords = {}
        z_coords = {}

        # Access lists using dictionary
        for i in range(1, 5):
            x_points_list, y_points_list, z_points_list = lists[f'0{i}']
            
            # Store coordinates in dictionaries
            x_coords[f'x_coord_0{i}'] = [array for sublist in x_points_list for array in sublist]
            y_coords[f'y_coord_0{i}'] = [array for sublist in y_points_list for array in sublist]
            z_coords[f'z_coord_0{i}'] = [array for sublist in z_points_list for array in sublist]

        # Increase the resolution by creating a finer mesh for each dataset
        X_fine_01 = np.linspace(min(np.concatenate(x_coords['x_coord_01'])), max(np.concatenate(x_coords['x_coord_01'])), 300)
        Y_fine_01 = np.linspace(min(np.concatenate(y_coords['y_coord_01'])), max(np.concatenate(y_coords['y_coord_01'])), 300)
        X_mesh_fine_01, Y_mesh_fine_01 = np.meshgrid(X_fine_01, Y_fine_01)

        X_fine_02 = np.linspace(min(np.concatenate(x_coords['x_coord_02'])), max(np.concatenate(x_coords['x_coord_02'])), 300)
        Y_fine_02 = np.linspace(min(np.concatenate(y_coords['y_coord_02'])), max(np.concatenate(y_coords['y_coord_02'])), 300)
        X_mesh_fine_02, Y_mesh_fine_02 = np.meshgrid(X_fine_02, Y_fine_02)

        X_fine_03 = np.linspace(min(np.concatenate(x_coords['x_coord_03'])), max(np.concatenate(x_coords['x_coord_03'])), 300)
        Y_fine_03 = np.linspace(min(np.concatenate(y_coords['y_coord_03'])), max(np.concatenate(y_coords['y_coord_03'])), 300)
        X_mesh_fine_03, Y_mesh_fine_03 = np.meshgrid(X_fine_03, Y_fine_03)

        X_fine_04 = np.linspace(min(np.concatenate(x_coords['x_coord_04'])), max(np.concatenate(x_coords['x_coord_04'])), 300)
        Y_fine_04 = np.linspace(min(np.concatenate(y_coords['y_coord_04'])), max(np.concatenate(y_coords['y_coord_04'])), 300)
        X_mesh_fine_04, Y_mesh_fine_04 = np.meshgrid(X_fine_04, Y_fine_04)


        # Interpolate the data on the finer mesh using linear interpolation for each dataset
        result_interp_fine1 = griddata((np.concatenate(x_coords['x_coord_01']), np.concatenate(y_coords['y_coord_01'])),
                                       np.concatenate(z_coords['z_coord_01']), (X_mesh_fine_01, Y_mesh_fine_01), method='linear')

        result_interp_fine2 = griddata((np.concatenate(x_coords['x_coord_02']), np.concatenate(y_coords['y_coord_02'])),
                                       np.concatenate(z_coords['z_coord_02']), (X_mesh_fine_02, Y_mesh_fine_02), method='linear')

        result_interp_fine3 = griddata((np.concatenate(x_coords['x_coord_03']), np.concatenate(y_coords['y_coord_03'])),
                                       np.concatenate(z_coords['z_coord_03']), (X_mesh_fine_03, Y_mesh_fine_03), method='linear')

        result_interp_fine4 = griddata((np.concatenate(x_coords['x_coord_04']), np.concatenate(y_coords['y_coord_04'])),
                                       np.concatenate(z_coords['z_coord_04']), (X_mesh_fine_04, Y_mesh_fine_04), method='linear')

        # Determine the combined extent
        extent1 = (X_mesh_fine_01.min(), X_mesh_fine_01.max(), Y_mesh_fine_01.min(), Y_mesh_fine_01.max())
        extent2 = (X_mesh_fine_02.min(), X_mesh_fine_02.max(), Y_mesh_fine_02.min(), Y_mesh_fine_02.max())
        extent3 = (X_mesh_fine_03.min(), X_mesh_fine_03.max(), Y_mesh_fine_03.min(), Y_mesh_fine_03.max())
        extent4 = (X_mesh_fine_04.min(), X_mesh_fine_04.max(), Y_mesh_fine_04.min(), Y_mesh_fine_04.max())
        plt.figure(figsize=(20, 12))

        # Plot each zone individually using imshow
        plt.imshow(result_interp_fine1, extent=extent1, aspect='auto', cmap='gist_rainbow', interpolation='bilinear', origin='lower', alpha=1, vmin=-12, vmax=0)
        plt.imshow(result_interp_fine2, extent=extent2, aspect='auto', cmap='gist_rainbow', interpolation='bilinear', origin='lower', alpha=1, vmin=-12, vmax=0)
        plt.imshow(result_interp_fine3, extent=extent3, aspect='auto', cmap='gist_rainbow', interpolation='bilinear', origin='lower', alpha=1, vmin=-12, vmax=0)
        plt.imshow(result_interp_fine4, extent=extent4, aspect='auto', cmap='gist_rainbow', interpolation='bilinear', origin='lower', alpha=1, vmin=-12, vmax=0)

        # Set the background color of the figure
        plt.gca().set_facecolor('lightgrey')

        # Concatenate all arrays to find the overall min and max values
        cbar = plt.colorbar(orientation='horizontal', shrink=0.8, pad=0.09)
        cbar.set_label('Amplitude (dB)', fontsize=17)
        cbar.ax.tick_params(labelsize=16)

        # Set the x and y limits to cover the entire extent of all zones
        x_min = min(extent1[0], extent2[0], extent3[0], extent4[0])
        x_max = max(extent1[1], extent2[1], extent3[1], extent4[1])
        y_min = min(extent1[2], extent2[2], extent3[2], extent4[2])
        y_max = max(extent1[3], extent2[3], extent3[3], extent4[3])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        # Add labels and title
        plt.xlabel('X-axis (feet)', fontsize=18)
        plt.ylabel('Y-axis (feet)', fontsize=18)
        plt.title('Reinforcing Reflection Amplitude on the entire bridge on the entire bridge', fontsize=25)

        # Add legend for untested area
        untested_patch = mpatches.Patch(color='lightgrey', label="Untested\nArea")

        plt.legend(handles=[untested_patch], loc='lower right', bbox_to_anchor=(1.05, -0.278), fontsize=16, frameon=False, handlelength=4, handleheight=4)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.gca().set_aspect(2, adjustable='box')
        
        # Show the plot
        plt.show()
        pass
    
    def contour_interpolate_entire_normal(self, lists):
        """
        Plot a contour map for the entire bridge by interpolating the rebar points from all zones.
        
        Parameters:
        - lists (dict): A dictionary containing lists of rebar points for all zones.

        """
        x_coord = []
        y_coord = []
        z_coord = []

        # Access lists using dictionary
        for i in range(1, 5):
            x_points_list, y_points_list, z_points_list = lists[f'0{i}']

            x_coord.extend([array for sublist in x_points_list for array in sublist])
            y_coord.extend([array for sublist in y_points_list for array in sublist])
            z_coord.extend([array for sublist in z_points_list for array in sublist])

        # Increase the resolution by creating a finer mesh for each dataset
        X_fine = np.linspace(min(np.concatenate(x_coord)), max(np.concatenate(x_coord)), 300)
        Y_fine = np.linspace(min(np.concatenate(y_coord)), max(np.concatenate(y_coord)), 300)
        X_mesh_fine, Y_mesh_fine = np.meshgrid(X_fine, Y_fine)

        # Interpolate the data on the finer mesh using linear interpolation for each dataset
        result_interp_fine = griddata((np.concatenate(x_coord), np.concatenate(y_coord)),
                                       np.concatenate(z_coord), (X_mesh_fine, Y_mesh_fine), method='linear')
        
        extent = (X_mesh_fine.min(), X_mesh_fine.max(), Y_mesh_fine.min(), Y_mesh_fine.max())
        plt.figure(figsize=(20, 12))

        # Plot each zone individually using imshow
        plt.imshow(result_interp_fine, extent=extent, aspect='auto', cmap='gist_rainbow', interpolation='bilinear', origin='lower', alpha=1, vmin=-12, vmax=0)

        # Set the background color of the figure
        plt.gca().set_facecolor('lightgrey')

        # Create a colorbar with adjusted size and position
        cbar = plt.colorbar(orientation='horizontal', shrink=0.8, pad=0.09)
        cbar.set_label('Amplitude (dB)', fontsize=17)
        cbar.ax.tick_params(labelsize=16)

        plt.xlim(X_mesh_fine.min(), X_mesh_fine.max())
        plt.ylim(Y_mesh_fine.min(), Y_mesh_fine.max())

        # Add labels and title
        plt.xlabel('X-axis (feet)', fontsize=18)
        plt.ylabel('Y-axis (feet)', fontsize=18)
        plt.title('Reinforcing Reflection Amplitude on the entire bridge', fontsize=25)

        # Add legend for untested area
        untested_patch = mpatches.Patch(color='lightgrey', label="Untested\nArea")

        plt.legend(handles=[untested_patch], loc='lower right', bbox_to_anchor=(1.05, -0.278), fontsize=16, frameon=False, handlelength=4, handleheight=4)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.gca().set_aspect(2, adjustable='box')
        
        # Show the plot
        plt.show()
        pass