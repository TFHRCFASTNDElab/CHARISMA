# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:51:55 2024

@author: steve.yang.ctr
"""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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

        # Plot each zone individually using imshow
        im1 = plt.imshow(result_interp_fine1, extent=extent1, aspect='auto', cmap='gist_rainbow', interpolation='bilinear', origin='lower', alpha=1)
        im2 = plt.imshow(result_interp_fine2, extent=extent2, aspect='auto', cmap='gist_rainbow', interpolation='bilinear', origin='lower', alpha=1)
        im3 = plt.imshow(result_interp_fine3, extent=extent3, aspect='auto', cmap='gist_rainbow', interpolation='bilinear', origin='lower', alpha=1)
        im4 = plt.imshow(result_interp_fine4, extent=extent4, aspect='auto', cmap='gist_rainbow', interpolation='bilinear', origin='lower', alpha=1)

        # Set the background color of the figure
        plt.gca().set_facecolor('lightgrey')

        # Concatenate all arrays to find the overall min and max values
        combined_result = np.concatenate((result_interp_fine1, result_interp_fine2, result_interp_fine3, result_interp_fine4), axis=None)
        z_min = np.nanmin(combined_result)
        z_max = np.nanmax(combined_result)

        # Set the color limits for imshow plots
        im1.set_clim(vmin=z_min, vmax=z_max)
        im2.set_clim(vmin=z_min, vmax=z_max)
        im3.set_clim(vmin=z_min, vmax=z_max)
        im4.set_clim(vmin=z_min, vmax=z_max)

        ticks = np.linspace(z_min, z_max, 7, endpoint=True)
        ticks = np.around(ticks, decimals=2)
        ticks[-1]-=0.01
        ticks[0]+=0.01
        ticks[1:-1] = np.round(ticks[1:-1]).astype(int)
        cbar = plt.colorbar(orientation='horizontal', shrink=0.8, pad=0.09, ticks=ticks)
        cbar.set_label('Rebar Cover Depth (inch)', fontsize=17)
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
        plt.title('Rebar Cover Depth on the entire bridge', fontsize=25)

        # Add legend for untested area
        untested_patch = mpatches.Patch(color='lightgrey', label="Untested\nArea")

        plt.legend(handles=[untested_patch], loc='lower right', bbox_to_anchor=(1.05, -0.291), fontsize=16, frameon=False, handlelength=4, handleheight=4)
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

        # Plot each zone individually using imshow
        plt.imshow(result_interp_fine, extent=extent, aspect='auto', cmap='gist_rainbow', interpolation='bilinear', origin='lower', alpha=1)

        # Set the background color of the figure
        plt.gca().set_facecolor('lightgrey')

        # Create a colorbar with adjusted size and position
        z_min = np.nanmin(result_interp_fine)
        z_max = np.nanmax(result_interp_fine)
        ticks = np.linspace(z_min, z_max, 7, endpoint=True)
        ticks = np.around(ticks, decimals=2)
        ticks[-1]-=0.01
        ticks[0]+=0.01
        ticks[1:-1] = np.round(ticks[1:-1]).astype(int)
        cbar = plt.colorbar(orientation='horizontal', shrink=0.8, pad=0.09, ticks=ticks)
        cbar.set_label('Rebar Cover Depth (inch)', fontsize=17)
        cbar.ax.tick_params(labelsize=16)

        plt.xlim(X_mesh_fine.min(), X_mesh_fine.max())
        plt.ylim(Y_mesh_fine.min(), Y_mesh_fine.max())

        # Add labels and title
        plt.xlabel('X-axis (feet)', fontsize=18)
        plt.ylabel('Y-axis (feet)', fontsize=18)
        plt.title('Rebar Cover Depth on the entire bridge', fontsize=25)

        # Add legend for untested area
        untested_patch = mpatches.Patch(color='lightgrey', label="Untested\nArea")

        plt.legend(handles=[untested_patch], loc='lower right', bbox_to_anchor=(1.05, -0.291), fontsize=16, frameon=False, handlelength=4, handleheight=4)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.gca().set_aspect(2, adjustable='box')
        
        # Show the plot
        plt.show()
        pass