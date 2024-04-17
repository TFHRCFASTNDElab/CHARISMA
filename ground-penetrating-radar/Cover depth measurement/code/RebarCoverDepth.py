# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:50:34 2024

@author: steve.yang.ctr
"""
import GPR_locate_rebars as gpr_lr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

class RebarCoverDepth:
    def __init__(self, df_coord, zone_number, chunk_size, gpr_lanes, home_dir, window, alpha, t0, tmax, rhf_espr, vmin, vmax, num_clusters,
                 amplitude_threshold, depth_threshold, minimal_y_index, redundancy_filter, scaling_factor):
        """
        Initialize RebarCoverDepth object with specified parameters.
        
        Parameters:
        - df_coord (pandas.DataFrame): DataFrame containing coordinates and information of GPR scans.
        - zone_number (int): Zone number to process.
        - chunk_size (int): Size of chunks for data processing.
        - gpr_lanes (int): GPR lane number.
        - home_dir (str): Directory path where GPR Zone data are stored.
        - window (int): Window size for horizontal background removal.
        - alpha (float): The exponent used in gain function. 
        - t0 (float): Initial time in gain function.
        - tmax (float): Maximum time in gain function.
        - rhf_espr (float): Dielectric constant.
        - vmin, vmax (float): The parameter used in visualizing B-scans. Controls B-scan amplitude contrast ranges.
        - num_clusters (int): Number of clusters for data processing.
        - amplitude_threshold (float): Threshold for rebar amplitude detection.
        - depth_threshold (float): The maximum depth of the bridge in inches.
        - minimal_y_index (int): The initial y-axis index of the B-scan to skip the 1st positive peak from the rebar counting.
        - redundancy_filter (float): Criterion for removing the rebar points if the points are too close along the x-axis.
        - scaling_factor (float): : Used for better contrast for B-scans.
        """
        self.df_coord = df_coord
        self.zone_number = zone_number
        self.chunk_size = chunk_size
        self.gpr_lanes = gpr_lanes
        self.home_dir = home_dir
        self.window = window
        self.alpha = alpha
        self.t0 = t0
        self.tmax = tmax
        self.rhf_espr = rhf_espr
        self.vmin = vmin
        self.vmax = vmax
        self.num_clusters = num_clusters
        self.amplitude_threshold = amplitude_threshold
        self.depth_threshold = depth_threshold
        self.minimal_y_index = minimal_y_index
        self.redundancy_filter = redundancy_filter
        self.scaling_factor = scaling_factor

    def ordinal(self, n):
        """
        Convert a number into its ordinal form.
    
        Parameters:
        - n (int): The number to be converted.
    
        Returns:
        - str: The ordinal representation of the input number.
        """
        if n % 10 == 1 and n % 100 != 11:
            suffix = 'st'
        elif n % 10 == 2 and n % 100 != 12:
            suffix = 'nd'
        elif n % 10 == 3 and n % 100 != 13:
            suffix = 'rd'
        else:
            suffix = 'th'
        return str(n) + suffix

    def execute_rebar_mapping(self):
        """
        Execute the rebar mapping process.

        Returns:
        - x_points_list: x coordinates of the rebars.
        - y_points_list: y coordinates of the rebars.
        - z_points_list: z coordinates of the rebars.
        - df_chunk_list: Split raw GPR B-scans.
        - time0ed_list: GPR B-scans after time-zero correction.
        - gained_list: GPR B-scans after applying gain.
        - dewowed_list: GPR B-scans after applying dewow.
        - bgrmed_list: GPR B-scans after applying background removal.
        - migrated_list: GPR B-scans after migration. 
        - contrasted_list: GPR B-scans after contrast adjustment. 
        - located_list: x and z coordinate of located rebars (not positioned along the entire bridge).
        """
        x_points_list = []
        y_points_list = []
        z_points_list = []
        df_chunk_list = []
        time0ed_list = []
        gained_list = []
        dewowed_list = []
        bgrmed_list = []
        migrated_list = []
        contrasted_list = []
        located_list = []
        
        formatted_zone = f"{int(self.zone_number):02d}"
        downloaded_path = self.home_dir + f'GPR Zone {formatted_zone}/'
        
        grouped_df = self.df_coord.groupby('Zone')
        zone_df = grouped_df.get_group(formatted_zone)
        print("\033[1m============================ Processing Zone {formatted_zone}============================\033[0m".format(formatted_zone=formatted_zone))
        for i in range(0, self.gpr_lanes):
            print("\033[1m-------------------Processing {i} lane-------------------\033[0m".format(i=self.ordinal(i)))
            df_process = zone_df[i * 4:(i + 1) * 4]
            df_1, df_2 = self.read_csv(downloaded_path + 'csv/', index=i)
            result_variables = gpr_lr.config_to_variable(df_2)
            locals().update(result_variables)

            rhf_position = result_variables.get('rhf_position')
            rhf_range = result_variables.get('rhf_range')
            rhf_spm = result_variables.get('rhf_spm')
            rhf_sps = result_variables.get('rhf_sps')
            rh_nsamp = result_variables.get('rh_nsamp')

            IQR_df_1 = gpr_lr.Interquartile_Range(df_1)
            IQR_df_1 = IQR_df_1.astype('float64')
            data_length_feet = (IQR_df_1.shape[1] / rhf_spm) * 3.28
            print('The data length is', IQR_df_1.shape[1], 'which is {:.2f} (feet)'.format(data_length_feet))
            clipped_df_chunk = gpr_lr.data_chunk(IQR_df_1, self.chunk_size)
            located, Ppos, time0ed, gained, dewowed, bgrmed, migrated, contrasted = self.rebar_coord_and_depth(clipped_df_chunk,
                                                                                    rhf_position,
                                                                                    rhf_range,
                                                                                    self.window,
                                                                                    rhf_spm,
                                                                                    rhf_sps,
                                                                                    rh_nsamp,
                                                                                    self.alpha,
                                                                                    self.t0,
                                                                                    self.tmax,
                                                                                    self.rhf_espr,
                                                                                    self.vmin,
                                                                                    self.vmax,
                                                                                    self.num_clusters,
                                                                                    self.amplitude_threshold,
                                                                                    self.depth_threshold,
                                                                                    self.minimal_y_index,
                                                                                    self.redundancy_filter,
                                                                                    self.scaling_factor)
            df_chunk_list.append(clipped_df_chunk)
            time0ed_list.append(time0ed)
            gained_list.append(gained)
            dewowed_list.append(dewowed)
            bgrmed_list.append(bgrmed)
            migrated_list.append(migrated)
            contrasted_list.append(contrasted)
            located_list.append(located)
            
            sorted_located = self.make_consecutive_rebar(located, Ppos)
            result_data = self.remove_redundancy_filter(sorted_located, 0.6)
            offset = self.determine_min_offset(result_data, df_process, offset_range=(0, 10))
            print('The offset is:', offset, 'feet')
            result_data_parts = self.plot_2d_cover_depth(result_data, df_process, offset, i)
            x_points, y_points, z_points = self.plot_rebar_depth_contour2d(result_data_parts, 
                                                                           df_process, 
                                                                           np.array(df_process['Scan_dir'])[0], 
                                                                           i)
            x_points_list.append(x_points)
            y_points_list.append(y_points)
            z_points_list.append(z_points)
            
        return x_points_list, y_points_list, z_points_list, df_chunk_list, time0ed_list, gained_list, dewowed_list, bgrmed_list, migrated_list, contrasted_list, located_list
    
    def read_csv(self, directory, index=0):
        '''
        Reads the CSV files and convert them as Pandas dataframe (df_1 or df_2).

        Parameters:
        - directory: directory path that holds data.csv and config.csv
        - index: lane number. If index is 1, it reads data1.csv and config1.csv
        
        Returns:
        - df_1: GPR scan data
        - df_2: GPR configuration settings
        '''
        if not directory.endswith('/') and not directory.endswith('\\'):
            directory += '/'

        # Generate file names with the specified index
        filepath_data = f"{directory}{'data'}{index}.csv"
        filepath_config = f"{directory}{'config'}{index}.csv"

        df_1 = pd.read_csv(filepath_data, header=None)
        df_2 = pd.read_csv(filepath_config)

        return df_1, df_2

    def rebar_coord_and_depth(self, clipped_df_chunk, rhf_position, rhf_range,
                              win, rhf_spm, rhf_sps, rh_nsamp, alpha, t0, tmax,
                              rhf_espr, vmin, vmax, num_clusters, amplitude_threshold,
                              depth_threshold, minimal_y_index, redundancy_filter, scaling_factor=0.4):
        """
        Process rebar coordinates and depths.
        
        Parameters:
        - clipped_df_chunk (list): List of split B-scan dataframes.
        - rhf_position (float):  The starting point for measuring positions in your GPR data (ns).
        - rhf_range (float): The time it takes for the radar signals to travel to the subsurface and return (ns).
        - win (int): Window size for horizontal background removal.
        - rhf_spm (float): GPR scans per meter.
        - rhf_sps (float): GPR scans per second.
        - rh_nsamp (int): Number of rows in B-scan.
        - alpha (float): The exponent used in gain function. 
        - t0 (float): Initial time in gain function.
        - tmax (float): Maximum time in gain function.
        - rhf_espr (float): Dielectric constant.
        - vmin, vmax (float): The parameter used in visualizing B-scans. Controls B-scan amplitude contrast ranges.
        - num_clusters (int): Number of clusters for data processing.
        - amplitude_threshold (float): Threshold for rebar amplitude detection.
        - depth_threshold (float): The maximum depth of the bridge in inches.
        - minimal_y_index (int): The initial y-axis index of the B-scan to skip the 1st positive peak from the rebar counting.
        - redundancy_filter (float): Criterion for removing the rebar points if the points are too close along the x-axis.
        - scaling_factor (float, optional): Used for better contrast for B-scans.
        
        Returns:
        - located: x and z coordinate of located rebars (not positioned along the entire bridge).
        - Ppos: x-axis after migration.
        - time0ed: GPR B-scans after time zero correction.
        - gained: GPR B-scans after applying gain.
        - dewowed: GPR B-scans after applying dewow.
        - bgrmed: GPR B-scans after applying background removal.
        - migrated: GPR B-scans after migration.
        - contrasted: GPR B-scans after contrast adjustment. 
        """
        time0ed = []
        gained = []
        dewowed = []
        bgrmed = [] 
        migrated = []
        velocity = []
        Ppos = []
        contrasted = []
        located = []

        # Correctly reference the methods and attributes using self
        for i, dataframe in enumerate(clipped_df_chunk):
            time0, rh_nsamp = gpr_lr.Timezero_individual(dataframe, rhf_position, rhf_range)
            time0ed.append(time0)
            bgrmed.append(gpr_lr.bgr(time0ed[i], win=win))
            gained.append(gpr_lr.gain(bgrmed[i], type="pow", alpha=self.alpha, t0=self.t0, tmax=self.tmax))  
            dewowed.append(gpr_lr.dewow(gained[i]))

            mdf, profilePos, _, _, vel = gpr_lr.FK_migration(dewowed[i], rhf_spm, rhf_sps, rhf_range, rh_nsamp, rhf_espr)
            migrated.append(mdf)
            velocity.append(vel)
            Ppos.append(profilePos)
            contrasted.append(np.where(migrated[i] < 0, migrated[i] * scaling_factor, migrated[i]))

            # Plot_migrated_advanced(contrasted[i], Ppos[i], rhf_depth, self.rh_nsamp, 0.75)
            # Check if it's the last iteration (for the residual dataframe)
            if i == len(clipped_df_chunk) - 1:
                amplitude_threshold = self.amplitude_threshold
                if clipped_df_chunk[i].shape[1] > 300:
                    num_clusters = self.num_clusters
                else:
                    num_clusters = 20
            else:
                amplitude_threshold = self.amplitude_threshold
                num_clusters = self.num_clusters

            located.append(gpr_lr.locate_rebar_consecutive(contrasted[i], velocity[i], rhf_range, rh_nsamp, profilePos,
                                                         vmin=vmin, vmax=vmax, amplitude_threshold=amplitude_threshold, 
                                                         depth_threshold=depth_threshold, minimal_y_index=minimal_y_index, 
                                                         num_clusters=num_clusters, random_state=42, redundancy_filter=redundancy_filter))

        return located, Ppos, time0ed, gained, dewowed, bgrmed, migrated, contrasted

    def make_consecutive_rebar(self, located , Ppos):
        """
        Make rebar position consecutive along x-axis (recover the split dataframe into one).
        
        Parameters:
        - located (list): List of rebar coordinates.
        - Ppos (list): List of migrated x-axis.
        
        Returns:
        - sorted_located (list): List of sorted consecutive rebar coordinates.
        """
        sorted_located = [arr[arr[:, 0].argsort()] for arr in located]

        for i in range(0, len(sorted_located)):
            #Meter to inch conversion
            sorted_located[i][:,0] += i * np.max(Ppos[0])* 39.37
        return sorted_located

    def remove_redundancy_filter(self, sorted_located, threshold=0.9):
        """
        Criterion for removing the centroid points if they are too close along the x-axis.
        
        Parameters:
        - sorted_located (list): List of sorted consecutive rebar coordinates.
        - threshold (float, optional): Threshold value. Defaults to 0.9 (inches).
        
        Returns:
        - numpy.ndarray: Filtered rebar coordinates.
        """
        filtered_located = []

        for k in range(len(sorted_located)):
            input_to_filter = sorted_located[k]
            sorted_data = input_to_filter[input_to_filter[:, 0].argsort()]

            result = []
            i = 0

            while i < len(sorted_data) - 1:
                current_point = sorted_data[i]
                next_point = sorted_data[i + 1]

                if current_point[0] == next_point[0]:
                    # Skip points with the same x-value
                    i += 1
                    continue

                if next_point[0] - current_point[0] >= threshold:
                    result.append(current_point)
                else:
                    # Skip consecutive redundancies
                    while i < len(sorted_data) - 1 and next_point[0] - current_point[0] < threshold:
                        i += 1
                        next_point = sorted_data[i]

                    # Compare y value and save the shallower one
                    if current_point[1] < next_point[1]:
                        result.append(current_point)
                    else:
                        result.append(next_point)

                i += 1

            if i == len(sorted_data) - 1:
                result.append(sorted_data[-1])

            filtered_located.append(np.array(result))

        return np.vstack(filtered_located)

    def determine_min_offset(self, result_data, df_coord, offset_range=(0, 10), max_x_difference_threshold=80):
        """
        Determine minimum offset to adjust the GPR scan distance along with the coordinate dataframe.
        
        Parameters:
        - result_data (numpy.ndarray): Filtered rebar coordinate array.
        - df_coord (pandas.DataFrame): DataFrame containing coordinates and information of GPR scans.
        - offset_range (tuple, optional): Range of offset. Defaults to (0, 10).
        - max_x_difference_threshold (int, optional): Maximum X difference threshold. Defaults to 80. If exceeds, it prints out a message.
        
        Returns:
        - min_offset (int): Optimized offset value.
        """
        min_offset = None
        min_max_x = float('inf')
        result_data_feet = np.array([result_data[:, 0] / 12, result_data[:, 1]]).T

        for offset in range(offset_range[0], offset_range[1] + 1):
            x_distance = np.abs(df_coord['End_X'] - df_coord['Start_X'])
            x_distance = x_distance - offset
            split_points = np.cumsum(x_distance)
            split_points = np.array(split_points)

            # Create a deep copy of result_data_feet
            result_data_parts = [part.copy() for part in np.split(result_data_feet, np.searchsorted(result_data_feet[:, 0], split_points[:-1]))]

            # Adjust the parts based on split_points
            for i in range(1, len(result_data_parts)):
                result_data_parts[i][:, 0] = result_data_parts[i][:, 0] - split_points[i - 1]

            # Perform your calculations on result_data_parts
            max_x_values = [np.max(part[:, 0]) for part in result_data_parts]
            max_x = max(max_x_values)

            if max_x < min_max_x:
                min_max_x = max_x
                min_offset = offset

        # Detect anomalies based on differences
        differences = np.diff(max_x_values)
        anomaly_indices = np.where(differences > max_x_difference_threshold)[0]

        if anomaly_indices.size > 0:
            print("Wrong GPR distance measurement detected (x-axis). Define the x-axis in the df_coord for this lane manually.")

        return min_offset

    def plot_2d_cover_depth(self, result_data, df_process, offset, lane_number):
        """
        Draws 2D scatter plots for one consecutive line and along four independent GPR scan lines.

        Parameters:
        - result_data (numpy.ndarray): Filtered rebar coordinate array.
        - df_process (pandas.DataFrame): DataFrame containing coordinates and information of GPR scans.
        - offset (int): Optimized offset value.
        - lane_number (int): Lane number.

        Returns:
        - result_data_parts: List of rebar coordinates along the four GPR scan lines.
        """
        x_values = result_data[:, 0]/12
        z_values = result_data[:, 1]

        # Plot the data
        plt.plot(x_values, z_values, marker='o', linestyle='-', color='b', markersize=4, label='Data Points')

        # Add labels and title
        plt.xlabel('X-axis (feet)')
        plt.ylabel('Depth (inch)')
        plt.gca().invert_yaxis()
        plt.title('Rebar Cover Depth on Lane {:02d} (4 GPR Lines consecutive)'.format(int(lane_number+1)))
        plt.ylim(0, 20)
        plt.gca().invert_yaxis()
        # Show the plot
        plt.show()

        x_distance =  np.abs(df_process['End_X']-df_process['Start_X'])
        x_distance = x_distance - offset # offset (distance data is wrong)
        split_points = np.cumsum(x_distance)
        split_points = np.array(split_points)

        result_data_feet = np.array([result_data[:, 0] / 12, result_data[:, 1]]).T
        # Split the result_data_feet array into parts
        result_data_parts = np.split(result_data_feet, np.searchsorted(result_data_feet[:, 0], split_points[:-1]))

        result_data_parts[1][:, 0] = result_data_parts[1][:, 0] - split_points[0]
        result_data_parts[2][:, 0] = result_data_parts[2][:, 0] - split_points[1]
        result_data_parts[3][:, 0] = result_data_parts[3][:, 0] - split_points[2]

        point_size = 10
        # Plotting result_data_parts[0]
        plt.scatter(result_data_parts[0][:, 0], result_data_parts[0][:, 1], label='Line 1', s=point_size)

        # Plotting result_data_parts[1]
        plt.scatter(result_data_parts[1][:, 0], result_data_parts[1][:, 1], label='Line 2', s=point_size)

        # Plotting result_data_parts[2]
        plt.scatter(result_data_parts[2][:, 0], result_data_parts[2][:, 1], label='Line 3', s=point_size)

        # Plotting result_data_parts[3]
        plt.scatter(result_data_parts[3][:, 0], result_data_parts[3][:, 1], label='Line 4', s=point_size)

        # Adding labels and legend
        plt.xlabel('X-axis (feet)')
        plt.ylabel('Depth (inch)')
        plt.title('Rebar Cover Depth on Lane {:02d} (4 GPR Lines separate)'.format(int(lane_number+1)))
        plt.legend()
        plt.ylim(0, 20)
        plt.gca().invert_yaxis()
        # Show the plot
        plt.show()
        return result_data_parts

    def plot_rebar_depth_contour2d(self, result_data_parts, df_process, scandir, lane_number):
        """
        Generate a 2D contour plot of rebar depth with respect to lane coordinates.
        
        Parameters:
        - result_data_parts (list): A list containing the rebar data for each part of the lane.
        - df_process (DataFrame): A DataFrame containing the processed data for the lane.
        - scandir (int): The scan direction.
        - lane_number (int): The lane number.
        
        Returns:
        - x_points (list): List of arrays containing the X coordinates of rebar points.
        - y_points (list): List of arrays containing the Y coordinates of rebar points.
        - z_points (list): List of arrays containing the rebar depth values.
        """
        X = np.linspace(np.min(df_process[['Start_X', 'End_X']].values), np.max(df_process[['Start_X', 'End_X']].values), 200)
        Y = np.linspace(np.min(df_process[['Start_Y', 'End_Y']].values), np.max(df_process[['Start_Y', 'End_Y']].values), 40)
        
        x_points = []
        y_points = []
        z_points = []
        
        # Create a 2D grid using meshgrid
        X_grid, Y_grid = np.meshgrid(X, Y)

        # Initialize an array to store the results
        result_grid = np.zeros((len(Y), len(X)))

        # Loop through each row in GPR0 and fill the corresponding part of the grid
        for i in range(len(np.array(df_process['Start_X']))):
            # Find the indices corresponding to the Y linspace for this row
            if df_process.iloc[i]['Scan_dir'] == 1:
                y_indices = np.where((Y >= np.array(df_process['End_Y'])[i]) & (Y <= np.array(df_process['Start_Y'])[i]))[0]
            elif df_process.iloc[i]['Scan_dir'] == 0:
                y_indices = np.where((Y >= np.array(df_process['Start_Y'])[i]) & (Y <= np.array(df_process['End_Y'])[i]))[0]
            else:
                y_indices = None
            
            # Use the X values from result_data_parts[i][:, 0]
            if df_process.iloc[i]['Scan_dir'] == 1:
                x_values = result_data_parts[i][:, 0] + np.array(df_process['Start_X'])[i]
            elif df_process.iloc[i]['Scan_dir'] == 0:
                x_values = result_data_parts[i][:, 0] + np.array(df_process['End_X'])[i]
            else:
                x_values = None
            x_points.append(x_values)
            z_points.append(result_data_parts[i][:, 1]) 
            
            if df_process.iloc[i]['diagonality'] == 1:
                lin_y=np.linspace(Y[y_indices].min(), Y[y_indices].max(), x_values.shape[0])
            elif df_process.iloc[i]['diagonality'] == -1:
                lin_y=np.linspace(Y[y_indices].min(), Y[y_indices].max(), x_values.shape[0])[::-1]
            else:
                lin_y=np.linspace(Y[y_indices].min(), Y[y_indices].max(), x_values.shape[0])
            y_points.append(lin_y)

            # Find the corresponding indices in the X and Y grid
            x_indices = np.searchsorted(X, x_values, side='right') - 1
            y_indices = np.searchsorted(Y, lin_y, side='right') - 1
            result_grid[y_indices, x_indices] = result_data_parts[i][:, 1]  # Use the second column as Z values
                
        # Collect non-zero points
        nonzero_indices = np.nonzero(result_grid)

        # Create an array of indices corresponding to the non-zero values
        points = np.column_stack((X_grid[nonzero_indices], Y_grid[nonzero_indices]))

        # Extract non-zero values
        values = result_grid[nonzero_indices]

        # Interpolate using griddata
        result_grid_interp = griddata(points, values, (X_grid, Y_grid), method='linear')

        # Plot the contour plot with interpolated values
        plt.figure(figsize=(15, 4))

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlabel('X-axis (feet)')
        plt.ylabel('Y-axis (feet)')
        plt.imshow(result_grid_interp, extent=[X.min(), X.max(), Y.min(), Y.max()], cmap='gist_rainbow', origin='lower')
        cbar = plt.colorbar(orientation='horizontal', shrink=0.6)
        cbar.set_label('Rebar Depth (inch)')
        plt.title('Contour Plot of Rebar Depth with Lane {:02d}'.format(int(lane_number+1)))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

        return x_points, y_points, z_points
    
    def plot_combined_zone_contour(self, x_points_list, y_points_list, z_points_list, zone_number):
        """
        Generate a contour plot of combined rebar depth for a specific zone.
        
        Parameters:
        - x_points_list (list): A list containing arrays of X coordinates for rebar depth points.
        - y_points_list (list): A list containing arrays of Y coordinates for rebar depth points.
        - z_points_list (list): A list containing arrays of rebar depth values.
        - zone_number (int): The zone number for which the contour plot is generated.
        """
        # Vertically stack the arrays
        x_coord = [array for sublist in x_points_list for array in sublist]
        y_coord = [array for sublist in y_points_list for array in sublist]
        z_coord = [array for sublist in z_points_list for array in sublist]

        x_flat = np.concatenate(x_coord)
        y_flat = np.concatenate(y_coord)
        z_flat = np.concatenate(z_coord)

        # Increase the resolution by creating a finer mesh
        X_fine = np.linspace(np.min(x_flat), np.max(x_flat), 300)
        Y_fine = np.linspace(np.min(y_flat), np.max(y_flat), 300)
        X_mesh_fine, Y_mesh_fine = np.meshgrid(X_fine, Y_fine)

        # Interpolate the data on the finer mesh using linear interpolation
        result_interp_fine = griddata((x_flat, y_flat),
                                      z_flat, (X_mesh_fine, Y_mesh_fine), method='linear')

        # Plot the contour using imshow and interpolation='bilinear'
        plt.imshow(result_interp_fine, extent=(X_fine.min(), X_fine.max(), Y_fine.min(), Y_fine.max()),
                   aspect='auto', cmap='gist_rainbow', interpolation='bilinear', origin='lower')

        # Add labels and title
        plt.xlabel('X-axis (feet)')
        plt.ylabel('Y-axis (feet)')
        plt.title('Rebar Cover Depth Contour on Zone {:02d}'.format(int(self.zone_number)))

        # Show the colorbar
        plt.gca().set_aspect('equal', adjustable='box')

        # Show the colorbar
        cbar = plt.colorbar(orientation='horizontal')
        cbar.set_label('Rebar Depth (inch)')

        # Show the plot
        plt.show()