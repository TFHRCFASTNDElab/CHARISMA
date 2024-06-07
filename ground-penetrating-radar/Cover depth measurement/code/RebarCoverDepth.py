import pandas as pd
import numpy as np
import copy
from scipy.interpolate import griddata
from scipy.interpolate import interp2d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import GPR_locate_rebars as gpr_lr
import warnings
warnings.filterwarnings('ignore')
import matplotlib.colors as mcolors

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
        - z_points_list: The amplitude values in dB.
        - df_chunk_list: Split raw GPR B-scans.
        - time0ed_list: GPR B-scans after time-zero correction.
        - gained_list: GPR B-scans after applying gain.
        - dewowed_list: GPR B-scans after applying dewow.
        - bgrmed_list: GPR B-scans after applying background removal.
        - migrated_list: GPR B-scans after migration. 
        - contrasted_list: GPR B-scans after contrast adjustment. 
        - located_list: x and z coordinate of located rebars (not positioned along the entire bridge).
        - max_row_list: Return the row that have maximum amplitude.
        - split_dfs_list: The rebar locations with amplitude, split based on the XML GPS data.
        - ref_tuple_list: Return the rebar points that have the maximum amplitude among each lane.
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
        split_dfs_list = []

        formatted_zone = f"{int(self.zone_number):02d}"
        downloaded_path = self.home_dir + f'GPR Zone {formatted_zone}/'
        
        grouped_df = self.df_coord.groupby('Zone')
        zone_df = grouped_df.get_group(formatted_zone)
        print("\033[1m============================ Processing Zone {formatted_zone}============================\033[0m".format(formatted_zone=formatted_zone))
        for i in range(0, self.gpr_lanes):
            print("\033[1m-------------------Processing {i} lane-------------------\033[0m".format(i=self.ordinal(i)))
            df_process = zone_df[i * 4:(i + 1) * 4]
            df_1, df_2 = self.read_csv(downloaded_path + 'csv/', index=i)
            # Check if i is odd
            if i % 2 != 0:
                # Flip x-axis for df_1
                df_1 = df_1.iloc[:, ::-1]

                # Reset column index starting from 0
                df_1.columns = range(len(df_1.columns))
            result_variables = gpr_lr.config_to_variable(df_2)
            locals().update(result_variables)

            rhf_position = result_variables.get('rhf_position')
            rhf_range = result_variables.get('rhf_range')
            rhf_spm = result_variables.get('rhf_spm')
            rhf_sps = result_variables.get('rhf_sps')
            rh_nsamp = result_variables.get('rh_nsamp')
            rh_nsamp = int(rh_nsamp)
            IQR_df_1 = gpr_lr.Interquartile_Range(df_1)
            IQR_df_1 = IQR_df_1.astype('float64')
            data_length_feet = (IQR_df_1.shape[1] / rhf_spm) * 3.28
            print('The data length is', IQR_df_1.shape[1], 'which is {:.2f} (feet)'.format(data_length_feet))
            clipped_df_chunk = gpr_lr.data_chunk(IQR_df_1, self.chunk_size)
            located, Ppos, velocity, time0ed, gained, dewowed, bgrmed, migrated, contrasted = self.rebar_coord_and_depth(clipped_df_chunk,
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
            
            updated_located = self.update_located_with_peaks(time0ed, located)
            self.plot_migrated_data_with_rebar(contrasted, updated_located)
            db_df = self.extract_cover_depth(updated_located, velocity, rhf_range, rh_nsamp, Ppos, time0ed, self.chunk_size)
            result_data = self.make_consecutive_rebar_unitless(updated_located, self.chunk_size)
            offset = self.determine_min_offset_unitless(result_data, rhf_spm, df_process, offset_range=(0, 10))
            print('The offset is:', offset, 'feet')
            split_dfs, split_points = self.split_cover_depth_df(db_df, df_process, IQR_df_1, rhf_spm, offset, i)
                        
            split_dfs_list.append(split_dfs)
            result_data_parts, result_data_consecutive, split_points_ft = self.plot_2d_cover_depth_discrete(result_data, df_process, IQR_df_1, offset, i)
            x_points, y_points, z_points = self.plot_rebar_depth_contour2d(split_dfs, df_process, np.array(df_process['Scan_dir'])[0], i)

            x_points_list.append(x_points)
            y_points_list.append(y_points)
            z_points_list.append(z_points)
        
        return x_points_list, y_points_list, z_points_list, df_chunk_list, time0ed_list, gained_list, dewowed_list, bgrmed_list, migrated_list, contrasted_list, located_list, split_dfs_list
    
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
        df_2 = pd.read_csv(filepath_config, index_col=0)
    
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
            gained.append(gpr_lr.gain(bgrmed[i], type="pow", alpha=alpha, t0=t0, tmax=tmax))  
            dewowed.append(gpr_lr.dewow(gained[i]))
    
            mdf, profilePos, _, _, vel = gpr_lr.FK_migration(dewowed[i], rhf_spm, rhf_sps, rhf_range, rh_nsamp, rhf_espr)
            migrated.append(mdf)
            velocity.append(vel)
            Ppos.append(profilePos)
            contrasted.append(np.where(migrated[i] < 0, migrated[i] * scaling_factor, migrated[i]))
    
            # Plot_migrated_advanced(contrasted[i], Ppos[i], rhf_depth, self.rh_nsamp, 0.75)
            # Check if it's the last iteration (for the residual dataframe)
            if i == len(clipped_df_chunk) - 1:
                amplitude_threshold = amplitude_threshold
                if clipped_df_chunk[i].shape[1] > 300:
                    num_clusters = num_clusters
                else:
                    num_clusters = 15
            else:
                amplitude_threshold = amplitude_threshold
                num_clusters = num_clusters
    
            located.append(gpr_lr.locate_rebar_consecutive_discrete(contrasted[i], velocity[i], rhf_range, rh_nsamp, profilePos,
                                                         vmin=vmin, vmax=vmax, amplitude_threshold=amplitude_threshold, 
                                                         depth_threshold=depth_threshold, minimal_y_index=minimal_y_index, 
                                                         num_clusters=num_clusters, random_state=42, redundancy_filter=redundancy_filter))
    
        return located, Ppos, velocity, time0ed, gained, dewowed, bgrmed, migrated, contrasted
    
    def update_located_with_peaks(self, time0ed, located, prominence=1, distance=40, width=5):
        """
        Update the 'located' array by replacing the second element in each sub-array with the index of the peak.
        
        Parameters:
        - time0ed: List of DataFrames containing the z values
        - located: List of arrays containing the rebar coordinates
        - prominence: Prominence parameter for find_peaks
        - distance: Distance parameter for find_peaks
        - width: Width parameter for find_peaks
        
        Returns:
        - updated_located: List of arrays with updated peak indices
        """
        updated_located = copy.deepcopy(located)
        
        for tz, rebar in zip(time0ed, updated_located):
            x = tz.columns.values
            y = tz.index.values
    
            # Get the z values
            z = tz.values
            
            # Create the interpolation function
            interp_func = interp2d(x, y, z, kind='linear')
            
            for i in rebar:

                # Interpolate the specific row
                interp_row = interp_func(i[0], x)
                interp_row_1d = interp_row.flatten()
                
                # Find peaks
                peaks, _ = find_peaks(interp_row_1d, prominence=prominence, distance=distance, width=width)
                if len(peaks) > 0:
                    # Check if the difference between i[1] and peaks[0] is above 10
                    if abs(i[1] - peaks[0]) > 10:
                        i[1] = i[1]  # Set to default value if the difference is above 10
                    else:
                        i[1] = peaks[0]
                else:
                    i[1] = i[1]

        return updated_located
    
    def custom_minmax_scale(self, data, new_min, new_max):
        min_val = data.min()
        max_val = data.max()
        scaled_data = ((data - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
        return scaled_data
    
    def plot_migrated_data_with_rebar(self, migrated, located, new_min=0, new_max=1):
        """
        Plot migrated data as a heatmap and overlay rebar locations as scatter points.
        
        Parameters:
        - migrated: List of 2D arrays representing the migrated data
        - located: List of 2D arrays representing rebar locations
        - new_min: Minimum value for normalization
        - new_max: Maximum value for normalization
        """
        for mg, rebar in zip(migrated, located):
            fig, ax = plt.subplots(figsize=(15, 12))
    
            # Normalize the data
            normalized_migrated_data = self.custom_minmax_scale(mg, new_min, new_max)
    
            # Plot the heatmap with custom colormap
            cmap = plt.cm.Greys_r
            norm = mcolors.Normalize(vmin=0.15, vmax=0.7)
            heatmap = ax.imshow(normalized_migrated_data, cmap=cmap, norm=norm)
            cbar = plt.colorbar(heatmap, ax=ax, shrink=0.5)
    
            temp_rebar = np.array(rebar)
    
            # Overlay scatter points at filtered cluster centers
            ax.scatter(temp_rebar[:, 0], temp_rebar[:, 1],
                       c='red', marker='o', s=30, edgecolors='black')
    
            # Set labels for axes
            ax.set_xlabel('GPR Survey line', fontsize=20)
            ax.set_ylabel('Depth', fontsize=20)
            cbar.ax.tick_params(labelsize=14)  # Adjust the font size as needed

            # Set font size for axis labels and ticks
            ax.tick_params(axis='both', which='major', labelsize=16)  # Adjust the font size as needed
            plt.gca().set_aspect('0.5', adjustable='box')
            plt.show()
    
    def extract_cover_depth(self, updated_located, velocity, rhf_range, rh_nsamp, Ppos, time0ed, length):
        
        meter_to_feet = 3.28084
        meter_to_inch = 39.37
        updated_located_feet =[]
        full_df_list = []
        
        for i in range(len(updated_located)):
        
            depth = (velocity[i]/2) * rhf_range # one way travel (m)
            depth_per_point = depth / rh_nsamp # (m)
            depth_axis = np.linspace(0, depth_per_point * rh_nsamp * meter_to_inch, rh_nsamp)
            survey_line_axis = Ppos[i] * meter_to_feet
            updated_located_feet.append(np.column_stack((updated_located[i][:, 0] * survey_line_axis.max() / time0ed[i].shape[1],
                                                                updated_located[i][:, 1] * depth_axis.max() / time0ed[i].shape[0])))
            
            rebar_loc_x_feet = [element[:, 0] for element in updated_located_feet]
            rebar_loc_y_inch = [element[:, 1] for element in updated_located_feet] 
            
            xyz_df = pd.DataFrame(    
            {'rebar_loc_x': rebar_loc_x_feet[i],
             'rebar_loc_y': rebar_loc_y_inch[i],
            })
            full_df_list.append(xyz_df)
            
        #make rebar_x consecutive    
        for i, df in enumerate(full_df_list):
            df['rebar_loc_x'] += i * (length * (Ppos[i][1] - Ppos[i][0]) * meter_to_feet)
        db_df = pd.concat(full_df_list)
            
        return db_df
    
    def make_consecutive_rebar_unitless(self, located, length):
        """
        Make rebar position consecutive along x-axis (recover the split dataframe into one). 
        Unitless, not in inches.
        
        Parameters:
        - located (list): List of rebar coordinates.
        - length (list): length of each split B-scan segment.
        
        Returns:
        - sorted_located (numpy.ndarray): numpy array of sorted consecutive rebar coordinates.
        """
        sorted_located = [arr[arr[:, 0].argsort()] for arr in located]
        for i in range(0, len(sorted_located)):
            sorted_located[i][:,0] += i * length
        return np.vstack(sorted_located)
    
    def determine_min_offset_unitless(self, result_data, rhf_spm, df_process, offset_range=(0, 10), max_x_difference_threshold=500):
        """
        Determine minimum offset to adjust the GPR scan distance along with the coordinate dataframe.
        Unitless, not in inches.
        
        Parameters:
        - result_data (numpy.ndarray): Filtered rebar coordinate array.
        - rhf_spm: Scans-per-meter to convert units.
        - df_process (pandas.DataFrame): DataFrame containing coordinates and information of GPR scans.
        - offset_range (tuple, optional): Range of offset. Defaults to (0, 10).
        - max_x_difference_threshold (int, optional): Maximum X difference threshold. Defaults to 500. If exceeds, it prints out a message.
        
        Returns:
        - min_offset (int): Optimized offset value.
        """
        offset_range=(0, 10)
        min_offset = None
        min_max_x = float('inf')
        result_data_parts_ft = (result_data[:, 0]/ rhf_spm) * 3.28084
        
        for offset in range(offset_range[0], offset_range[1] + 1):
            x_distance = np.abs(df_process['End_X'] - df_process['Start_X'])
            x_distance = x_distance - offset
    
            split_points = np.cumsum(x_distance)
            split_points = np.array(split_points)
    
            # Create a deep copy of result_data_feet
            result_data_parts_process = [part.copy() for part in np.split(result_data, np.searchsorted(result_data_parts_ft, split_points[:-1]))]

            # Adjust the parts based on split_points
            for i in range(1, len(result_data_parts_process)):
                result_data_parts_process[i][:, 0] = result_data_parts_process[i][:, 0] - split_points[i - 1]
            # Perform your calculations on result_data_parts
            max_x_values = [np.max(part[:, 0]) for part in result_data_parts_process]
            # Calculate differences between consecutive elements
            differences = [max_x_values[i+1] - max_x_values[i] for i in range(len(max_x_values)-1)]
            
            # Adding the first element of max_x_values to the differences list
            result = [max_x_values[0]] + differences
            max_x = max(result)

            if max_x < min_max_x:
                min_max_x = max_x
                min_offset = offset
                min_max_x_values = max_x_values
    
        # Detect anomalies based on differences
        differences = np.diff(np.diff(min_max_x_values))
        anomaly_indices = np.where(differences > max_x_difference_threshold)[0]
    
        if anomaly_indices.size > 0:
            print("Wrong measurement on x_axis detected. Define the offset manually.")
            
        return min_offset
    
    def split_cover_depth_df(self, db_df, df_process, IQR_df_1, rhf_spm, offset, lane_number):
        '''
        Splits the amplitude DataFrame based on distances and rebar locations for specified lane number.
        
        Parameters:
        - db_df: DataFrame containing rebar coordinates, amplitudes, and decibel values.
        - df_process: DataFrame containing coordinates and information of GPR scans.
        - IQR_df_1: DataFrame before splitting B-scans.
        - rhf_spm: Scans-per-meter to convert units.
        - offset: Offset value to adjust distances.
        - lane_number: Lane number.
        
        Returns:
        - split_dfs: List of DataFrames, each containing a segment of the original DataFrame split by rebar locations
        - split_points: Array of split points
        '''
        # Calculate distances and split points
        x_distance = np.abs(df_process['End_X'] - df_process['Start_X'])
        x_distance = x_distance - offset
        split_points_ft = np.cumsum(x_distance)
        split_points_ft = np.array(split_points_ft)
        # Split the DataFrame based on rebar_loc_x
        gpr_x = db_df['rebar_loc_x'].values
        split_indices = np.searchsorted(gpr_x, split_points_ft[:-1])
        split_dfs = np.split(db_df, split_indices)

        # Adjust rebar_loc_x in each split DataFrame
        for i in range(1, len(split_dfs)):
            split_dfs[i]['rebar_loc_x'] -= split_points_ft[i - 1]
            
        return split_dfs, split_points_ft

    def plot_rebar_depth_contour2d(self, split_dfs, df_process, scandir, lane_number):
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
    
            # Use the X values from sections[i][:, 0]
            if df_process.iloc[i]['Scan_dir'] == 1:
                x_values = list(split_dfs[i]['rebar_loc_x'].values) + np.array(df_process['Start_X'])[i]
            elif df_process.iloc[i]['Scan_dir'] == 0:
                x_values = list(split_dfs[i]['rebar_loc_x'].values) + np.array(df_process['End_X'])[i]
            else:
                x_values = None
            x_points.append(x_values)
            z_points.append(list(split_dfs[i]['rebar_loc_y'].values)) 
    
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
            result_grid[y_indices, x_indices] = list(split_dfs[i]['rebar_loc_y'].values)  # Use the second column as Z values
        
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

    def plot_2d_cover_depth_discrete(self, result_data, df_process, IQR_df_1, offset, lane_number):
        """
        Draws 2D scatter plots for one consecutive line and along four independent GPR scan lines.
    
        Parameters:
        - result_data (numpy.ndarray): Filtered rebar coordinate array.
        - df_process (pandas.DataFrame): DataFrame containing coordinates and information of GPR scans.
        - IQR_df_1 (pandas.DataFrame): DataFrame before splitting B-scans.
        - offset (int): Optimized offset value.
        - lane_number (int): Lane number.
    
        Returns:
        - result_data_parts: List of rebar coordinates along the four GPR scan lines.
        - result_data: Consecutive list of rebar coordinates.
        - split_points_ft: Split points used to divide result_data into result_data_parts in feet.
        """
        x_values = result_data[:, 0]
        z_values = result_data[:, 1]
    
        # Plot the data
        plt.plot(x_values, z_values, marker='o', linestyle='-', color='b', markersize=4, label='Data Points')
    
        # Add labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Depth')
        plt.title('Rebar Cover Depth on Lane {:02d} (4 GPR Lines consecutive)'.format(int(lane_number+1)))
        plt.ylim(0, 200)
        plt.gca().invert_yaxis()
        # Show the plot
        plt.show()
           
        x_distance = np.abs(df_process['End_X'] - df_process['Start_X'])
        x_distance = x_distance - offset
        split_points_ft = np.cumsum(x_distance)
        split_points_ft = np.array(split_points_ft)
        split_m = split_points_ft /3.281
        unit_x_distance_m = split_m[-1]/(IQR_df_1.shape[1])
        split_points = split_m / unit_x_distance_m
        
        result_data_copy = np.copy(result_data)
        # Split the result_data_feet array into parts
        result_data_parts = np.split(result_data_copy, np.searchsorted(result_data_copy[:, 0], split_points[:-1]))
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
        plt.xlabel('X-axis')
        plt.ylabel('Depth')
        plt.title('Rebar Cover Depth on Lane {:02d} (4 GPR Lines separate)'.format(int(lane_number+1)))
        plt.legend()
        plt.ylim(0, 200)
        plt.gca().invert_yaxis()
        # Show the plot
        plt.show()
        return result_data_parts, result_data, split_points_ft
        
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
        
        plt.figure(figsize=(15, 12))
        # Plot the contour using imshow and interpolation='bilinear'
        plt.imshow(result_interp_fine, extent=(X_fine.min(), X_fine.max(), Y_fine.min(), Y_fine.max()),
                   aspect='auto', cmap='gist_rainbow', interpolation='bilinear', origin='lower')

        # Add labels and title
        plt.xlabel('X-axis (feet)', fontsize=18)
        plt.ylabel('Y-axis (feet)', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title('Rebar Cover Depth Contour on Zone {:02d}'.format(int(self.zone_number)), fontsize=25)

        # Show the colorbar
        plt.gca().set_aspect('2', adjustable='box')

        # Show the colorbar
        cbar = plt.colorbar(orientation='horizontal')
        cbar.set_label('Rebar Depth (inch)', fontsize=17)
        cbar.ax.tick_params(labelsize=16)
        # Show the plot
        plt.show()
