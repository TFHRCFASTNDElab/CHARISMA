import sys 
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from scipy.constants import c as c
import struct
sys.path.append('C:/directory/path/downloaded_py_files/')
import mig_fk
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import GPR_plot as plot

def readdzt(filename, minheadsize = 1024, infoareasize = 128):
    '''
    Reads DZT files and returns Pandas dataframe.
    Additional details about the header format can be found in the GSSI SIR 3000 Manual pg 55 https://www.geophysical.com/wp-content/uploads/2017/10/GSSI-SIR-3000-Manual.pdf
    Parameters:
    - filename : DZT file name including path.
    - minheadsize : The default is 1024.
    - infoareasize : The default is 128.

    Returns:
    - df1 : GPR data.
    - df2 : GPR configuration settings.

    '''
    
    fid = open(filename,'rb');
    
    rh_tag = struct.unpack('h', fid.read(2))[0]  # Pos 00 // 0x00ff if header, 0xfnff for old file
    rh_data = struct.unpack('h', fid.read(2))[0] # Pos 02 // constant 1024 (obsolete)
    rh_nsamp = struct.unpack('h', fid.read(2))[0] # Pos 04 // samples per scan
    rh_bits = struct.unpack('h', fid.read(2))[0] # Pos 06 // bits per data word (8 or 16)
    rh_zero = struct.unpack('h', fid.read(2))[0] # Pos 08 // Offset (0x80 or 0x8000 depends on rh_bits)
    rhf_sps = struct.unpack('f', fid.read(4))[0] # Pos 10 // scans per second
    rhf_spm = struct.unpack('f', fid.read(4))[0] # Pos 14 // scans per meter
    rhf_mpm = struct.unpack('f', fid.read(4))[0] # Pos 18 // meters per mark
    rhf_position = struct.unpack('f', fid.read(4))[0] # Pos 22 // position (ns)
    rhf_range = struct.unpack('f', fid.read(4))[0] # Pos 26 // range (ns)
    rh_npass = struct.unpack('h', fid.read(2))[0] # Pos 30 // num of passes for 2-D files
    rhb_cdt = struct.unpack('f', fid.read(4))[0] # Pos 32 // Creation date & time
    rhb_mdt = struct.unpack('f', fid.read(4))[0]  # Pos 36 // Last modification date & time
    rh_mapOffset = struct.unpack('h', fid.read(2))[0] # Pos 40 // offset to range gain function
    rh_mapSize = struct.unpack('h',fid.read(2))[0] # Pos 42 // size of range gain function
    rh_text = struct.unpack('h',fid.read(2))[0] # Pos 44 // offset to text
    rh_ntext = struct.unpack('h',fid.read(2))[0] # Pos 46 // size of text
    rh_proc = struct.unpack('h',fid.read(2))[0] # Pos 48 // offset to processing history
    rh_nproc = struct.unpack('h',fid.read(2))[0] # Pos 50 // size of processing history
    rh_nchan = struct.unpack('h',fid.read(2))[0] # Pos 52 // number of channels
    rhf_espr = struct.unpack('f', fid.read(4))[0] # Pos 54 // average dielectric constant
    rhf_top = struct.unpack('f',fid.read(4))[0] # Pos 58 // position in meters
    rhf_depth = struct.unpack('f',fid.read(4))[0] # Pos 62 // range in meters
    fid.close()

    # Figuring out the datatype
    if rh_data < minheadsize:
        offset = minheadsize*rh_data
    else:
        offset = minheadsize*rh_nchan   

    if rh_bits == 8:
        datatype = 'uint8' # unsigned char
    elif rh_bits == 16:
        datatype = 'uint16' # unsigned int
    elif rh_bits == 32:
        datatype = 'int32'

    # Organize the data from the reading
    vec = np.fromfile(filename,dtype=datatype)
    headlength = offset/(rh_bits/8)
    datvec = vec[int(headlength):]
    if rh_bits == 8 or rh_bits == 16:
        datvec = datvec - (2**rh_bits)/2.0
    data = np.reshape(datvec,[int(len(datvec)/rh_nsamp),rh_nsamp])
    data = np.asmatrix(data)
    data = data.transpose()
    df1 = pd.DataFrame(data)
    
    # Save the configurations
    config = {'minheadsize': minheadsize,
              'infoareasize': infoareasize,
              'rh_tag': rh_tag,
              'rh_data': rh_data,
              'rh_nsamp': rh_nsamp,
              'rh_bits': rh_bits,
              'rh_zero': rh_zero,
              'rhf_sps': rhf_sps,
              'rhf_spm': rhf_spm,
              'rhf_mpm': rhf_mpm,
              'rhf_position': rhf_position,
              'rhf_range': rhf_range,
              'rh_npass': rh_npass,
              'rhb_cdt': rhb_cdt,
              'rhb_mdt': rhb_mdt,
              'rh_mapOffset': rh_mapOffset,
              'rh_mapSize': rh_mapSize,
              'rh_text': rh_text,
              'rh_ntext': rh_ntext,
              'rh_proc': rh_proc,
              'rh_nproc': rh_nproc,
              'rh_nchan': rh_nchan,
              'rhf_espr': rhf_espr,
              'rhf_top': rhf_top,
              'rhf_depth': rhf_depth,
              } 

    index_config = ['config']
    df2 = pd.DataFrame(config, index = index_config)
    df2 = df2.transpose()
    
    return df1, df2

def save_to_csv(dataframe, directory, filename, include_index=False, include_header=False):
    '''
    Saves the Pandas dataframe (df1 or df2) from the readdzt function into CSV format.

    Parameters:
    - dataframe: df1 or df2.
    - directory: directory path to save the files.
    - filename: file name.
    - include_index: decides including indices or not, default: False.
    - include_header: decides including headers or not, default: False.

    '''
    if not directory.endswith('/') and not directory.endswith('\\'):
        directory += '/'

    filepath = f"{directory}{filename}.csv"

    # Save dataframe with specified options
    dataframe.to_csv(filepath, index=include_index, header=include_header)

def read_csv(directory):
    '''
    Reads the CSV files and convert them as Pandas dataframe (df_1 or df_2).

    Parameters:
    - directory: directory path (should be the same with the directory used in the save_to_csv function).
    
    Returns:
    - df_1: GPR scan data.
    - df_2: GPR configuration settings.
    '''
    if not directory.endswith('/') and not directory.endswith('\\'):
        directory += '/'
    
    filepath_data = f"{directory}{'data'}.csv"
    filepath_config = f"{directory}{'config'}.csv"
    
    # Read dataframe
    df_1 = pd.read_csv(filepath_data, header=None)
    df_2 = pd.read_csv(filepath_config)
    
    return df_1, df_2

def config_to_variable(df_2):
    '''
    Declares the items in df_2 (GPR configuration settings) as a variable dictionary.

    Parameters:
    - df_2: Pandas dataframe df_2 (GPR configuration settings).
    
    Returns:
    - variables_dict: a dictionary to declare variables. (locals().update(result_variables))
    '''
    # Create an empty dictionary to store variables
    variables_dict = {}

    # Allocate variables for config dataframe (df_2)
    for index, row in df_2.iterrows():
        variable_name = row['Unnamed: 0']
        variable_value = row['config']

        # Assign the variable dynamically using locals()
        locals()[variable_name] = variable_value

        # Store the variable in the dictionary
        variables_dict[variable_name] = variable_value
    
    # Adjust the wave traveling time as 0 to value (sometimes position has negative value)
    if 'rhf_position' in variables_dict and variables_dict['rhf_position'] != 0:
        wave_travel_time = variables_dict['rhf_range'] - variables_dict['rhf_position']
        variables_dict['rhf_position'] = 0
        variables_dict['rhf_range'] = wave_travel_time

    # Return the dictionary containing all variables
    return variables_dict
    
def Interquartile_Range(df_1, min_value=0.10, max_value=0.90, multiplier=1.5):
    '''
    This function clips the DataFrame to remove outliers based on the interquartile range (IQR).

    Parameters:
    - df_1: Input DataFrame (GPR scan data).
    - min_value: The lower quantile value to calculate Q1 (default is 0.10).
    - max_value: The upper quantile value to calculate Q3 (default is 0.90).
    - multiplier: A multiplier to control the range for defining outliers (default is 1.5).

    Returns:
    - clipped_df: DataFrame with outliers clipped based on the calculated bounds.

    '''
    # Calculate the first and third quartiles (Q1 and Q3)
    Q1 = df_1.quantile(min_value).min()
    Q3 = df_1.quantile(max_value).max()

    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for clipping
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    # Clip the entire DataFrame using the calculated bounds
    clipped_df = df_1.clip(lower=lower_bound, upper=upper_bound)
    clipped_df = clipped_df.astype('float64')
    return clipped_df

def data_chunk(df_1, chunk_size=300):
    '''
    Splits the input DataFrame along the GPR survey line (x-axis) into chunks.

    Parameters:
    - df_1: Input DataFrame containing GPR scan data.
    - chunk_size: The size of each chunk along the x-axis. Default is 300.

    Returns:
    - A list containing full chunks of the DataFrame, each of size chunk_size,
      and possibly a last chunk containing the remaining data.
    '''
    # Calculate the number of full chunks
    num_full_chunks = df_1.shape[1] // chunk_size

    # Split the DataFrame into full chunks along the columns and reset the index
    df_full_chunks = [df_1.iloc[:, i * chunk_size:(i + 1) * chunk_size].copy().reset_index(drop=True) for i in range(num_full_chunks)]

    # Calculate the start index of the last chunk
    last_chunk_start = num_full_chunks * chunk_size

    # Include the leftover chunk
    df_last_chunk = df_1.iloc[:, last_chunk_start:].copy().reset_index(drop=True)

    return df_full_chunks + [df_last_chunk]

def gain(df, type="exp", alpha=0.2, t0=40, tmax=None):
    """
    Apply gain to a specific time frame of the data in a DataFrame.

    Parameters:
    - df: Input DataFrame with shape (512, 300).
    - type: Type of gain ('exp' for exponential, 'pow' for power).
    - alpha: Exponent value for the gain.
    - t0: Start time of the gain application.
    - tmax: Maximum time for gain application.

    Returns:
    - df_g: DataFrame after applying gain.
    """
    t = np.arange(df.shape[0], dtype=np.float64)  # Assuming time indices are represented by column indices

    # Determine the time indices where the gain is applied
    if tmax is not None:
        mask = (t >= t0) & (t <= tmax)
    else:
        mask = (t >= t0)

    # Apply gain based on the specified type to each row in the DataFrame
    if type == "exp":
        df_g = df.copy()
        expont_df = pd.DataFrame(np.exp(alpha * (t[mask] - t0)), index=df_g.loc[mask, :].index)
        df_g.loc[mask, :] = df.loc[mask, :] * expont_df.values

    elif type == "pow":
        df_g = df.copy()
        expont_df = pd.DataFrame(((t[mask] - t0) ** alpha), index=df_g.loc[mask, :].index)
        df_g.loc[mask, :] = df.loc[mask, :] * expont_df.values
    else:
        raise ValueError("Invalid type. Use 'exp' or 'pow'.")

    return df_g

def dewow(df):
    '''
    Dewow to correct the baseline of the wave.
    Reference: https://github.com/iannesbitt/readgssi/blob/master/readgssi/functions.py

    Parameters:
    - df: DataFrame with signal data.

    Returns:
    - dewowed_df: DataFrame with dewowed signal.
    - average_predicted_values: Numpy array with average predicted values.
    '''
    # Fit the polynomial model for each column and average the predicted values
    predicted_values = np.zeros_like(df.values, dtype=float)

    for i, column in enumerate(df.columns):
        signal_column = df[column]
        model = np.polyfit(range(len(signal_column)), signal_column, 3)
        predicted_values[:, i] = np.polyval(model, range(len(signal_column)))

    average_predicted_values = np.mean(predicted_values, axis=1)

    # Apply the filter to each column
    dewowed_df = df - average_predicted_values[:, np.newaxis]

    return dewowed_df

def bgr(ar, win=0):
    '''
    Horizontal background removal. It uses the moving average method to cancel out continuous horizontal signal along size of the window. 
    
    Parameters:
    - ar: GPR B-scan Pandas dataframe.
    - win: window for uniform_filter1d.
    
    Returns:
    - ar_copy: Dataframe after background removal.
    '''
    # Make a copy of the input array
    ar_copy = ar.copy()

    window = int(win)

    # Perform background removal on the copied array
    ar_copy -= uniform_filter1d(ar_copy, size=window, mode='nearest')

    # Return the modified copy
    return ar_copy

def Timezero_mean(df_1, rhf_position, rhf_range):
    '''
    Mean time-zero correction.
    
    Parameters:
    - df_1 : GPR B-scan Pandas dataframe.
    - rhf_position : The starting point for measuring positions in your GPR data (ns).
    - rhf_range : The time it takes for the radar signals to travel to the subsurface and return (ns).

    Returns:
    - adjusted_time0data_t : dataframe after time zero correction.
    - time0linspace: The discrete time value at the mean time.
    - new_rh_nsamp : The number of rows after cut out.
    '''
    time0array = []
    # Define time space (n) along with depth dimension
    n = np.linspace(rhf_position, rhf_range, df_1.shape[0])
    # from 0 to scans
    for i in range(0, df_1.shape[1]):
        # find peaks
        temp = df_1[i]
        temp = minmax_scale(temp, feature_range=(-1, 1))
        peaks, _ = find_peaks(temp, prominence=0.1, distance = 40)

        # index of the peaks
        neg_peaks = []
        for peak in peaks:
            if peak > 100:
                neg_peaks.append(peak)
        # time0array contains the indices for time zero 
        time0array.append(n[neg_peaks[0]] -  0.06) #0.06 is the small gap for showing the first peaks
    
    # time0linspace is the average time zero index in the time space (n)
    time0linspace = next((i for i, value in enumerate(n) if value >= np.mean(time0array)), None)
    # cut out indices before the first positive peak
    time0data = df_1[time0linspace:-1] if time0linspace is not None else []
    new_rh_nsamp = time0data.shape[0]

    return time0data, time0linspace, new_rh_nsamp

def Timezero_individual(df_1, rhf_position, rhf_range):
    '''
    Scan-by-scan Time-zero correction.

    Parameters:
    - df_1 : GPR B-scan Pandas dataframe.
    - rhf_position : The starting point for measuring positions in your GPR data (ns).
    - rhf_range : The time it takes for the radar signals to travel to the subsurface and return (ns).

    Returns:
    - adjusted_time0data_t : dataframe after time zero correction.
    - new_rh_nsamp : The number of rows after cut out.
    '''
    
    first_peaks_index = []
    time0data_cutout = []
    time0linspace = []
    time0data_reindex = []
    #Define time space (n) along with depth dimension
    n = np.linspace(rhf_position, rhf_range, df_1.shape[0])
    #from 0 to scans
    for i in range(df_1.shape[1]):
        temp = df_1.iloc[:, i]  # Access the ith column using iloc
        temp = minmax_scale(temp, feature_range=(-1, 1))
        peaks, _ = find_peaks(temp, prominence=0.2, distance=15)

        #first peaks
        first_peaks_index.append(peaks[0])

        #time0linspace is the average time zero index in the time space (n)
        time0linspace.append(n[peaks[0]])

        #time0data_cutout is cut out indices before the first positive peak
        time0data_cutout.append(df_1.iloc[:, i][peaks[0]:-1])

        #new index is for time zeroing based on the 1st positive peak (depth indices are adjusted based on the 1st positive peak)
        new_index = np.arange(-peaks[0], -peaks[0] + len(df_1))

        #reindexed dataframe (without cutting)
        df_reindexed = (new_index, np.array(df_1.iloc[:, i]))
        #reindexed time0data
        time0data_reindex.append(df_reindexed)

    x=[]
    y=[]

    #Need to reindex again since the data length and order is not consistant. We cut out the uncommon indices
    for i in range (0, df_1.shape[1], 1):
        x.append(np.arange(0, len(time0data_cutout[i]),1))
        y.append(time0data_cutout[i])

    #Calculate common index based on max and min value of x
    common_range = np.arange(max(map(min, x)), min(map(max, x)) + 1)
    adjusted_time0data = []

    #Append the cut out data in the data frame 
    for i in range (0, df_1.shape[1], 1):
        df_temp = pd.DataFrame({'X': x[i], 'Y': y[i]})
        df_temp_common_range = df_temp[df_temp['X'].isin(common_range)]
        adjusted_time0data.append(np.array(df_temp_common_range['Y']))
    adjusted_time0data = pd.DataFrame(adjusted_time0data)
    adjusted_time0data_t = adjusted_time0data.transpose()
    new_rh_nsamp = adjusted_time0data_t.shape[0]
    return adjusted_time0data_t, new_rh_nsamp

def FK_migration(data, rhf_spm, rhf_sps, rhf_range, rh_nsamp, rhf_espr):
    '''
    F-K migration. Recommend changing the dielectric if the migration result is poor.
    
    Parameters: 
    - data : GPR B-scan dataframe after time-zero correction.
    - rhf_spm : Scans per meter.
    - rhf_sps : Scans per second.
    - rhf_range : The time it takes for the radar signals to travel to the subsurface and return (ns).
    - rh_nsamp : The number of rows in the DataFrame.
    - rhf_espr : Dielectric constant of the media.
    
    Returns:
    - migrated_data : Migrated DataFrame.
    - profilePos : Linear space of the x-axis (along the survey line) after migration.
    - dt : Time space interval.
    - dx : x space interval.
    - velocity : Average velocity.

    '''
    # Calculate pos_x and dx based on the scans per second or scans per meter
    if rhf_spm != 0:
        pos_x, dx = np.linspace(0.0, data.shape[1]/rhf_spm, data.shape[1], retstep=True)
        profilePos = pos_x
    else:
        time_x, dx = np.linspace(0.0, data.shape[1]/rhf_sps, data.shape[1], retstep=True)
        profilePos = time_x
    
    # Calculate the Two-way traveling time and time interval
    twtt, dt = np.linspace(0, rhf_range, int(rh_nsamp), retstep=True)
    
    # Calculate velocity based on the dielectric constant in the configuration (df2)
    velocity = (c)/math.sqrt(rhf_espr) * 1e-9 #m/ns
    
    # Apply F-K migration
    migrated_data,twtt,migProfilePos = mig_fk.fkmig(data, dt, dx, velocity)
    
    # Update Linear space of the x-axis
    profilePos = migProfilePos + profilePos[0]
    
    return migrated_data, profilePos, dt, dx, velocity


def Locate_rebar(migrated_data, rhf_depth, rh_nsamp, profilePos, amplitude_threshold = 0.70, depth_threshold = 0.15, num_clusters = 14, random_state = 42, midpoint_factor=0.4):
    '''
    Locates rebar positions in migrated data based on specified parameters. (Old version)

    Parameters:
    - migrated_data: Input DataFrame containing migrated data.
    - rhf_depth: Depth (m).
    - rh_nsamp: The number of rows in the DataFrame.
    - amplitude_threshold: Threshold for rebar amplitude detection (Gets more points when the value is low and vice versa).
    - depth_threshold: Threshold to skip the 1st positive and negative peak (Plot the migrated result first, and determine the value).
    - num_clusters: Number of clusters for k-means clustering (Count the white spots in the migrated plot, and put that number here).
    - random_state: Random seed for reproducibility in clustering (default is 42).
    - midpoint_factor: Controls the contrast of the plot. (Higher value outputs more darker plot and vice versa).

    Returns:
    - rebar_positions: DataFrame containing the detected rebar positions.

    '''

    fig, ax = plt.subplots(figsize=(15, 2))

    # Calculate depth per point and depth axis in inches
    depth_per_point = rhf_depth / rh_nsamp
    depth_axis = np.linspace(0, depth_per_point * len(migrated_data) * 39.37, len(migrated_data))
    
    # Convert survey line axis to inches
    survey_line_axis = profilePos * 39.37
    
    vmin, vmax = migrated_data.min(), migrated_data.max()
    
    # Calculate the midpoint based on the provided factor
    midpoint = vmin + (vmax - vmin) * midpoint_factor
    
    cmap = plt.cm.gray
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=midpoint, vmax=vmax)
    
    heatmap = ax.imshow(migrated_data, cmap='Greys_r', extent=[survey_line_axis.min(), survey_line_axis.max(), depth_axis.max(), depth_axis.min()], norm=norm)

    # Add a colorbar
    cbar = plt.colorbar(heatmap, ax=ax)
    
    # Normalize the data
    normalized_migrated_data = minmax_scale(migrated_data, feature_range=(-1, 1))
    
    # Create meshgrid of indices
    x_indices, y_indices = np.meshgrid(np.arange(normalized_migrated_data.shape[1]), np.arange(normalized_migrated_data.shape[0]))
    
    # Highlight data points with values higher than threshold
    threshold = amplitude_threshold  
    # Skip the first positive peak based on the depth threshold you defined
    threshold_index = int(depth_threshold * normalized_migrated_data.shape[0]) 
    highlighted_points = np.column_stack((x_indices[(normalized_migrated_data > threshold) & (y_indices > threshold_index)],
                                          y_indices[(normalized_migrated_data > threshold) & (y_indices > threshold_index)]))
    
    # Use KMeans clustering to identify representative points
    num_clusters = num_clusters  
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(highlighted_points)
    
    # Get the cluster centers
    cluster_centers_m = kmeans.cluster_centers_
    
    # Convert cluster centers to inches
    cluster_centers_inches_m = np.column_stack((cluster_centers_m[:, 0] * survey_line_axis.max() / normalized_migrated_data.shape[1],
                                              cluster_centers_m[:, 1] * depth_axis.max() / normalized_migrated_data.shape[0]))
    
    # Overlay scatter points at cluster centers
    scatter = ax.scatter(cluster_centers_inches_m[:, 0], cluster_centers_inches_m[:, 1],
                         c='red', marker='o', s=50, edgecolors='black')
    
    # Set labels for axes
    ax.set_xlabel('GPR Survey line (inch)')
    ax.set_ylabel('Depth (inch)')
    
    # Show the plot
    return plt.show()

def custom_minmax_scale(data, new_min, new_max):
    '''
    Custom minmax scale function without losing resolution. GPR B-scan amplitude resolution goes bad with scikit-learn minmax scale. 

    Parameters:
    - data: numpy array.
    - new_min: the minimum value in your normalization.
    - new_max: the maximum value in your normalization.
    
    Returns:
    - scaled_data: Normalized numpy array.
    '''
    min_val = data.min()
    max_val = data.max()
    scaled_data = ((data - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
    return scaled_data

def locate_rebar_consecutive(migrated_data, velocity, rhf_range, rh_nsamp, profilePos, 
                             vmin=0.15, vmax=0.7, amplitude_threshold=0.55, depth_threshold=8, 
                             minimal_y_index=10, num_clusters=50, random_state=42, 
                             redundancy_filter=0.6):
    '''
    Locates rebar positions in migrated data based on specified parameters.

    Parameters:
    - migrated_data: migrated GPR B-scan numpy array .
    - velocity: The wave speed in the media. (c)/math.sqrt(rhf_espr) * 1e-9 m/ns
    - rhf_range: The time it takes for the radar signals to travel to the subsurface and return (ns).
    - rh_nsamp: The number of rows in the GPR B-scan.
    - profilePos: x axis (survey line axis) after migration.
    - vmin, vmax: The parameter used in visualizing B-scans. Controls B-scan amplitude contrast ranges. 
    - amplitude_threshold: Threshold for rebar amplitude detection.
    - depth_threshold: The maximum depth of the bridge in inches.
    - minimal_y_index: The initial y-axis index of the B-scan to skip the 1st positive peak from the rebar counting.
    - num_clusters: Number of clusters for k-means clustering (Count the white spots in the migrated plot, and put that number here).
    - random_state: Random seed for reproducibility in clustering (default is 42).
    - midpoint_factor: Controls the contrast of the plot. (Higher value outputs more darker plot and vice versa).

    Returns:
    - figure: Scatter rebar points on the migrated B-scan.

    '''
    if rh_nsamp != len(migrated_data):
        raise ValueError("Length of migrated_data should be equal to rh_nsamp")

    fig, ax = plt.subplots(figsize=(15, 12))
    # Calculate depth per point and depth axis in inches
    depth = (velocity/2) * rhf_range # one way travel (m)
    depth_per_point = depth / rh_nsamp # (m)
    depth_axis = np.linspace(0, depth_per_point * rh_nsamp * 39.37, rh_nsamp)

    # Convert survey line axis to inches
    survey_line_axis = profilePos * 39.37

    new_min = 0
    new_max = 1

    # Normalize the data
    normalized_migrated_data = custom_minmax_scale(migrated_data, new_min, new_max)

     # Plot the heatmap with custom colormap
    cmap = plt.cm.Greys_r
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    heatmap = ax.imshow(normalized_migrated_data, cmap=cmap, norm=norm, extent=[survey_line_axis.min(), survey_line_axis.max(), depth_axis.max(), depth_axis.min()])
    cbar = plt.colorbar(heatmap, ax=ax, shrink=0.5)
    # Create meshgrid of indices
    x_indices, y_indices = np.meshgrid(np.arange(normalized_migrated_data.shape[1]), np.arange(normalized_migrated_data.shape[0]))
    threshold = amplitude_threshold  # Adjust this threshold based on your data
    bridge_depth = int(depth_threshold / (depth_per_point * 39.37))
    highlighted_points = np.column_stack((x_indices[(normalized_migrated_data > threshold) & (y_indices < bridge_depth) & (y_indices > minimal_y_index)],
                                          y_indices[(normalized_migrated_data > threshold) & (y_indices < bridge_depth) & (y_indices > minimal_y_index)]))

    # Use KMeans clustering to identify representative points
    num_clusters = num_clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state).fit(highlighted_points)

    # Get the cluster centers
    cluster_centers_m = kmeans.cluster_centers_

    # Convert cluster centers to inches
    cluster_centers_inches_m = np.column_stack((cluster_centers_m[:, 0] * survey_line_axis.max() / normalized_migrated_data.shape[1],
                                                cluster_centers_m[:, 1] * depth_axis.max() / normalized_migrated_data.shape[0]))

    # Remove points that are very close together and deeper, keeping only the one with the smaller y value
    filtered_cluster_centers = []
    for x, y in cluster_centers_inches_m:
        close_points = [(x2, y2) for x2, y2 in filtered_cluster_centers if abs(x2 - x) < redundancy_filter]
        if close_points:
            # There are already points close to this one
            if y < min(close_points, key=lambda p: p[1])[1]:
                # This point has a smaller y value, replace the existing ones
                filtered_cluster_centers = [p for p in filtered_cluster_centers if (abs(p[0] - x) >= redundancy_filter) or (abs(p[1] - y) >= redundancy_filter)]
                filtered_cluster_centers.append((x, y))
        else:
            # No close points yet, add this one
            filtered_cluster_centers.append((x, y))

    filtered_cluster_centers = np.array(filtered_cluster_centers)

    # Overlay scatter points at filtered cluster centers
    scatter = ax.scatter(filtered_cluster_centers[:, 0], filtered_cluster_centers[:, 1],
                         c='red', marker='o', s=30, edgecolors='black')

    # Set labels for axes
    ax.set_xlabel('GPR Survey line (inch)', fontsize=20)
    ax.set_ylabel('Depth (inch)', fontsize=20)
    ax.set_aspect(3)
    #ax.set_ylim(15, 0)
    cbar.ax.tick_params(labelsize=14)  # Adjust the font size as needed

    # Set font size for axis labels and ticks
    ax.tick_params(axis='both', which='major', labelsize=16)  # Adjust the font size as needed
    plt.show()
    return cluster_centers_inches_m