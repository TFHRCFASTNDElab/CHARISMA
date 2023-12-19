import sys 
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
from scipy.signal import find_peaks
from scipy.constants import c as c
import struct
sys.path.append('C:/directory/path/downloaded_py_files/')
import mig_fk
import plot
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def readdzt(filename, minheadsize = 1024, infoareasize = 128):
    '''
    Reads DZT files and returns Pandas dataframe.
    Additional details about the header format can be found in the GSSI SIR 3000 Manual pg 55 https://www.geophysical.com/wp-content/uploads/2017/10/GSSI-SIR-3000-Manual.pdf
    Parameters:
    - filename : DZT file name including path
    - minheadsize : The default is 1024.
    - infoareasize : The default is 128.

    Returns:
    - df1 : GPR data
    - df2 : GPR configuration settings

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

def save_to_csv(dataframe, directory, filename):
    
    if not directory.endswith('/') and not directory.endswith('\\'):
        directory += '/'
        
    filepath = f"{directory}{filename}.csv"
    
    if filename == 'data':
        dataframe.to_csv(filepath, index=False, header=False)
        
    else:
        dataframe.to_csv(filepath)


def read_csv(directory):
    
    if not directory.endswith('/') and not directory.endswith('\\'):
        directory += '/'
    
    filepath_data = f"{directory}{'data'}.csv"
    filepath_config = f"{directory}{'config'}.csv"
    
    df_1 = pd.read_csv(filepath_data, header=None)
    df_2 = pd.read_csv(filepath_config)
    
    return df_1, df_2
    
def Interquartile_Range(df_1, min_value=0.10, max_value=0.90, multiplier=1.5):
    '''
    This function clips the DataFrame to remove outliers based on the interquartile range (IQR).

    Parameters:
    - df_1: Input DataFrame
    - min_value: The lower quantile value to calculate Q1 (default is 0.10)
    - max_value: The upper quantile value to calculate Q3 (default is 0.90)
    - multiplier: A multiplier to control the range for defining outliers (default is 1.5)

    Returns:
    - clipped_df: DataFrame with outliers clipped based on the calculated bounds

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
    Split the dataframe along the column dimension
    '''
    # Split the DataFrame into chunks along the columns
    df_chunks = np.array_split(df_1, np.arange(chunk_size, df_1.shape[1], chunk_size), axis=1)
    
    # Calculate the number of splits needed
    num_splits = df_1.shape[1] // chunk_size

    # Split the DataFrame into chunks along the columns and reset the index
    df_chunks = [df_1.iloc[:, i * chunk_size:(i + 1) * chunk_size].copy().reset_index(drop=True) for i in range(num_splits)]

    # Rename the columns to have a constant index
    for i, df_chunk in enumerate(df_chunks):
        df_chunk.columns = range(chunk_size)
    return df_chunks

def power_gain_dataframe(df, type="exp", alpha=0.2, t0=90):
    '''
    Apply power gain to a specific time frame of the data in a DataFrame.
    Reference: https://emanuelhuber.github.io/RGPR/02_RGPR_tutorial_basic-GPR-data-processing/

    Parameters:
    - df: Input DataFrame
    - type: Type of power gain ('exp' for exponential, 'pow' for power)
    - alpha: Exponent value for the power gain
    - t0: Start time index of the power gain application

    Returns:
    - df_g: DataFrame after applying gain
    '''
    t = np.arange(df.shape[0], dtype=np.float64)  # Assuming time indices are represented by column indices

    # Determine the time indices where the power gain is applied
    mask = (t >= t0) 

    # Apply power gain based on the specified type to each row in the DataFrame
    if type == "exp":
        df_g = df.copy()
        expont_df = pd.DataFrame(np.exp(alpha * (t[mask] - t0)), index=df_g.loc[mask,:].index)
        df_g.loc[mask,:] = df.loc[mask,:] * expont_df.values
        
    elif type == "pow":
        df_g = df.copy()
        expont_df = pd.DataFrame(((t[mask] - t0) ** alpha), index=df_g.loc[mask,:].index)
        df_g.loc[mask,:] = df.loc[mask,:] * expont_df.values
    else:
        raise ValueError("Invalid type. Use 'exp' or 'pow'.")

    return df_g

def dewow(df):
    '''
    Dewow to correct the baseline of the wave
    Reference: https://github.com/iannesbitt/readgssi/blob/master/readgssi/functions.py

    '''

    # Initialize
    predicted_values = np.zeros_like(df.values, dtype=float)

    # Read all columns of data and fitting into polynomial function
    for i, column in enumerate(df.columns):
        signal_column = df[column]
        model = np.polyfit(range(len(signal_column)), signal_column, 3)
        predicted_values[:, i] = np.polyval(model, range(len(signal_column)))

    # Calculates averaged polynomial fit
    average_predicted_values = np.mean(predicted_values, axis=1)

    # Apply the dewow filter to each column by subtracting mean polynomial fit
    dewowed_df = df - average_predicted_values[:, np.newaxis]

    return dewowed_df

def Timezero_mean(df_1, rhf_position, rhf_range):
    '''
    Mean time-zero correction.
    
    Parameters:
    - df_1 : Input DataFrame
    - rhf_position : The starting point for measuring positions in your GPR data  (ns)
    - rhf_range : The time it takes for the radar signals to travel to the subsurface and return (ns)

    Returns:
    - adjusted_time0data_t : DataFrame after time zero correction
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

    return time0data, time0linspace

def Timezero_individual(df_1, rhf_position, rhf_range):
    '''
    Scan-by-scan Time-zero correction.

    Parameters:
    - df_1 : Input DataFrame
    - rhf_position : The starting point for measuring positions in your GPR data  (ns)
    - rhf_range : The time it takes for the radar signals to travel to the subsurface and return (ns)

    Returns:
    - adjusted_time0data_t : DataFrame after time zero correction
    '''
    
    first_peaks_index = []
    time0data_cutout = []
    time0linspace = []
    time0data_reindex = []
    #Define time space (n) along with depth dimension
    n = np.linspace(rhf_position, rhf_range, df_1.shape[0])
    #from 0 to scans(column)
    for i in range(0, df_1.shape[1]):
        temp = df_1[i]
        temp = minmax_scale(temp, feature_range=(-1, 1))
        peaks, _ = find_peaks(temp, prominence=0.1, distance = 40)

        #first peaks index
        first_peaks_index.append(peaks[0])
        
        #time0linspace is the average time zero index in the time space (n)
        time0linspace.append(n[peaks[0]])
        
        #time0data_cutout is cut out indices before the first positive peak
        time0data_cutout.append(df_1[i][peaks[0]:-1])
        
        #new index is for time zeroing based on the 1st positive peak (depth indices are adjusted based on the 1st positive peak)
        new_index = np.arange(-peaks[0], -peaks[0] + len(df_1))
        
        #reindexed dataframe (without cutting)
        df_reindexed = (new_index, np.array(df_1[i]))
        
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
    
    return adjusted_time0data_t

def FK_migration(data, rhf_spm, rhf_sps, rhf_position, rhf_range, rh_nsamp, rhf_espr):
    '''
    
    Parameters: 
    - data : Input DataFrame after time-zero correction
    - rhf_spm : Scans per meter
    - rhf_sps : Scans per second
    - rhf_position : The starting point for measuring positions in your GPR data  (ns)
    - rhf_range : The time it takes for the radar signals to travel to the subsurface and return (ns)
    - rh_nsamp : The number of rows in the DataFrame
    - rhf_espr : Dielectric constant of the media
    

    Returns:
    - migrated_data : Migrated DataFrame
    - profilePos : Linear space of the x-axis (along the survey line) after migration
    - dt : Time space interval
    - dx : x space interval
    - velocity : Average velocity

    '''
    
    # Calculate pos_x and dx based on the scans per second or scans per meter
    if rhf_spm != 0:
        pos_x, dx = np.linspace(0.0, data.shape[1]/rhf_spm, data.shape[1], retstep=True)
        profilePos = rhf_position + pos_x
    else:
        time_x, dx = np.linspace(0.0, data.shape[1]/rhf_sps, data.shape[1], retstep=True)
        profilePos = rhf_position + time_x
    
    # Calculate the Two-way traveling time and time interval
    twtt, dt = np.linspace(0, rhf_range, int(rh_nsamp), retstep=True)
    
    # Calculate velocity based on the dielectric constant in the configuration (df2)
    velocity = (c)/math.sqrt(rhf_espr) * 1e-9 #m/ns
    
    # Apply F-K migration
    migrated_data,twtt,migProfilePos = mig_fk.fkmig(data, dt, dx, velocity)
    
    # Update Linear space of the x-axis
    profilePos = migProfilePos + profilePos[0]
    
    return migrated_data, profilePos, dt, dx, velocity


def Locate_rebar(migrated_data, rhf_depth, rh_nsamp, amplitude_threshold = 0.70, depth_threshold = 0.15, num_clusters = 14, random_state = 42, midpoint_factor=0.4):
    '''
    Locates rebar positions in migrated data based on specified parameters.

    Parameters:
    - migrated_data: Input DataFrame containing migrated data
    - rhf_depth: Depth (m)
    - rh_nsamp: The number of rows in the DataFrame
    - amplitude_threshold: Threshold for rebar amplitude detection (Gets more points when the value is low and vice versa)
    - depth_threshold: Threshold to skip the 1st positive and negative peak (Plot the migrated result first, and determine the value)
    - num_clusters: Number of clusters for k-means clustering (Count the white spots in the migrated plot, and put that number here)
    - random_state: Random seed for reproducibility in clustering (default is 42)
    - midpoint_factor: Controls the contrast of the plot. (Higher value outputs more darker plot and vice versa)

    Returns:
    - rebar_positions: DataFrame containing the detected rebar positions

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