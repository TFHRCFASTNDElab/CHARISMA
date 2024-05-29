# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:48:29 2024

@author: steve.yang.ctr
"""

import GPR_locate_rebars as gpr_lr
import os

class DztToCsvConverter:
    """
    A class to convert DZT files to CSV format.
    
    Attributes:
    - df_coord (DataFrame): DataFrame containing coordinates and information of GPR scans.
    - zone (str): Zone number.
    - file_path (str): Path to the directory containing DZT files.
    """
    def __init__(self, df_coord, zone, file_path):
        """
        Initialize the DztToCsvConverter object.
        
        Parameters:
        - df_coord (DataFrame): DataFrame containing coordinates and information of GPR scans.
        - zone (str): Zone number.
        - file_path (str): Path to the directory containing DZT files.
        """
        self.df_coord = df_coord
        self.zone = zone
        self.file_path = file_path

    def get_zone_df(self):
        """
        Get the DataFrame corresponding to the specified zone.
        
        Returns:
        DataFrame: DataFrame containing coordinates information for the specified zone.
        """
        formatted_zone = f"{int(self.zone):02d}"

        #self.df_coord = self.df_coord.sort_values('GPR_FileName')

        # Group by the 'Zone' column
        grouped_df = self.df_coord.groupby('Zone')

        # Access the specified zone
        zone_df = grouped_df.get_group(formatted_zone)
        zone_df = zone_df.reset_index(drop=True)
        return zone_df

    def format_filename(self, group):
        """
        Format the filename.
        
        Parameters:
        - group (DataFrame): DataFrame group containing file information.
        
        Returns:
        str: Formatted filename.
        """
        file_name = group['file'].iloc[0]  # Taking the first value as all should be same
        return f"{file_name.split('.')[0]}.DZT"

    def process_dzt_files(self):
        """
        Process DZT files and save them as CSV.

        """
        # Create CSV directory if it doesn't exist
        output_path = os.path.join(self.file_path, 'csv/')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        zone_df = self.get_zone_df()

        dzt_files = zone_df.groupby('file').apply(self.format_filename).tolist()

        # Process each DZT file
        for i, file in enumerate(dzt_files):
            file_path = os.path.join(self.file_path, file)
            # Read DZT file
            df1, df2 = gpr_lr.readdzt(file_path)

            # Generate output file names
            data_file_name = f'data{i}'
            config_file_name = f'config{i}'

            # Save Data and Config to CSV with different options
            gpr_lr.save_to_csv(df1, output_path, data_file_name, include_index=False, include_header=False)
            gpr_lr.save_to_csv(df2, output_path, config_file_name, include_index=True, include_header=True)
            