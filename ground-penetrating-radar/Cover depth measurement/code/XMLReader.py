# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:46:04 2024

@author: steve.yang.ctr
"""
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class XMLReader:
    """
    A class to read GPR scan coordinate XML files.
    
    Attributes:
    - xml_file (str): The directory path to the XML file.
    - df_coord (DataFrame): A DataFrame extracted from the XML file.
    """
    def __init__(self, xml_file):
        """
        Initializes the XMLReader object.

        Parameters:
        - xml_file (str): The directory path to the XML file.
        """
        self.xml_file = xml_file
        self.df_coord = None
        self.create_df_coord()
        self.generate_gpr_map()

    def xml_file_reader(self):
        """
        Reads the XML file and extracts start coordinates, end coordinates, and GPR file names.

        Returns:
        - start_coordinates: Cartesian coordinate of GPR starting point.
        - end_coordinates: Cartesian coordinate of GPR ending point.
        - GPR_file_names: Corresponding DZT file name on the GPR scan line.
        """
        with open(self.xml_file, 'r') as file:
            xml_data = file.read()
        
        # Parse XML data
        root = ET.fromstring(xml_data)

        # Initialize lists to store start and end coordinates
        start_coordinates = []
        end_coordinates = []
        gpr_file_names = []

        # Iterate over each GPR-Readings node
        for reading in root.findall('.//GPR-Readings'):
            # Extract start coordinates
            start_x = [float(reading.find(f"X{i}StartLocation").text) for i in range(1, 5)]
            start_y = [float(reading.find(f"Y{i}StartLocation").text) for i in range(1, 5)]
            start_coordinates.extend(zip(start_x, start_y))

            # Extract end coordinates
            end_x = [float(reading.find(f"X{i}EndLocation").text) for i in range(1, 5)]
            end_y = [float(reading.find(f"Y{i}EndLocation").text) for i in range(1, 5)]
            end_coordinates.extend(zip(end_x, end_y))

            # Extract GPR file name
            gpr_file_names.append(reading.find('GPRFileName').text)
        
        return start_coordinates, end_coordinates, gpr_file_names

    def calculate_diagonality(self, row):
        '''
        Calculate slope of GPR scan line. value of 1 indicating positive slope and -1 indicating negative slope.
        '''
        if row['Start_Y'] > row['End_Y'] and row['Scan_dir'] == 1:
            return -1
        elif row['Start_Y'] < row['End_Y'] and row['Scan_dir'] == 1:
            return 1
        elif row['Start_Y'] > row['End_Y'] and row['Scan_dir'] == 0:
            return 1
        elif row['Start_Y'] < row['End_Y'] and row['Scan_dir'] == 0:
            return -1
        else:
            return 1
    
    def create_df_coord(self):
        """
        Creates a DataFrame from the XML file.
        
        Returns:
        - df_coord: A DataFrame containing GPR scan coordinates.
        """
        start_coordinates, end_coordinates, gpr_file_names = self.xml_file_reader()
        duplicate_gpr_file_names = np.repeat(gpr_file_names, 4)
        data = {
            'GPR_FileName': duplicate_gpr_file_names,
            'Start_X': [coord[0] for coord in start_coordinates],
            'Start_Y': [coord[1] for coord in start_coordinates],
            'End_X': [coord[0] for coord in end_coordinates],
            'End_Y': [coord[1] for coord in end_coordinates],
        }
        df_coord = pd.DataFrame(data)
        columns_titles = ['GPR_FileName','Start_X', 'End_X', 'Start_Y', 'End_Y']
        df_coord = df_coord.reindex(columns=columns_titles)
        df_coord['Scan_dir'] = np.where(df_coord['Start_X'] < df_coord['End_X'], 1, 0)
        # df_coord = df_coord.sort_values('GPR_FileName')
        df_coord['Zone'] = df_coord['GPR_FileName'].str.extract(r'Region (\d+)')
        df_coord['diagonality'] = df_coord.apply(self.calculate_diagonality, axis=1)
        self.df_coord = df_coord

        return self.df_coord
    
    def generate_gpr_map(self):
        """
        Generates a GPR map plot based on df_coord.
        
        Returns:
        - Figure: Generated GPR scan map on the bridge.
        """
        # Extracting GPR zone from file names
        self.df_coord['Region'] = self.df_coord['GPR_FileName'].apply(lambda x: x.split('/')[0])
        self.df_coord['file'] = self.df_coord['GPR_FileName'].apply(lambda x: x.split('/')[2].replace('.dzt', '_3d.dzt'))
        
        zone_extents = self.df_coord.groupby('Region').agg({'Start_X': 'min', 'End_X': 'max', 'Start_Y': 'min', 'End_Y': 'max'}).reset_index()
        
        # Define colors for different zones
        zone_colors = {
            'Region 01': 'lightpink',
            'Region 02': 'lightgreen',
            'Region 03': '#FFE37A',  
            'Region 04': 'lightblue',
            'Region 05': '#FFDAB9',  # PeachPuff
            'Region 06': 'lightorange',
            'Region 07': '#98FB98',  # PaleGreen
            'Region 08': 'lightpurple',
            'Region 09': '#FFE4E1',  # MistyRose
        }

        # Create a figure and axis
        fig, ax = plt.subplots()

        for index, row in zone_extents.iterrows():
            color = zone_colors.get(row['Region'], 'lightgray')  # Default color for unknown zones
            ax.fill_between([row['Start_X'], row['End_X']], [row['Start_Y'], row['Start_Y']], [row['End_Y'], row['End_Y']], color=color, alpha=0.5)
            ax.text((row['Start_X'] + row['End_X']) / 2, (row['End_Y']) +2 , row['Region'], ha='center', va='center', fontsize=10, color='black')

            # Plot each data point
        for index, row in self.df_coord.iterrows():
            if row['Scan_dir'] == 1:
                text_x, text_y = row['Start_X'], row['Start_Y']
                ha, va = 'right', 'bottom'
            else:
                text_x, text_y = row['End_X'], row['End_Y']
                ha, va = 'left', 'bottom'
            if index % 4 == 0: 
                ax.text(text_x, text_y, (self.df_coord['file'])[index], ha=ha, va=va, fontsize=8)

            start = (row['Start_X'], row['Start_Y'])
            end = (row['End_X'], row['End_Y'])
            plt.plot([start[0], end[0]], [start[1], end[1]], color='gray', linestyle='--')

        # Scatter plot for connecting start and end coordinates
        start_coordinates = self.df_coord[['Start_X', 'Start_Y']].values
        end_coordinates = self.df_coord[['End_X', 'End_Y']].values
        plt.scatter(*zip(*start_coordinates), color='blue', label='Start Location')
        plt.scatter(*zip(*end_coordinates), color='red', label='End Location')

        # Set axis labels
        ax.set_xlabel('X-axis (feet)')
        ax.set_ylabel('Y-axis (feet)')
        ax.yaxis.set_label_position('right')
        ax.yaxis.set_ticks_position('right')
        plt.legend()
        return plt.show()