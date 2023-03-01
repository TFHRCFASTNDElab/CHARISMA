# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
from plotly import io
import plotly.express as px

hcpcolorscale = [[0, 'rgb(128,0,0)'],
                   [0.052, 'rgb(159,0,0)'],
                   [0.105, 'rgb(191,0,0)'],
                   [0.157, 'rgb(223,0,0)'],
                   [0.210, 'rgb(255,0,0)'],
                   [0.263, 'rgb(255,128,0)'],
                   [0.315, 'rgb(255,159,0)'],
                   [0.368, 'rgb(225,191,0)'],
                   [0.421, 'rgb(255,223,0)'],
                   [0.473, 'rgb(255,255,0)'],
                   [0.526, 'rgb(0,255,0)'],
                   [0.578, 'rgb(0,223,0)'],
                   [0.631, 'rgb(0,191,0)'],
                   [0.684, 'rgb(0,159,0)'],
                   [0.736, 'rgb(0,128,0)'],
                   [0.789, 'rgb(0,0,255)'],
                   [0.842, 'rgb(0,0,223)'],
                   [0.894, 'rgb(0,0,191)'],
                   [0.947, 'rgb(0,0,159)'],
                   [1, 'rgb(0,0,128)']]

def HCPMapping(filelist):
    extt = ['.xls']
    for file in filelist:
        if file.endswith(tuple(extt)):
            df = pd.read_excel(file, header = 0, index_col = 0)
            df = df.T
            fig = px.imshow(df, text_auto = False, color_continuous_scale = hcpcolorscale, zmin=-650, zmax=-50)
            fig.layout.height = df.shape[0]*40
            fig.layout.width = df.shape[1]*40
            fig.update_layout(coloraxis_colorbar=dict(len=1))
            io.write_image(fig, 'output.png')
        elif file.endswith('.json'):
            df = pd.read_json(file)
            fig = px.imshow(df, text_auto = False, color_continuous_scale = hcpcolorscale, zmin=-650, zmax=-50)
            fig.layout.height = df.shape[0]*40
            fig.layout.width = df.shape[1]*40
            fig.update_layout(coloraxis_colorbar=dict(len=1))
            io.write_image(fig, 'output.png')
            
if __name__ == "__main__":
    
    for a in sys.argv:
        if a == '-i':
            path_ = sys.argv[2]
    
    filelist_ = []
    files  = os.listdir(path_)
    ext = ['.xls', '.json']
    for file in files:
        if file.endswith(tuple(ext)):
            filelist_.append(file)        
    HCPMapping(filelist_)
