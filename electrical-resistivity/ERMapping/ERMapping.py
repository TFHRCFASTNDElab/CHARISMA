# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:44:51 2023

@author: Rahul.Torlapati
"""
import sys
import os
import pandas as pd
from plotly import io
import plotly.express as px

ercolorscale = [
    [0, 'rgb(127,0,3)'],
    [0.052, 'rgb(255,1,2)'],
    [0.105, 'rgb(255,126,0)'],
    [0.157, 'rgb(254,191,0)'],
    [0.210, 'rgb(253,255,0)'],
    [0.263, 'rgb(0,129,0)'],
    [0.315, 'rgb(0,189,0)'],
    [0.368, 'rgb(0,255,0)'],
    [0.421, 'rgb(0,0,252)'],
    [0.473, 'rgb(0,0,252)'],
    [0.526, 'rgb(0,0,233)'],
    [0.578, 'rgb(0,0,233)'],
    [0.631, 'rgb(0,0,233)'],
    [0.684, 'rgb(0,0,197)'],
    [0.736, 'rgb(0,0,197)'],
    [0.789, 'rgb(0,0,173)'],
    [0.842, 'rgb(0,0,147)'],
    [0.894, 'rgb(0,0,147)'],
    [0.947, 'rgb(0,0,143)'],
    [1, 'rgb(4,3,114)'],
]

def ERMapping(filelist):
    extt = ['.csv','.txt']
    for file in filelist:
        if file.endswith(tuple(extt)):
            df = pd.read_csv(path_+'/'+file, header = None)
            df = df[0].str.split("\t",expand=True)
            df = df.T
            df = pd.DataFrame(df, dtype="float")
            fig = px.imshow(df, text_auto = False, color_continuous_scale = ercolorscale, zmin=0, zmax=100)
            fig.update_layout(coloraxis_colorbar=dict(len=0.72))
            io.write_image(fig, 'output.png')
        elif file.endswith('.json'):
            df = pd.read_json(path_+'/'+file)
            fig = px.imshow(df, text_auto = False, color_continuous_scale = ercolorscale, zmin=0, zmax=100)
            fig.update_layout(coloraxis_colorbar=dict(len=0.72))
            io.write_image(fig, 'output.png')
            
if __name__ == "__main__":
    
    for a in sys.argv:
        if a == '-i':
            path_ = sys.argv[2]
    
    filelist_ = []
    files  = os.listdir(path_)
    ext = ['.csv', '.txt' , '.json']
    for file in files:
        if file.endswith(tuple(ext)):
            filelist_.append(file)        
    ERMapping(filelist_)
