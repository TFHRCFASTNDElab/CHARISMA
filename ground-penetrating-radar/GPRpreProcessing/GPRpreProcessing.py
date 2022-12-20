#!/usr/bin/env/python3
# -*- coding: utf-8 -*-

import sys 
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
from scipy.signal import find_peaks
from statistics import mean
from scipy.constants import c as c
import struct
import mig_fk
from plotly import io
import plotly.graph_objects as go

def readdzt(filename):
    info = {}
    fid = open(filename,'rb');
    minheadsize = 1024
    infoareasize = 128
    
    rh_tag = struct.unpack('h', fid.read(2))[0]  # Pos 00
    rh_data = struct.unpack('h', fid.read(2))[0] # Pos 02
    rh_nsamp = struct.unpack('h', fid.read(2))[0] # Pos 04
    info["rh_nsamp"] = rh_nsamp
    rh_bits = struct.unpack('h', fid.read(2))[0] # Pos 06
    rh_zero = struct.unpack('h', fid.read(2))[0] # Pos 08
    rhf_sps = struct.unpack('f', fid.read(4))[0] # Pos 10
    info["rhf_sps"] = rhf_sps
    rhf_spm = struct.unpack('f', fid.read(4))[0] # Pos 14
    info["rhf_spm"] = rhf_spm
    rhf_mpm = struct.unpack('f', fid.read(4))[0] # Pos 18
    rhf_position = struct.unpack('f', fid.read(4))[0] # Pos 22
    info["rhf_position"] = rhf_position
    rhf_range = struct.unpack('f', fid.read(4))[0] # Pos 26
    info["rhf_range"] = rhf_range
    rh_npass = struct.unpack('h', fid.read(2))[0] # Pos 30
    rhb_cdt = struct.unpack('f', fid.read(4))[0] # Pos 32
    rhb_mdt = struct.unpack('f', fid.read(4))[0]  # Pos 36
    rh_mapOffset = struct.unpack('h', fid.read(2))[0] # Pos 40
    rh_mapSize = struct.unpack('h',fid.read(2))[0] # Pos 42
    rh_text = struct.unpack('h',fid.read(2))[0] # Pos 44
    rh_ntext = struct.unpack('h',fid.read(2))[0] # Pos 46
    rh_proc = struct.unpack('h',fid.read(2))[0] # Pos 48
    rh_nproc = struct.unpack('h',fid.read(2))[0] # Pos 50
    rh_nchan = struct.unpack('h',fid.read(2))[0] # Pos 52
    rhf_espr = struct.unpack('f', fid.read(4))[0] # Pos 54
    info["rhf_espr"] = rhf_espr
    rhf_top = struct.unpack('f',fid.read(4))[0] # Pos 58
    rhf_depth = struct.unpack('f',fid.read(4))[0] # Pos 62
    info["rhf_depth"] = rhf_depth
    fid.close()

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

    vec = np.fromfile(filename,dtype=datatype)
    headlength = offset/(rh_bits/8)
    datvec = vec[int(headlength):]
    if rh_bits == 8 or rh_bits == 16:
        datvec = datvec - (2**rh_bits)/2.0
    data = np.reshape(datvec,[int(len(datvec)/rh_nsamp),rh_nsamp])
    data = np.asmatrix(data)
    
    return data.transpose(), info

def timeZero(data):
    time0array = []
    n = np.linspace(0, 8, data.shape[0])
    for i in range(0, data.shape[1]):
        temp = data[i]
        temp = minmax_scale(temp, [-1,1])
        peaks, _ = find_peaks(temp, distance = 80)
        neg_peaks = []
        for peak in peaks:
            if peak > 100:
                neg_peaks.append(peak)
        time0array.append(n[neg_peaks[0]] -  0.06)
    
    time0 = []
    for i in range(0 , len(n)):
        if n[i] >= mean(time0array):
            time0.append(i)
            
    time0data = data[time0[0] : -1]
    
    return time0data, time0[0]

def fkMigration(data, hdr):
    if hdr['rhf_spm'] != 0:
        profilePos = hdr['rhf_position']+np.linspace(0.0, data.shape[1]/hdr['rhf_spm'], data.shape[1])
    else:
        profilePos = hdr['rhf_position']+np.linspace(0.0, data.shape[1]/hdr['rhf_sps'], data.shape[1])
    twtt = np.linspace(0, hdr['rhf_range'], hdr['rh_nsamp'])
    dt=twtt[3]-twtt[2]
    dx=(profilePos[-1]-profilePos[0])/(len(profilePos)-1)
    velocity = (c)/math.sqrt(hdr['rhf_espr']) * 1e-9 #m/ns
    migrated_data,twtt,migProfilePos = mig_fk.fkmig(data, dt, dx, velocity)
    profilePos = migProfilePos + profilePos[0]
    
    return migrated_data, profilePos, velocity

def tpowGain(data,twtt,power):
    factor = np.reshape(twtt**(float(power)),(len(twtt),1))
    factmat = np.matlib.repmat(factor,1,data.shape[1])  
    return np.multiply(data,factmat)


def preProcessing(file, flag):
    data, hdr = readdzt(file)
    df = pd.DataFrame(data)
    time0df, zeroTime = timeZero(df)
    if flag == '1':
        fig = go.Figure(data=go.Heatmap(z = time0df, colorscale = 'greys_r'))
        fig.update_yaxes(range=[time0df.shape[0] , 0])
        io.write_html(fig, 'output.html', include_plotlyjs = 'cdn', include_mathjax = False, auto_open = True)
    
    elif flag == '2':
        migrated_df, profilePos, velocity = fkMigration(time0df, hdr)
        depth_index = np.linspace(0, hdr['rhf_depth'], hdr['rh_nsamp'])
        migrated_depth_index = np.linspace(0, hdr['rhf_depth'] - depth_index[zeroTime], hdr['rh_nsamp'] - zeroTime - 1)
        
        layout = go.Layout(xaxis=dict(title="Profile Position (inch)"), yaxis=dict( title="Depth (inch)")) 
        fig = go.Figure(data=go.Heatmap(z = migrated_df, y = migrated_depth_index*39.37, x = profilePos*39.37, colorscale = 'greys_r'), layout = layout)
        fig.update_yaxes(range=[migrated_depth_index[-1]*39.37 , 0])
        
        io.write_image(fig, 'output.png')


if __name__ == "__main__":
    
    for a in sys.argv:
        if a == '-i':
            file = sys.argv[2]
        if a == '-f':
            flag = str(sys.argv[4])
            
    ext = [".dzt", ".DZT"]        
    if file.endswith(tuple(ext)):
        preProcessing(file, flag)