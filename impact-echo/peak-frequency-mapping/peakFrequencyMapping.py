#!/usr/bin/env/python3
# -*- coding: utf-8 -*-
import os
import sys
from scipy import signal
import numpy as np
import pandas as pd
from natsort import natsorted
from plotly import io
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def peakFrequencyMapping(filelist, f_s, flag):
    
    for entry in filelist:
        f = open(path_+'/'+entry, "r")
        data = f.read()
        data = data.split('\n')
        if data[-1] == "":
            data = data[:-1]
        timeSeries = []
        for i in range(0,len(data)):
                temp = data[i].split('\t')
                timeSeries.append(float(temp[1]))
        entry = entry.strip('.txt')
        entry = entry.split('-')
        YLocation.append(entry[0]) #get x location from filename.
        XLocation.append(entry[1]) #get y location from filename.
        
        #======================================================================
        
        detrendedTimeSeries = signal.detrend(timeSeries) #linear detrending.
        amplitude = np.fft.fft(detrendedTimeSeries) #fourier transforming.
        freq = np.fft.fftfreq(detrendedTimeSeries.shape[-1]) * f_s #calculating frequency using sampling rate.
        for i in range(0,f_s):
            if abs(amplitude[i]) == max(abs(amplitude[0:f_s])):
                FrequencyVal.append(freq[i]) #peak freqeuncy value.
        
        # to make changes to the way peak frequency is calculated.
        #======================================================================
    
    index = natsorted(list(set(YLocation))) #matrix column values
    cols = natsorted(list(set(XLocation))) #matirc row values
    fdf = pd.DataFrame(np.reshape(FrequencyVal, (9,28)), index = index, columns = cols) 
    if flag == 1:
        fig = px.imshow(fdf, text_auto = True) #heatmap with annotation .
    else:
        fig = px.imshow(fdf, text_auto = False)#heatmap without annotation.
    
    fig.update_layout(hoverlabel=dict(bgcolor="white",font_size=16))
    fig.update_traces(hovertemplate="<br>".join(["Y: %{y}","X: %{x}"]), name = '')
    io.write_html(fig, 'IE-Peakmap.html', include_plotlyjs = 'cdn', include_mathjax = False, auto_open = True)
    
    inp = input('To continue, Enter location for spot results or Enter 0 to Exit \n')
    while inp != 0:
        filter_object = filter(lambda a: inp in a, filelist)
        filter_list = list(filter_object)
        if len(filter_list) == 1:
            f = open(path_+'/'+filter_list[0], "r")
            data = f.read()
            data = data.split('\n')
            if data[-1] == "":
                data = data[:-1]
            timeSeries = []
            for i in range(0,len(data)):
                    temp = data[i].split('\t')
                    timeSeries.append(float(temp[1]))
            detrendedTimeSeries = signal.detrend(timeSeries) #linear detrending.
            amplitude = abs(np.fft.fft(detrendedTimeSeries))
            amplitude = amplitude[0:500]
            normalizedAmplitude = (amplitude-np.min(amplitude))/(np.max(amplitude)-np.min(amplitude))
            freq = np.fft.fftfreq(detrendedTimeSeries.shape[-1]) * f_s
            
            #==================================================================
            # plotting timeseries and frequency spectrum.
            
            fig = make_subplots(rows=2, cols=1, subplot_titles=("Time Series", "Frequency Spectrum"))
            fig.add_trace(go.Scatter(x = 1/f_s * np.arange(len(detrendedTimeSeries)), y=detrendedTimeSeries, name='', hovertemplate='Time (sec): %{x}'+'<br>Amplitude: %{y}'), col = 1, row = 1)
            fig.add_trace(go.Scatter(x=freq[0:500], y=normalizedAmplitude[0:500], mode='lines', name='', hovertemplate='Freq (Khz): %{x}'+'<br>Amplitude: %{y}'), col =1 , row =2)
            
            fig.update_xaxes(title_text="Time (sec)", row=1, col=1)
            fig.update_xaxes(title_text="Frequency (kHz)", row=2, col=1)

            fig.update_yaxes(title_text="Amplitude", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=2, col=1)
            
            fig.update_layout(height=800, width=1000 ,showlegend=False)
            
            io.write_html(fig, 'loc.html', include_plotlyjs = 'cdn', include_mathjax = False, auto_open = True)
            
            #==================================================================
            inp = input('To continue, Enter location for spot results or Enter 0 to Exit \n')
            continue
        if len(filter_list) == 0:
            print('Invalid Selection') # invalid input
            inp = input('To continue, Enter location for spot results or Enter 0 to Exit \n')
            continue
        else:
            print('End of Program')#ending the program
            break

    
        
    
if __name__ == "__main__":
        
    for a in sys.argv:
        if a == '-i':
            path_ = sys.argv[2] #-i input.
        if a == '-f':
            f_s_ = np.int32(sys.argv[4]) #-f sampling rate.
        if a == '-a':
            annotation_ = np.int32(sys.argv[6]) # -a annotation.

    
    filelist_ = []
    files  = os.listdir(path_)
    ext = [".dat", ".txt"]
    for file in files:
        if file.endswith(tuple(ext)):
            filelist_.append(file)  
    YLocation = []
    XLocation = []
    FrequencyVal = []

    peakFrequencyMapping(filelist_, f_s_, annotation_)


