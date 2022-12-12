#!/usr/bin/env/python3
# -*- coding: utf-8 -*-
import os
from glob import glob
from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from natsort import natsorted


def peakFrequencyMapping(filelist, outputfilename, f_s, flag):
    
    for entry in filelist:
        f = open(path+'/'+entry, "r")
        data = f.read()
        data = data.split('\n')
        if data[-1] == "":
            data = data[:-1]
        data_arr = []
        for i in range(0,len(data)):
                temp = data[i].split('\t')
                data_arr.append(float(temp[1]))
        entry = entry.strip('.txt')
        entry = entry.split('-')
        YLocation.append(entry[0]) #get x location from filename
        XLocation.append(entry[1]) #get y location from filename
        data_detrended = signal.detrend(data_arr) #linear detrending
        Amplitude = np.fft.fft(data_detrended)
        freq = np.fft.fftfreq(data_detrended.shape[-1]) * f_s
        for i in range(0,f_s):
            if abs(Amplitude[i]) == max(abs(Amplitude[0:f_s])):
                FrequencyVal.append(freq[i]) #peak freqeuncy value
    
    index = natsorted(list(set(YLocation)))
    cols = natsorted(list(set(XLocation)))
    fdf = pd.DataFrame(np.reshape(FrequencyVal, (9,28)), index = index, columns = cols)
    cmap = sns.cm.rocket_r
    f, ax = plt.subplots(figsize=(12,4))
    plt.axis('on')
    if flag == 1:
        ax = sns.heatmap(fdf, cbar = True, annot=True, cmap=cmap, cbar_kws={'label': 'kHz'})
    else:
        ax = sns.heatmap(fdf, cbar = True, annot=False, cmap=cmap, cbar_kws={'label': 'kHz'})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    ax.figure.axes[-1].yaxis.label.set_size(10)
    ax.set_xticklabels(cols, fontsize=10)
    ax.set_yticklabels(index, fontsize=10)
    f.savefig(outputfilename+'.png')
    
    
if __name__ == "__main__":
    
    for a in sys.argv:
        if a == '-i':
            path_ = sys.argv[2]
        if a == '-o':
            outputfilename_ = str(sys.argv[4])
        if a == '-f':
            f_s_ = sys.argv[6]
        if a == '-a':
            annotation_ = sys.argv[8]
        elif:
            print("Flag error")
    
    filelist_ = []
    files  = os.listdir(path_)
    ext = [".dat", ".txt"]
    for file in files:
        if file.endswith(tuple(ext)):
            filelist_.append(file)  
    YLocation = []
    XLocation = []
    FrequencyVal = []
    
    peakFrequencyMapping(filelist_, outputfilename_, f_s_, annotation_)


