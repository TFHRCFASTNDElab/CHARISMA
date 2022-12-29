import sys
import pandas as pd
import numpy as np
import plotly.express as px
from plotly import io
import math


pautcolorscale = [
    [0, 'rgb(255,255,255)'],
    [0.066, 'rgb(184,212,244)'],
    [0.133, 'rgb(113,170,233)'],
    [0.2, 'rgb(62,105,190)'],
    [0.266, 'rgb(14,37,143)'],
    [0.333, 'rgb(27,72,129)'],
    [0.4, 'rgb(59,140,127)'],
    [0.466, 'rgb(126,187,94)'],
    [0.533, 'rgb(211,223,45)'],
    [0.6, 'rgb(241,211,43)'],
    [0.666, 'rgb(222,156,80)'],
    [0.733, 'rgb(209,121,87)'],
    [0.8, 'rgb(205,116,49)'],
    [0.866, 'rgb(194,98,23)'],
    [0.933, 'rgb(167,50,26)'],
    [1, 'rgb(145,12,29)']
]


def bscan(file):
    length = int(input('What is the length?\n'))
    header_size = int(input('What is the Header Size?\n'))
    
    df = pd.read_excel(file, header = None)
    
    paut_index = list(np.arange(0, len(df), length))
    scan_start = float(df.iloc[3][0].split("=")[1])
    scan_quantity = float(df.iloc[4][0].split("=")[1])
    scan_resolution = float(df.iloc[5][0].split("=")[1])
    scan_end = scan_start + ((scan_quantity - 1) * scan_resolution)
    
    angles = []
    for j in list(np.arange(45,71,1)):
        angles.append(0.94 * math.cos(math.radians(j)))
    
    for i in range(0, len(paut_index)):
        temp_df = df.iloc[paut_index[i]+header_size: paut_index[i]+length].T
        fig = px.imshow(temp_df, x = list(np.linspace(scan_start,scan_end,(length-header_size))), y = list(map(lambda k: k * angles[i], list(np.linspace(0, 320, 320)))), text_auto = False, color_continuous_scale = pautcolorscale, zmin=0, zmax=100, aspect = 'auto')
        fig.update_coloraxes(showscale=False)
        fig.update_traces(hovertemplate="<br>".join(["x: %{x}","y: %{y}", "Amplitude: %{z}" ]), name = '')
        io.write_image(fig, 'bscan - ' + str(i) + '.png')
        
def cscan(file):
    length = int(input('What is the length?\n'))
    header_size = int(input('What is the Header Size?\n'))
    
    df = pd.read_excel(file, header = None)
    
    paut_index = list(np.arange(0, len(df), length))
    scan_start = float(df.iloc[3][0].split("=")[1])
    scan_quantity = float(df.iloc[4][0].split("=")[1])
    scan_resolution = float(df.iloc[5][0].split("=")[1])
    scan_end = scan_start + ((scan_quantity - 1) * scan_resolution)
    
    cscan = []
    for i in range(0, len(paut_index)):
        cscan_row = []
        temp_df = df.iloc[paut_index[i]+header_size: paut_index[i]+length].T
        
        for p in range(paut_index[i]+header_size,paut_index[i]+length):
            cscan_row.append(max(temp_df[p]))   
        cscan.insert(0, cscan_row)
        
    cscan_index = np.linspace(26,1,26)
    cscan_df = pd.DataFrame(cscan, index = cscan_index)
    fig = px.imshow(cscan_df, x = list(np.linspace(scan_start,scan_end,(length-header_size))), text_auto = False, color_continuous_scale = pautcolorscale, zmin=0, zmax=100, aspect = 'auto')
    fig.update_traces(hovertemplate="<br>".join(["x: %{x}","y: %{y}", "Amplitude: %{z}" ]), name = '')
    fig.update_yaxes(autorange=True)
    io.write_image(fig, 'cscan.png')
        
def sscan(file):
    length = int(input('What is the length?\n'))
    header_size = int(input('What is the Header Size?\n'))
    s_scan_val = int(input('Enter the A-scan to view \n'))
    skew = int(input('Enter Skew \n'))
    
    df = pd.read_excel(file, header = None)
    
    paut_index = list(np.arange(0, len(df), length))
    angle_offset = [0, 0.1782, 0.3564, 0.5508, 0.729, 0.9072, 1.1016, 1.2798, 1.458, 1.6362, 1.8144, 1.9926, 2.187, 2.349, 2.5272, 2.7054, 2.8836, 3.0456, 3.2076, 3.3858, 3.5478, 3.6936, 3.8556, 4.0014, 4.1634, 4.3092]
    
    s_scan = []
    for i in range(0, len(paut_index)):
        temp_df = df.iloc[paut_index[i]+header_size: paut_index[i]+length].T
        s_scan.append(list(temp_df[+s_scan_val+paut_index[i]+header_size]))
    
    x = np.linspace(0,319,320)
    y = np.linspace(0,-319,320)
    fig = px.scatter(x = ((x/math.sqrt(2)) * 0.9728) - 86.3, y = (y/math.sqrt(2)) * -0.9728, color=s_scan[0], range_color = [0,100], color_continuous_scale=pautcolorscale)
    i = 1
    for angle in range(46, 71):
        x1 = np.round(math.sin(np.radians(angle)) * x, 4)
        y1 = np.round(-1 * math.tan(np.radians(90 - angle)) * x1, 4)
        
        xhalf = np.round(math.sin(np.radians(angle - 0.5)) * x, 4)
        yhalf = np.round(-1 * math.tan(np.radians(90 - angle - 0.5)) * x1, 4)
        zhalf = np.divide([s_scan[i][x] + s_scan[i-1][x] for x in range (len(s_scan[i]))] , 2)
        
        fig.add_traces(list(px.scatter(x = ((x1 + angle_offset[i]) * 0.9728) - 86.3, y = y1 * -0.9728 , color = s_scan[i], range_color = [0,100]).select_traces()))
        fig.add_traces(list(px.scatter(x = ((xhalf + angle_offset[i]) * 0.9728) - 86.3, y = yhalf * -0.9728 , color = zhalf , range_color = [0,100]).select_traces()))
    
        i = i + 1
    
    fig.update_traces(marker=dict(size=10, symbol="square"))
    fig.update_layout({'plot_bgcolor': 'rgb(0, 0, 0)','paper_bgcolor': 'rgb(255, 255, 255)',})
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    if skew == 270:
        fig.update_xaxes(autorange='reversed')
        fig.update_yaxes(autorange='reversed')
        io.write_image(fig, 'sscan.png')
    elif skew == 90:
        fig.update_yaxes(autorange='reversed')
        io.write_image(fig, 'sscan.png')

if __name__ == "__main__":
    for a in sys.argv:
        if a == '-i':
            file = sys.argv[2]
            
    scan_type = str(input('Enter the scan type: b or c or s \n'))
    if scan_type == 'b':
        bscan(file)
    elif scan_type == 'c':
        cscan(file)
    elif scan_type == 's':
        sscan(file)
    else:
        print('scan type error')
                