#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 18:49:09 2019

@author: mjknight
"""

from pydub import AudioSegment
import numpy as np
from scipy.fftpack import fft, fftshift
from os import getcwd
from pydub.utils import mediainfo
from os import path
import pandas as pd

class SoundSegment:
    def __init__(self):
        self.f_name = None
        self.t_win = 5 # segment length seconds
        self.sig = 1 # sigma in gaussian window function
        self.max_segments = 100 # max 100 segments from 1 file
        self.AudioObject = None
        self.downsample_pts = 5000 # retain this many pts in downsampled fft
        self.downsample_rate = 10 # downsample by this integer factor in fft
        
        # These will be calculated - NOT user settable!
        self.window_fun = None
        self.freq = None
        self.freq_downsampled = None
        
    def generateSegments(self,**kwargs):
        # just use channel zero
        # kwargs["save_segments"] = true or {false} for saving .mp3 segments
        # kwargs["segments_dir"] = directory to save .mp3 segments
        
        save_segments = False
        if "save_segments" in kwargs:
            save_segments = kwargs["save_segments"]
            
        segments_dir = None
        if "segments_dir" in kwargs:
            segments_dir = kwargs["segments_dir"]
        if (save_segments) and (segments_dir is None):
            segments_dir = getcwd()
            
        mi=mediainfo(self.f_name)
        br = mi["bit_rate"]
        bn = path.basename(self.f_name)
        sn = path.splitext(bn)[0]
        
        
        mf = int(self.AudioObject.frame_rate/2)
        self.freq = np.linspace(-mf, mf, self.AudioObject.frame_rate*self.t_win)
        ix0=int(len(self.freq)/2)
        self.freq_downsampled=self.freq[ix0:][0::self.downsample_rate][0:self.downsample_pts]
        self.sig = self.t_win / 10
        
        segs = None
        seginfo = []
        
        for i in range(self.max_segments):
            ix0 = i*self.t_win*1000
            ix1 = ix0+self.t_win*1000
            if ix1<self.AudioObject.duration_seconds*1000 and i<self.max_segments:
                if save_segments:
                    fn = path.join(segments_dir,str(i)+"-"+sn+".mp3")
                    seg = self.AudioObject[ix0:ix1]
                    seg.export(out_f=fn, format="mp3",bitrate=br)
                    seg = np.array(seg.split_to_mono()[0].get_array_of_samples())
                    seg_info = {"SegmentName":fn, "SongName":bn, "StartTime":ix0/1000,"EndTime":ix1/1000}
                    seginfo = seginfo + seg_info
                else:
                    seg = np.array(self.AudioObject[ix0:ix1].split_to_mono()[0].get_array_of_samples())
                    seg_info = {"SegmentName":None, "SongName":bn, "StartTime":ix0/1000,"EndTime":ix1/1000}
                    seginfo.append(seg_info)
                fseg = self.getFreqDomain(seg).astype(int)
                if segs is None:
                    segs = fseg
                else:
                    segs = np.vstack((segs,fseg))
            else:
                break;
                
        segInfo = pd.DataFrame(seginfo)
        return segs,segInfo
        
    def readFile(self):
        self.AudioObject = AudioSegment.from_file(self.f_name)
        
    def getFreqDomain(self,seg):
        tt = np.linspace(-int(self.t_win/2),int(self.t_win/2),num=len(seg))
        g_window = np.exp(-tt**2)
        A = fft(seg*g_window) / (len(g_window)/2.0)
        
        A = fftshift(A)
        ix0=int(len(A)/2)
        a = np.abs(A[ix0:])[0::self.downsample_rate][0:self.downsample_pts]
        
        return a