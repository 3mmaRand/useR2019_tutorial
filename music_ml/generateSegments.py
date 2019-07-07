#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 19:39:40 2019

@author: mjknight
"""
import os
from SoundSegment import SoundSegment
import pandas as pd

# Directories to use
dirs_to_use = ["test"]

SegInfo = []

for d in dirs_to_use:
    songs = [f for f in os.listdir(d) if f.endswith(".m4a")]
    for s in songs:
        ss = SoundSegment()
        ss.f_name = os.path.join(d,s)
        ss.readFile()
        segs,segInfo = ss.generateSegments()
        bn = os.path.basename(s)
        sn = os.path.splitext(bn)[0]
        segInfo.to_excel(os.path.join(d,sn+"_SegmentInfo_test.xlsx"))
        pd.DataFrame(segs).to_excel(os.path.join(d,sn+"_segments_test.xlsx"))