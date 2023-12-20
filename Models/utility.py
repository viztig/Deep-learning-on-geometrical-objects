import numpy as np
import pandas as pd
import open3d as o3d
import scipy
from scipy.stats import norm
import statistics
import os
import sys
import openpyxl
import warnings
warnings.filterwarnings('ignore')
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
import torch
import torch.nn as nn

def normalize_pc(pcd):
    points=torch.tensor(np.asarray([pcd.points]),dtype=torch.float32)
    centroid = torch.mean(points, axis=1)
    points -= centroid
    furthest_distance = torch.max(torch.sqrt(torch.sum(abs(points)**2,axis=-1)))
    points /= furthest_distance
    points=points.reshape(1,points.shape[2],points.shape[1])
    return points,centroid,furthest_distance

def denormalize_pc(_points,centroid,furthest_distance):
    points=torch.clone(_points)
    points[0]*=furthest_distance
    points[1]*=furthest_distance
    points[2]*=1000
    points[0]+=centroid[0][0]
    points[1]+=centroid[0][2]
    return points

def list_files(directory):
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

