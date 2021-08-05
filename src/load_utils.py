import pandas as pd
from pathlib import Path
from .features.signal import lofar

import numpy as np
import scipy
import soundfile as sf

import re

datasets_path = Path('datasets')

def read_audio_file(filepath):
        signal, fs = sf.read(filepath, always_2d=True)  
        return signal, fs

def get_id(filepath):
    str_id = re.match(r"[0-9]+(?=_)", filepath.name).group(0)
    return int(str_id)
def filter_id(filepath, id_list):
    return get_id(filepath) in id_list

def load_info(dataset_name):
    if dataset_name == "4classes":
        datapath = datasets_path / "4classes"
        info = dict()
        for folder in filter(lambda x: x.is_dir(), datapath.iterdir()):
            info[folder.name] = dict()
            for file in filter(lambda x: ".wav" in x.name, folder.iterdir()):
                info[folder.name][file.name.replace(".wav", "")] = file
        return info, ("Ship Class", "Run Id", )
    elif dataset_name == "os-training":
        datapath = datasets_path / "os-training"
        info = dict()
        for folder in filter(lambda x: x.is_dir(), datapath.iterdir()):
            info[folder.name] = dict()
            for file in filter(lambda x: ".wav" in x.name, folder.iterdir()):
                info[folder.name][file.name.replace(".wav", "")] = file
        return info, ("Ship Class", "Run Id", )
    elif dataset_name == "shipsear":
        infofilename = Path('info.csv')
        datasetpath = datasets_path  / 'shipsear' 
        df = pd.read_csv(f"{datasetpath / infofilename}")
        df = df.sort_values(by='Id')
        df = df.drop(['Pic', 'Localization', 'Video', 'AIS link'], axis=1)
        
        df["Ship"] = df.Name.str.replace(r'\(.*\)', '', regex=True)
        df["Ship"] = df.Ship.str.replace(r'\s+$', '', regex=True)
        df["Ship"] = df.Ship.str.replace(r'"', '', regex=True)
        df["Ship"] = df.Ship.str.replace('Motorboat Duda', 'Duda')
        df["Ship"] = df.Ship.str.replace('Motorboat2', 'Motorboat 2')
        df["Ship"] = df.Ship.str.replace('Minho uno', 'Minho Uno')
        df["Ship"] = df.Ship.str.replace('Mar de Cangas (arrives, interference', 'Mar de Cangas', regex=False)
        df.Type = df.Type.str.replace('fishboat', 'Fishboat')
        
        files = {get_id(file): file for file in (datasetpath / 'audios').iterdir()}
        
        info = dict()
        for s_class in df["Type"].unique():
            info[s_class] = dict()
            class_df = df.loc[df["Type"] == s_class]
            for ship in class_df["Ship"].unique():
                info[s_class][ship] = dict()
                ship_df = class_df.loc[class_df["Ship"] == ship]
                for run in ship_df.itertuples():
                    file_id = run.Id
                    info[s_class][ship][run.Name] = files[file_id]
        return info, ("Ship Class", "Ship Name", "Run Name")
    else:
        raise ValueError("Invalid dataset")
                    
def load_file_as_lofar(filepath, use_tpsw=True, thres=-4):
    audio, sr = read_audio_file(filepath)
    final_sr = 7350
    sxx, freq, time = lofar(audio, sr, 
          final_sr, 
          nfft=1024, 
          noverlap=0, 
          channel_axis=1,
          max_freq=None, 
          tonal_threshold=thres, # dB
          use_tpsw=use_tpsw)
    sxx = sxx.mean(axis=-1)[..., np.newaxis]
    return sxx, freq, time