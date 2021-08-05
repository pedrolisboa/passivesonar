import pandas as pd
from pathlib import Path
import sys
from modified_lofar import lofar

import numpy as np
import scipy

import sys
from offline import load_raw_data

def load_4classes(use_tpsw=True, thres=-4):
    time_data = load_raw_data('./../../acoustic_lane/4classes', run_pattern=r'[0-9]+$')

    sxxs = list()
    for data in time_data:
        sxx, freq, time = lofar(data.data, sr=data.sr, final_sr=7350, use_tpsw=use_tpsw, tonal_threshold=thres)
        sxxs.append((sxx, (data.ship_class, data.id),))

    return sxxs, ("Ship Class", "Run Id", )

import soundfile as sf
def read_audio_file(filepath):
        signal, fs = sf.read(filepath, always_2d=True)  
        return signal, fs

import re
def get_id(filepath):
    str_id = re.match(r"[0-9]+(?=_)", filepath.name).group(0)
    return int(str_id)
def filter_id(filepath, id_list):
    return get_id(filepath) in id_list

def load_info(dataset_name):
    if dataset_name == "4classes":
        datapath = Path("./../../datasets/4classes")
        info = dict()
        for folder in filter(lambda x: x.is_dir(), datapath.iterdir()):
            info[folder.name] = dict()
            for file in filter(lambda x: ".wav" in x.name, folder.iterdir()):
                info[folder.name][file.name.replace(".wav", "")] = file
        return info, ("Ship Class", "Run Id", )
    elif dataset_name == "os-training":
        datapath = Path("./../../datasets/os-training/")
        info = dict()
        for folder in filter(lambda x: x.is_dir(), datapath.iterdir()):
            info[folder.name] = dict()
            for file in filter(lambda x: ".wav" in x.name, folder.iterdir()):
                info[folder.name][file.name.replace(".wav", "")] = file
        return info, ("Ship Class", "Run Id", )
    elif dataset_name == "shipsear":
        infofilename = Path('info.csv')
        datasetpath = Path('../..') / 'datasets' / 'shipsear' 
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
    print("AUDO AUDO AUDO UAO", audio.shape)
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


def load_shipsear(use_tpsw=True, thres=-4): 
    infofilename = Path('info.csv')
    datasetpath = Path('../..') / 'datasets' / 'shipsear' 

    def min2sec(x):
        m, s = x.split(':')
        return int(m)*60 + int(s)

    def count_ships(g):
        print(g.name)
        print(g[["Ship", "Duration"]])

    def get_valid_ships(datasetpath, infofilename):
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

        df['Seconds'] = df['Duration'].apply(min2sec)

        df = df.loc[df.Type != 'Dredger']
        df = df.loc[df.Type != 'Natural ambient noise']
        df = df.loc[df.Type != 'Pilot ship']
        df = df.loc[df.Type != 'Sailboat']
        df = df.loc[df.Type != 'Trawler']
        df = df.loc[df.Type != 'Tugboat']

        df = df.loc[df.Ship != 'Motorboat1']
        df = df.loc[df.Ship != 'Fishboat 1']
        df = df.loc[df.Ship != 'Mussel boat1']

        return df


    df = get_valid_ships(datasetpath, infofilename)

    import re
    def get_id(filepath):
        str_id = re.match(r"[0-9]+(?=_)", filepath.name).group(0)
        return int(str_id)

    def filter_id(filepath, id_list):
        return get_id(filepath) in id_list

    id_list = df.Id.values
    filtered_files = list(filter(lambda x: filter_id(x, id_list), (datasetpath / 'audios').iterdir()))
    filtered_files

    import soundfile as sf

    def read_audio_file(filepath):
        signal, fs = sf.read(filepath, always_2d=True)  
        return signal, fs

    def load_raw_data(filepaths):
        signal = [(get_id(filepath), read_audio_file(filepath))
                   for filepath in filepaths]
        return signal

    audios = load_raw_data(filtered_files)

    final_sr = 7350
    lofar_data = list()
    for id_ship, (audio, sr) in audios:
        sxx, freq, time = lofar(audio, sr, 
              final_sr, 
              nfft=1024, 
              noverlap=0, 
              channel_axis=1,
              max_freq=None, 
              tonal_threshold=thres, # dB
              use_tpsw=use_tpsw)

        sxx = sxx.squeeze()
        ship_class = str(df.loc[df.Id == id_ship, "Type"].values[0])
        ship_name = str(df.loc[df.Id == id_ship, "Ship"].values[0])
        run_name = str(df.loc[df.Id == id_ship, "Name"].values[0])
        
        lofar_data.append((sxx, (ship_class, ship_name, run_name),))
        
    return lofar_data, ("Ship Class", "Ship Name", "Run Name")