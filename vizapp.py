import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from unet import custom_objects
from unet.utils import crop_to_shape
from src.load_utils import load_info, load_file_as_lofar
import matplotlib.pyplot as plt
from pathlib import Path
from src.visualization import waterfall_spectrogram

@st.cache
def load_file(filepath, use_tpsw, thres):
    return load_file_as_lofar(filepath, use_tpsw, thres)

def filter_dataset(metadata):
    metadata = metadata.copy()
    prev_meta_name = "Ship Class"
    prev_meta_value = st.sidebar.selectbox(f"Select {prev_meta_name}", metadata[prev_meta_name].unique())
    metadata = metadata.loc[metadata[prev_meta_name] == prev_meta_value]

    for meta in metadata_names[1:]:
        meta_value = st.sidebar.selectbox(f"Select {meta}", metadata.loc[:, meta].unique())
        metadata = metadata.loc[metadata[meta] == meta_value]
    return metadata.index.values[0]
  
def tile_dataset(tile_size, hop, data, freq, time):
    size = sxx.shape[0]
    run = list()
    for i in range(hop, size-tile_size-(hop+1), tile_size):
        print(sxx[(i-hop):(i+tile_size+hop+1), (sxx.shape[1]//2-hop):, :].shape)
        run.append(sxx[(i-hop):(i+tile_size+hop+1), (sxx.shape[1]//2-hop):, :].astype(np.float32))
    
    return np.stack(run, axis=0), freq[(freq.shape[0]//2):-(hop-1)], time[hop:(i+tile_size)]

@st.cache(allow_output_mutation=True)
def load_model():
    reconstructed_model = tf.keras.models.load_model("unet_lofar_run_1_back", custom_objects=custom_objects)
    return reconstructed_model

def predict_entire_lofargram(tiled_sxx, reconstructed_model):
    for sample in list(dset_4classes):
        run_pred = list()
        original_image = list()
        for sample in list(dset_4classes):
            prediction = reconstructed_model.predict(sample[None, ...])
            prediction = prediction[0, :, :, 0]
            out_shape = prediction.shape
            
#             print(sample.shape)
#             print(prediction.shape)
    #         prediction[prediction[:, :] > p] = 1
    #         prediction[prediction[:, :] <= p] = 0
            original_image.append(crop_to_shape(sample.numpy().squeeze(), out_shape))
            run_pred.append(prediction[:, :])

    original_image = np.concatenate(original_image, axis=0)
    run_pred = np.concatenate(run_pred, axis=0)
    print(original_image.shape)
    print(run_pred.shape)
    return original_image, run_pred

h = 143 - 43

st.title('Spectrogram Viz')

# use_tpsw = st.sidebar.checkbox("Use TPSW", value=True)
# thres    = st.sidebar.checkbox("Use Threshold", value=True)

# thres = -4 if thres else None

dataset_name = st.sidebar.selectbox('Select dataset for visualization', ["4classes", "shipsear", "os-training"])
dataset_info, info_names = load_info(dataset_name)
options = list(dataset_info.keys())
name = info_names[0]
key = st.sidebar.selectbox(f"Select {name}", options)

file_id = key
for name in info_names[1:]:
    dataset_info = dataset_info[key]
    options = list(dataset_info.keys())
    key = st.sidebar.selectbox(f"Select {name}", options)
    file_id = file_id + "_" + key
filepath = dataset_info[key]


cachedir = Path("cache")
cachefile = cachedir / f"{file_id}.npz"
if not cachefile.exists():
    print(cachefile)
    sxx, freq, time = load_file(filepath, True, -4)
    tiled_sxx, freq, time = tile_dataset(h, 21, sxx, freq, time)
    print(tiled_sxx.shape)

    reconstructed_model = load_model()

    dset_4classes = tf.data.Dataset.from_tensor_slices(tiled_sxx)

    print(freq.shape)
    print(time.shape)
    original_image, run_pred = predict_entire_lofargram(dset_4classes, reconstructed_model)
    np.savez_compressed(cachefile, original=original_image, prediction=run_pred, freq=freq, time=time)
else:
    data = np.load(cachefile)
    original_image, run_pred, freq, time = data['original'], data['prediction'], data['freq'], data['time']

plots = ["Side-by-side", "Overlaped"]
plot_type = st.sidebar.selectbox("Select plot type", plots)
p = st.sidebar.slider('Decision Threshold', .50, .99, .70)

run_pred[run_pred > p] = 1
run_pred[run_pred <= p] = 0
   
if plot_type == plots[0]:
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,8))

    _, cbar = waterfall_spectrogram(ax1, freq, time[::], original_image[::], title='', cmap='jet', cbar_unit='dB', show_rpm=False)
    cbar.remove()
    _, cbar = waterfall_spectrogram(ax2, freq, time[::], run_pred[::], title='', cmap='inferno', cbar_unit='dB', show_rpm=False)
    cbar.remove()
    # ax1.imshow(original_image, cmap="jet")
    # ax2.imshow(run_pred, cmap="inferno")
elif plot_type == plots[1]:
    passthrough = st.sidebar.slider("Passthrough", .0, 1., .0)
    fig, ax = plt.subplots(ncols=1, figsize=(8,4))
    ax.imshow(original_image[::-1], cmap="Greys", extent=[freq[0], freq[-1], time[0], time[-1]], aspect="auto")
    ax.imshow(run_pred[::-1], alpha=passthrough, cmap="jet", extent=[freq[0], freq[-1], time[0], time[-1]], aspect="auto")

else:
    raise ValueError("Invalid plot type")
st.pyplot(fig)