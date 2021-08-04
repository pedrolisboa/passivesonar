import tensorflow as tf
from pathlib import Path
import numpy as np
from load_utils import load_4classes, load_shipsear

# def tile_dataset(tile_size, hop, data):
#     tiled_sxx = list()
#     for sxx in data:
#         size = sxx.shape[0]
#         for i in range(0, size-tile_size, hop):
#             tiled_sxx.append(sxx[i:(i+tile_size), (512-hop):, :].astype(np.float32))
#     return np.stack(tiled_sxx, axis=0)
# tiled_data = tile_dataset(h, 21, data)

# data = load_shipsear()
# dset_shipsear = tf.data.Dataset.from_tensor_slices(tile_dataset(h, 21, data)).shuffle(1000)

# # data = load_4classes()
# # dset_4classes = tf.data.Dataset.from_tensor_slices(tile_dataset(h, 21, data)).shuffle(1000)


data = np.load(Path('generated_samples') / "train" /  '000.npz')
h, w, c = data["arr_0"].shape

gen_samples_path = Path("generated_samples")
def load_images(gen_samples_path, mode, class_weights):
    def gen_img():
        for file in list((gen_samples_path / mode).iterdir())[:500]:
            data = np.load(file)
            
            sample_weights = np.take(np.array(class_weights), data["arr_1"].astype(np.int32)[:, :, 0])
            
            yield data["arr_0"].astype(np.float32), data["arr_1"].astype(np.float32), sample_weights[:, :, None]
    return gen_img
#train_dataset = tf.data.Dataset.list_files(str(gen_samples_path / "train" / "*.npz"), shuffle=True)

class_weights = [1, 100]
train_dataset = tf.data.Dataset.from_generator(load_images(gen_samples_path, "train", class_weights), 
                                               output_signature= (tf.TensorSpec((h, w, c), dtype=tf.float32), 
                                                                  tf.TensorSpec((h, w, 2), dtype=tf.float32),
                                                                  tf.TensorSpec((h, w, 1), dtype=tf.float32)))
validation_dataset = tf.data.Dataset.from_generator(load_images(gen_samples_path, "validation", class_weights), 
                                               output_signature= (tf.TensorSpec((h, w, c), dtype=tf.float32), 
                                                                  tf.TensorSpec((h, w, 2), dtype=tf.float32),
                                                                  tf.TensorSpec((h, w, 1), dtype=tf.float32)))

from unet.callbacks import TensorBoardImageSummary
log_dir_path = "unet_lofar_run_1"
tb_img = TensorBoardImageSummary("validation", logdir=log_dir_path, dataset=validation_dataset, max_outputs=10)


import unet

#building the model
unet_model = unet.build_model(channels=c,
                              num_classes=2,
                              layer_depth=3,
                              filters_root=16)

unet.finalize_model(unet_model)

#training and validating the model
trainer = unet.Trainer(name="unet_gen_test", log_dir_path ="unet_lofar_run_1", checkpoint_callback =True, tensorboard_callback=True, tensorboard_images_callback=tb_img)
trainer.fit(unet_model,
           train_dataset,
           validation_dataset,
           epochs=10,
           batch_size=2)
