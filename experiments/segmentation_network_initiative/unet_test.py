import tensorflow as tf
from pathlib import Path
import numpy as np


data = np.load(Path('generated_samples') / "train" /  '00000.npz')
h, w, c = data["arr_0"].shape

gen_samples_path = Path("generated_samples")
def load_images(gen_samples_path, mode):
    def gen_img():
        for file in list((gen_samples_path / mode).iterdir())[:1000]:
            data = np.load(file)
            
            yield data["arr_0"].astype(np.float32), data["arr_1"].astype(np.float32)
    return gen_img
#train_dataset = tf.data.Dataset.list_files(str(gen_samples_path / "train" / "*.npz"), shuffle=True)

train_dataset = tf.data.Dataset.from_generator(load_images(gen_samples_path, "train"), 
                                               output_signature= (tf.TensorSpec((h, w, c), dtype=tf.float32), 
                                                                  tf.TensorSpec((h, w, 2), dtype=tf.float32)))
validation_dataset = tf.data.Dataset.from_generator(load_images(gen_samples_path, "validation"), 
                                               output_signature= (tf.TensorSpec((h, w, c), dtype=tf.float32), 
                                                                  tf.TensorSpec((h, w, 2), dtype=tf.float32)))

from unet.callbacks import TensorBoardImageSummary
log_dir_path = "unet_lofar_run_1"
tb_img = TensorBoardImageSummary("validation", logdir=log_dir_path, dataset=validation_dataset, max_outputs=10)

import unet
from unet.datasets import circles

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
           batch_size=64)
