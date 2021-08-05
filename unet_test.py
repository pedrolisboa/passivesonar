import tensorflow as tf
from pathlib import Path
import numpy as np
from unet.callbacks import TensorBoardImageSummary
import unet

gen_samples_path = Path("generated_samples")

data = np.load(gen_samples_path / "train" /  '000.npz')
h, w, c = data["arr_0"].shape

def load_images(gen_samples_path, mode, class_weights):
    def gen_img():
        for file in list((gen_samples_path / mode).iterdir())[:500]:
            data = np.load(file)
            
            sample_weights = np.take(np.array(class_weights), data["arr_1"].astype(np.int32)[:, :, 0])
            
            yield data["arr_0"].astype(np.float32), data["arr_1"].astype(np.float32), sample_weights[:, :, None]
    return gen_img

class_weights = np.load(gen_samples_path / "class_weights.npy")
train_dataset = tf.data.Dataset.from_generator(load_images(gen_samples_path, "train", class_weights), 
                                               output_signature= (tf.TensorSpec((h, w, c), dtype=tf.float32), 
                                                                  tf.TensorSpec((h, w, 2), dtype=tf.float32),
                                                                  tf.TensorSpec((h, w, 1), dtype=tf.float32)))
validation_dataset = tf.data.Dataset.from_generator(load_images(gen_samples_path, "validation", class_weights), 
                                               output_signature= (tf.TensorSpec((h, w, c), dtype=tf.float32), 
                                                                  tf.TensorSpec((h, w, 2), dtype=tf.float32),
                                                                  tf.TensorSpec((h, w, 1), dtype=tf.float32)))


log_dir_path = "unet_lofar_run_1"
tb_img = TensorBoardImageSummary("validation", logdir=log_dir_path, dataset=validation_dataset, max_outputs=10)

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
