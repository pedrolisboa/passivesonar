mkdir -p generated_samples/train
mkdir -p generated_samples/validation

mkdir cache

git clone https://github.com/jakeret/unet.git
cd unet
pip install .
cd ..