## Step 1: Create Conda Environment

```bash
conda create -n mambavo python=3.10 -y
conda activate mambavo
```

## Step 2: Install PyTorch

```bash
pip install torch==2.5.0+cu118 torchvision==0.20.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

## Step 3: Install Python Dependencies

```bash
pip install tensorboard numba tqdm einops pypose kornia \
    numpy==1.26.4 plyfile evo opencv-python yacs
```

## Step 4: Install DPVO (BA layer and CUDA)

```bash
git clone https://github.com/princeton-vl/DPVO.git --recursive
cd DPVO

# Download Eigen 3.4.0 (required for BA layer)
mkdir -p thirdparty
cd thirdparty
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar -xzf eigen-3.4.0.tar.gz
rm eigen-3.4.0.tar.gz
cd ..

# then compile and install DPVO (BA layer which is needed for the training to work)
export TMPDIR=~/cuda_tmp
mkdir -p ~/cuda_tmp
export MAX_JOBS=4
pip install --no-build-isolation ./
```

## Notes

- Load the CUDA 11.8 module before starting:
  `module load cuda/11.8`

