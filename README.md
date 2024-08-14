## Getting Started

### Environment Setup

To set up the environment for training, please run
```bash
conda create -n fastclip python=3.11.9
conda activate fastclip
pip install -r requirements-training.txt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

### Training

Sample script to run the program on 16 PVC GPUs (4 nodes and 4 GPUs per node). Please
- change the path of the conda exectuable
- change the path of the intel library
- change the data path (specified by --train-data)

```bash
#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --mem=120G
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=fastclipv3
#SBATCH --partition=pvc
#SBATCH --output=%x_%j.log
#SBATCH --exclude=ac[024,026,030,068]

##### load intel lib
oneapi_2024_0='/sw/hprc/sw/oneAPI/2024.0'
oneapi_2024_1='/sw/hprc/sw/oneAPI/2024.1'
source "${oneapi_2024_1}"/compiler/latest/env/vars.sh
source "${oneapi_2024_0}"/mkl/latest/env/vars.sh
source "${oneapi_2024_1}"/ccl/latest/env/vars.sh
source "${oneapi_2024_1}"/mpi/latest/env/vars.sh

##### activate conda env
__conda_setup="$('/sw/eb/sw/Miniconda3/23.5.2-0/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sw/eb/sw/Miniconda3/23.5.2-0/etc/profile.d/conda.sh" ]; then
        . "/sw/eb/sw/Miniconda3/23.5.2-0/etc/profile.d/conda.sh"
    else
        export PATH="/sw/eb/sw/Miniconda3/23.5.2-0/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate fastclip

##### distributed training
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12805

##### program-specfic environment variable
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'

##### encountered the following warning
# |CCL_WARN| topology recognition shows PCIe connection between devices. If this is not correct, you can disable topology recognition, with CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0. This will assume
# XeLinks across devices
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0

world_size=16
mpirun -n "${world_size}" -l python -u src/training/main.py \
    --save-frequency 1 \
    --train-data './datasets/cc3m_webdataset/cc3m_train/{00000..00331}.tar' \
    --train-num-samples 2723840 --data_size 3318333 \
    --warmup 10000 \
    --batch-size 128 \
    --epochs 37 \
    --workers 6 \
    --dist-backend ccl \
    --model ViT-B-32 \
    --name medium_fastclipv3 \
    --seed 2024 \
    --profile \
    --wd 0.1 \
    --local-loss \
    --fastclip --multiply_tau --temperature_scheme global_learnable \
    --lr 1e-3 --lr_tau 2e-4 --lr_tau_scheduler step_thresh --rho 6.5 \
    --gamma 0.2 --gamma_schedule cosine --gamma_decay_epochs 18
```

**Non-Slurm**: For non-slurm single-node training, we need to change `MASTER_ADDR` and explicitly set `WORLD_SIZE`, the following is a sample script to run the program on 2 PVC GPUs.
```bash
#!/bin/bash

##### load intel lib
oneapi_2024_0='/sw/hprc/sw/oneAPI/2024.0'
oneapi_2024_1='/sw/hprc/sw/oneAPI/2024.1'
source "${oneapi_2024_1}"/compiler/latest/env/vars.sh
source "${oneapi_2024_0}"/mkl/latest/env/vars.sh
source "${oneapi_2024_1}"/ccl/latest/env/vars.sh
source "${oneapi_2024_1}"/mpi/latest/env/vars.sh

##### activate conda env
__conda_setup="$('/sw/eb/sw/Miniconda3/23.5.2-0/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sw/eb/sw/Miniconda3/23.5.2-0/etc/profile.d/conda.sh" ]; then
        . "/sw/eb/sw/Miniconda3/23.5.2-0/etc/profile.d/conda.sh"
    else
        export PATH="/sw/eb/sw/Miniconda3/23.5.2-0/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate fastclip

##### distributed training
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT=12805
world_size=2
export WORLD_SIZE="${world_size}"

##### program-specfic environment variable
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'

mpirun -np "${world_size}" -l python -u src/training/main.py \
    --save-frequency 1 \
    --train-data './datasets/cc3m_webdataset/cc3m_train/{00000..00331}.tar' \
    --train-num-samples 2723840 --data_size 3318333 \
    --warmup 10000 \
    --batch-size 128 \
    --epochs 37 \
    --workers 6 \
    --dist-backend ccl \
    --model ViT-B-32 \
    --name medium_fastclipv3 \
    --seed 2024 \
    --profile \
    --wd 0.1 \
    --local-loss \
    --fastclip --multiply_tau --temperature_scheme global_learnable \
    --lr 1e-3 --lr_tau 2e-4 --lr_tau_scheduler step_thresh --rho 6.5 \
    --gamma 0.2 --gamma_schedule cosine --gamma_decay_epochs 18
```
