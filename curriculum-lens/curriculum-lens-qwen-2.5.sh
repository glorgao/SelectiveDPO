# If you are slurm user, you can use this script to run the experiments in the Qwen-2.5-UF curriculum.
srun --partition=[your-partition] --time=24:00:00 --nodes=1 --ntasks-per-node=1 --gres=gpu:8 --exclusive --pty accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero2.yaml scripts/run_first_half.py  recipes/qwen-2.5-7b-uf-seed41.yaml 
srun --partition=[your-partition] --time=24:00:00 --nodes=1 --ntasks-per-node=1 --gres=gpu:8 --exclusive --pty accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero2.yaml scripts/run_first_half.py  recipes/qwen-2.5-7b-uf-seed42.yaml 

srun --partition=[your-partition] --time=24:00:00 --nodes=1 --ntasks-per-node=1 --gres=gpu:8 --exclusive --pty accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero2.yaml scripts/run_first_half.py  recipes/qwen-2.5-7b-uf-seed43.yaml 
srun --partition=[your-partition] --time=24:00:00 --nodes=1 --ntasks-per-node=1 --gres=gpu:8 --exclusive --pty accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero2.yaml scripts/run_second_half.py  recipes/qwen-2.5-7b-uf-seed41.yaml 

srun --partition=[your-partition] --time=24:00:00 --nodes=1 --ntasks-per-node=1 --gres=gpu:8 --exclusive --pty accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero2.yaml scripts/run_second_half.py  recipes/qwen-2.5-7b-uf-seed42.yaml 
srun --partition=[your-partition] --time=24:00:00 --nodes=1 --ntasks-per-node=1 --gres=gpu:8 --exclusive --pty accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero2.yaml scripts/run_second_half.py  recipes/qwen-2.5-7b-uf-seed43.yaml 

