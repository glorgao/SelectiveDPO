## Introduction to SelectiveDPO

This repository contains the code and released models for our paper [Principled Data Selection for Alignment: The Hidden Risks of Difficult Examples](https://arxiv.org/pdf/2502.09650). The proposed algorithm, SelectiveDPO, selectively trains on preference examples within the model's capacity. It improves alignment performance by 9-16% in win rates on the AlpacaEval 2 benchmark compared to the DPO baseline, suppressing a series of DPO variants with different algorithmic adjustments. For the released model trained with SelectiveDPO, please visit this [collection page](https://huggingface.co/collections/glorgao/selectivedpo-676966c5bf01f8eb91a8fb85).

![](selective-dpo-illustration.jpg)

## Reproduce SelectiveDPO
To reproduce the benchmarking results from our paper, please follow the steps below.
#### Step 0: Preparing the environment
- Hardware. The released code is designed to run on a node with 8 H100 GPUs. However, it is possible to reproduce the results with 4 GPUs (80GB memory each). If using fewer GPUs, make sure to adjust `per_device_train_batch_size` and `gradient_accumulation_steps` accordingly.

- Software. Our codebase is built on the [huggingface/alignment-handbook](https://github.com/huggingface/alignment-handbook). Here are the steps to install the environment.

```bash
conda create -n sdpo python=3.10
conda activate sdpo

# install the pytorch v2.2.2 at https://pytorch.org/get-started/locally/. For example, for linux
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# install the specified version of transformers, peft, trl, and deepspeed
pip install transformers==4.42.4 peft==0.13.2 trl==0.11.4 deepspeed==0.14.4 

# Install alignment-handbook
git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
python -m pip install .
 
# install flash-attn for fast inference 
pip install flash-attn --no-build-isolation

# Other dependencies
pip install wandb
```

#### Step 1: Score the example difficulty by validation loss (Optional)
```bash
cd curriculum-lens 
bash curriculum-lens-qwen-2.5.sh
python make_curricula.py 
```
You may need to modify the Bash and Python scripts to fit your own computational environment.


- The Bash script runs the DPOTrainer using a standard *RandomSampler* on the first half of the training data. It then evaluates the hold-out examples from the other half to compute `validation loss`. The training and evaluation partitions are reversed, and this process is repeated three times with different random seeds and data-splitting strategies, yielding six reference models (note: in this context, a reference model is used to compute validation loss, which differs from the reference model in the DPO objective).

- The Python script averages the `validation loss` statistics and outputs a CSV file recording the aggregated measures. To verify the correctness of the resulting difficulty measures, please refer to our paper.



#### Step 2: Run SelectiveDPOTrainer.
```bash
cd selective-dpo 
bash run-selectivedpo-uf.sh
```
You may need to modify the Bash script to fit your computational environment.

Our training scripts are straightforward:

- `script.run_selective_dpo.py` applies the chat template, selects the easiest 50% of training examples (a tunable hyperparameter) from the ultrafeedback_binarized dataset, and runs SelectiveDPOTrainer.

- `script.selective_dpo_trainer.py` implements SelectiveDPOTrainer, where the only meaningful modification is replacing *RandomSampler* with *SequentialSampler* to query examples from easy to difficut.

We have included four precomputed `validation loss` records for four models (qwen2.5-7b-sft, mistral-7b-sft, llama3-8b-sft, and gemma2-9b-sft), in the `/curricula` folder. If you want to use your own curriculum, please rename your CSV file accordingly.
