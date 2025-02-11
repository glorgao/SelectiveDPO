## Introduction to SelectiveDPO

<ins>*Preference data vary in difficulty, and overly difficult examples hinder alignment, by exceeding the model's capacity.*</ins>

This is the key insight we want to convey. Based on this principle, we propose SelectiveDPO, a DPO-like trainer that selectively trains on preference examples within the model's capacity. It improves alignment performance by 9-16% in win rates on the AlpacaEval 2 benchmark compared to the DPO baseline, suppressing a series of DPO variants with different algorithmic adjustments.

The significant performance gain suggests that difficult examples, characterized by high validation loss, not only contribute little to alignment but may actually be detrimental. For a more detailed investigation and evaluation, please refer to our paper:


*Principled Data Selection for Alignment: The Hidden Risks of Difficult Examples*, 


## Reproduce SelectiveDPO
To reproduce the benchmarking results from our paper, please follow the steps below.


#### Step 1: Scoring the example difficulty by validation loss (Optional)
```bash
cd curriculum-lens 
bash curriculum-lens-qwen-2.5.sh
python make-curricula-uf.py 
```
You may need to modify the Bash and Python scripts to fit your own computational environment.


- The Bash script runs the DPOTrainer using a standard *RandomSampler* on the first half of the training data. It then evaluates the hold-out examples from the other half to compute `validation loss`. The training and evaluation partitions are reversed, and this process is repeated three times with different random seeds and data-splitting strategies, yielding six reference models (note: in this context, a reference model is used to compute validation loss, which differs from the reference model in the DPO objective).

- The Python script averages the `validation loss` statistics and outputs a CSV file recording the aggregated measures. To verify the correctness of the resulting difficulty measures, please refer to our paper.



#### Step 2: Rerun DPOTrainer on the selected partition of easy examples.
```bash
cd selective-dpo 
bash run-selectivedpo-uf.sh
```
You may need to modify the Bash script to fit your computational environment.


Our training scripts are straightforward:

- `script.run_selective_dpo.py` applies the chat template, selects the easiest 50% of training examples (a tunable hyperparameter) from the ultrafeedback_binarized dataset, and runs SelectiveDPOTrainer.

- `script.selective_dpo_trainer.py` implements SelectiveDPOTrainer, where the only meaningful modification is replacing *RandomSampler* with *SequentialSampler* to fully utilize the computed difficulty measure.

We have included four precomputed `validation loss` records in the /curricula folder. If you want to use your own curriculum, please rename your CSV file accordingly.
