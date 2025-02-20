import pandas as pd 
import argparse
import os

SEED_LIST = ['seed41', 'seed43', 'seed42']
STEP_LIST = ['step0', 'step1', 'step2', 'step3', 'step4']

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process a dataset.')
    parser.add_argument('--input_dir', type=str, default='./output/qwen-2.5-7b-uf')

    # Parse arguments
    args = parser.parse_args()
    output_file = os.path.join(args.input_dir, 'learning_order.csv')
    
    
    # Use the arguments
    print(f"Processing dataset: {args.input_dir}")
    print(f"Output will be saved to: {output_file}")
    
    merged_df = None 
    for seed in SEED_LIST: 
        for step in STEP_LIST:
            df1 = pd.read_csv(os.path.join(args.input_dir, f'learning_record-{seed}-{step}-first.csv'), header=None, index_col=False)
            df2 = pd.read_csv(os.path.join(args.input_dir, f'learning_record-{seed}-{step}-second.csv'), header=None, index_col=False)
            df = pd.concat([df1, df2], ignore_index=True)
            print(df.shape)
            # name the column, 'prompt_id', 'loss_{step}'
            df.columns = ['prompt_id', f'loss_{step}_{seed}']

            df['prompt_id'] = df['prompt_id'].astype(str)
            df = df.reset_index(drop=True)

            # drop duplicates prompt_id, keeping the first 
            df = df.drop_duplicates(subset='prompt_id', keep='first')
            print(f"Before merge: {seed}_{step}, df shape: {df.shape}, prompt_id unique: {df['prompt_id'].nunique()}")

            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on='prompt_id', how='left')

    # create a new column, 'loss_mean', calculate the mean of all loss_{step}_{seed}
    merged_df['learning_order'] = merged_df.filter(like='loss').mean(axis=1)

    # check the spearman correlation between loss_{step}_{seed}
    correlation = merged_df.iloc[:, 1:].corr(method='spearman')
    print(correlation)

    print(merged_df.columns, merged_df.shape)
    # save the merged_df to csv file
    merged_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()