import pandas as pd 
# This script is a demo, you are need to modify the FOLDER, SEED_LIST, STEP_LIST, and the path of the csv files.


FOLDER = 'qwen-2.5-7b-argilla'
SEED_LIST = ['seed41', 'seed43', 'seed42']
STEP_LIST = ['step00', 'step01', 'step02', 'step03', 'step04']

merged_df = None 
for seed in SEED_LIST: 
    for step in STEP_LIST:
        df = pd.read_csv('learning_record' + '/' + FOLDER + '/' + f'{seed}_{step}.csv', header=None, index_col=False)
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
merged_df.to_csv('qwen-2.5-7b-argilla_learning_order.csv', index=False)