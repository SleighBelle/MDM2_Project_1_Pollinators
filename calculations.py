import numpy as np
import pandas as pd

input_path = 'MDM2/csv_files/42k_flowers.csv'

df = pd.read_csv(input_path)

relevant = ['flower_visited', 'latin', 'TotalCount']

relevent_df = df[relevant]

aggregate = {
    'TotalCount': 'sum'
}

df_agg = relevent_df.groupby(['flower_visited']).agg(aggregate).reset_index()

sorted_df = df_agg.sort_values(by = 'TotalCount', ascending = False)

# top_rows = sorted_df.head(50)
# top_plants = top_rows[['flower_visited', 'TotalCount']]
# print(top_plants)

total_inter = sorted_df['TotalCount'].sum()
print(f'\n The total number of interactions are: {total_inter}')

threshold = total_inter / 2

num = 0
idx = 0

while num < threshold:
    num += sorted_df['TotalCount'].iloc[idx]
    idx += 1

print(f'\n The itteration stopped at index {idx}')

print(f'\n the number of interactions is now {num}')

# Now lets find out what pct of the total and top half on interactions the top 3, 5 and 10 plants account for
top_3_rows = sorted_df.head(3)
top_5_rows = sorted_df.head(5)
top_10_rows = sorted_df.head(10)

top_3_num = top_3_rows['TotalCount'].sum()
top_5_num = top_5_rows['TotalCount'].sum()
top_10_num = top_10_rows['TotalCount'].sum()

top_3_pct = (top_3_num / total_inter) * 100
top_5_pct = (top_5_num / total_inter) * 100
top_10_pct = (top_10_num / total_inter) * 100

msg_1 = f'The total number of interactions of the top 3 plants are {top_3_num}, which has a pct value of {top_3_pct}'
msg_2 = f'The total number of interactions of the top 5 plants are {top_5_num}, which has a pct value of {top_5_pct}'
msg_3 = f'The total number of interactions of the top 10 plants are {top_10_num}, which has a pct value of {top_10_pct}'

print('\n', msg_1)
print('\n', msg_2)
print('\n', msg_3)