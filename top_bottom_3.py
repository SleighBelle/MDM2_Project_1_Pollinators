import pandas as pd

# Read the flower lines

df = pd.read_csv('(42K flowers) Beewalk data 2008-23 31012024.csv')

df_flowers = df['flower_visited'].to_list()

# identify top 3 flower names by mode
from collections import Counter
flower_counts = Counter(df_flowers)
top_3_flowers = flower_counts.most_common(3)
print("Top 3 flower names by frequency:")
for flower, count_flow in top_3_flowers:
    print(f"-{flower}: {count_flow}")
    
print('==============================')

# identify bottom 3 flower names by mode (with, minimum, over 20 occurrences)
bottom_3_flowers_initial = flower_counts.most_common()
bottom_3_flowers_initial_rev = bottom_3_flowers_initial[::-1]

bottom_3_flowers_processed = []
processed_count_flow = 0
for flower_name_bottom, flower_count_bottom in bottom_3_flowers_initial_rev:
    if flower_count_bottom>20:
        bottom_3_flowers_processed.append((flower_name_bottom, flower_count_bottom))
        processed_count_flow +=1
        if processed_count_flow == 3:
            break

print("Bottom 3 flower names by frequency:")
for flower, count_flow in bottom_3_flowers_processed:
    print(f"-{flower}: {count_flow}")
print('==============================')

################

df_pollinators = df['latin'].to_list()


# identify top 3 flower names by mode
from collections import Counter
pollinator_counts = Counter(df_pollinators)
top_3_pollinator = pollinator_counts.most_common(3)
print("Top 3 pollinator names by frequency:")
for pollinator, count_poll in top_3_pollinator:
    print(f"-{pollinator}: {count_poll}")
print('==============================')

# identify bottom 3 flower names by mode (with at least 20 occurrences)
bottom_3_pollinator_initial = pollinator_counts.most_common()
bottom_3_pollinator_initial_rev = bottom_3_pollinator_initial[::-1]

bottom_3_pollinator_processed = []
processed_count_poll = 0
for pollinator_name_bottom, pollinator_count_bottom in bottom_3_pollinator_initial_rev:
    if pollinator_count_bottom>20:
        bottom_3_pollinator_processed.append((pollinator_name_bottom, pollinator_count_bottom))
        processed_count_poll +=1
        if processed_count_poll == 3:
            break

print("Bottom 3 pollinator names by frequency:")
for pollinator, count_poll in bottom_3_pollinator_processed:
    print(f"-{pollinator}: {count_poll}")
print('==============================')
