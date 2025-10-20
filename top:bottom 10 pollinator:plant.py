import csv

# Read the flower lines
with open("flower_list.csv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    flower_lines = [row[0] for row in reader if row]  # assuming 1 column


# identify top 10 flower names by mode
from collections import Counter
flower_counts = Counter(flower_lines)
top_10_flowers = flower_counts.most_common(10)
print("Top 10 flower names by frequency:")
for flower, count_flow in top_10_flowers:
    print(f"-{flower}: {count_flow}")

# identify bottom 10 flower names by mode (with at least 20 occurrences)
bottom_10_flowers_initial = flower_counts.most_common()
bottom_10_flowers_initial_rev = bottom_10_flowers_initial[::-1]

bottom_10_flowers_processed = []
processed_count_flow = 0
for flower_name_bottom, flower_count_bottom in bottom_10_flowers_initial_rev:
    if flower_count_bottom>20:
        bottom_10_flowers_processed.append((flower_name_bottom, flower_count_bottom))
        processed_count_flow +=1
        if processed_count_flow == 10:
            break

print("Bottom 10 flower names by frequency:")
for flower, count_flow in bottom_10_flowers_processed:
    print(f"-{flower}: {count_flow}")

################

# Read the flower lines
with open("pollinator_list.csv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    pollinator_lines = [row[0] for row in reader if row]  # assuming 1 column


# identify top 10 flower names by mode
from collections import Counter
pollinator_counts = Counter(pollinator_lines)
top_10_pollinator = pollinator_counts.most_common(10)
print("Top 10 pollinator names by frequency:")
for pollinator, count_poll in top_10_pollinator:
    print(f"-{pollinator}: {count_poll}")

# identify bottom 10 flower names by mode (with at least 20 occurrences)
bottom_10_pollinator_initial = pollinator_counts.most_common()
bottom_10_pollinator_initial_rev = bottom_10_pollinator_initial[::-1]

bottom_10_pollinator_processed = []
processed_count_poll = 0
for pollinator_name_bottom, pollinator_count_bottom in bottom_10_pollinator_initial_rev:
    if pollinator_count_bottom>20:
        bottom_10_pollinator_processed.append((pollinator_name_bottom, pollinator_count_bottom))
        processed_count_poll +=1
        if processed_count_poll == 10:
            break

print("Bottom 10 pollinator names by frequency:")
for pollinator, count_poll in bottom_10_pollinator_processed:
    print(f"-{pollinator}: {count_poll}")