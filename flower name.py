
import csv
import re
from rapidfuzz import fuzz, process

# Read the flower lines
with open("Book1.csv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    flower_lines = [row[0] for row in reader if row]  # assuming 1 column

# Read Latin names
with open("latin_names_cleaned.csv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    latin_names = [row[0].strip() for row in reader if row]

processed_lines = []

threshold = 80 #threshold percentage for approximate match

# --- Step 2: Pre-clean Latin names ---
latin_names_cleaned = [
    re.sub(r'[^a-zA-Z\s]', '', name).lower()
    for name in latin_names
]

# --- Step 3: Match each flower line ---
processed_lines = []
threshold = 85  # minimum similarity percentage

for line in flower_lines:
    line_clean = re.sub(r'[^a-zA-Z\s]', '', line).lower()  # remove punctuation, lowercase. ^ means negate, a-zA-Z means all letters for both cases, \s is whitespace, ' ' means it is replaced with a space

    # Find best match among all Latin names
    result = process.extractOne(query=line_clean, choices=latin_names_cleaned, scorer=fuzz.partial_ratio)
    #process.extract0ne finds best match from given list and fuzz.partial_ratio is used to calculate similarity score by comparing the shorter string against all possible substrings of the longer string and returning the highest match percentage.

    if result is not None:
        best_match = result[0]  # the matched string
        best_score = result[1]  # the similarity score
        if best_score >= threshold:
            index = latin_names_cleaned.index(best_match)
            processed_lines.append(latin_names[index])
        else:
            processed_lines.append('n/a')
    else:
        processed_lines.append('n/a')


# Save results
with open("processed_flower_names.csv", 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows([[line] for line in processed_lines])



