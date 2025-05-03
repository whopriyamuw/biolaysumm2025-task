import csv
import textstat

# input csv with summaries in second column
input_file = 'validation.csv'

def compute_readability(text):
    return {
        "Flesch-Kincaid Grade": textstat.flesch_kincaid_grade(text),
        "Dale-Chall Score": textstat.dale_chall_readability_score(text),
        "Coleman-Liau Index": textstat.coleman_liau_index(text),
    }

# initialize score totals
fk_total = dc_total = cl_total = 0
count = 0

# score each row
with open(input_file, newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    headers = next(reader)

    for row in reader:
        summary = row[1]
        scores = compute_readability(summary)

        fk_total += scores["Flesch-Kincaid Grade"]
        dc_total += scores["Dale-Chall Score"]
        cl_total += scores["Coleman-Liau Index"]
        count += 1

# print averages
if count > 0:
    print("\nAverage readability scores:")
    print(f"Flesch-Kincaid Grade: {round(fk_total / count, 2)}")
    print(f"Dale-Chall Score: {round(dc_total / count, 2)}")
    print(f"Coleman-Liau Index: {round(cl_total / count, 2)}")
else:
    print("none found")
