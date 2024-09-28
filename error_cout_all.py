import os
import pandas as pd
from tqdm import tqdm
import pickle

# Load the dataframe
df = pd.read_csv('/mnt/data1/liupeizhuo/detect/final_new/120_cosine_va_clean/retrieval_res_cos.csv')

# Initialize error and all categories dictionaries
cate_err = {
    'financial': [],
    'legal': [],
    'educational': [],
    'political': [],
    'medical': []
}

cate_all = {
    'financial': [],
    'legal': [],
    'educational': [],
    'political': [],
    'medical': []
}

# Initialize nested dictionaries for category-dataset error and total counts
cate_dataset_err = {category: {} for category in cate_err.keys()}
cate_dataset_all = {category: {} for category in cate_err.keys()}

# Initialize a list to store speakerIDs of incorrect samples
error_speakers = []

# Count the total number of non-split_datasets entries
all_am = 0
for i in tqdm(range(0, len(df), 1)):
    info = df.iloc[i]['test_speaker'].split('/')
    dataset_name = info[4]  # Extract dataset name from the fourth position in the path
    category_name = info[5]  # Extract category name

    # Initialize dataset error and all counts if not already done for the category
    if dataset_name not in cate_dataset_err[category_name]:
        cate_dataset_err[category_name][dataset_name] = []
        cate_dataset_all[category_name][dataset_name] = []

    t = 0
    names = []
    while t < 1:
        names.append(str(df.iloc[i+t]['similar_speaker']))
        t += 1
    all_am += 1

    if info[6].split('-')[0] not in names:
        cate_err[category_name] += ['/'.join(info)]
        cate_dataset_err[category_name][dataset_name] += ['/'.join(info)]
        error_speakers.append(info[6].split('-')[0])  # Record the speakerID of the incorrect sample

    cate_all[category_name] += [info[6] + '-' + info[7]]
    cate_dataset_all[category_name][dataset_name] += [info[6] + '-' + info[7]]

# Calculate and print error rates
count = 0
results = []

for category in cate_err.keys():
    category_error_count = len(cate_err[category])
    category_all_count = len(cate_all[category])
    category_error_rate = category_error_count / all_am
    specific_category_error_rate = category_error_count / category_all_count
    results.append(f"Category {category} Error Count: {category_error_count}\n")
    results.append(f"All Err {category}: {category_error_rate:.4f}\n")
    results.append(f"Cate Err {category}: {specific_category_error_rate:.4f}\n")
    count += category_error_rate

    # Calculate error rates for each dataset within the category
    for dataset in cate_dataset_err[category].keys():
        dataset_error_count = len(cate_dataset_err[category][dataset])
        dataset_all_count = len(cate_dataset_all[category][dataset])
        dataset_error_rate = dataset_error_count / dataset_all_count if dataset_all_count > 0 else 0
        results.append(f"Category {category}, Dataset {dataset} Error Count: {dataset_error_count}\n")
        results.append(f"Category {category}, Dataset {dataset} Error Rate: {dataset_error_rate:.4f}\n")

results.append(f"Final Err: {count:.4f}\n")

# Print the results
for result in results:
    print(result)

# Save the results to a .txt file
with open('./error_report_all_d_120.txt', 'w') as fw:
    fw.writelines(results)
    print("Error report saved to error_report.txt")

# Save the list of speakerIDs with errors to a .txt file
with open('./error_speaker_ids_d_120.txt', 'w') as fw:
    for speaker_id in error_speakers:
        fw.write(speaker_id + '\n')
    print("Speaker IDs with errors saved to error_speaker_ids.txt")
