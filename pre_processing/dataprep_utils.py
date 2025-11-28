
import json
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import jensenshannon

def get_files(path, file_type):
    sort_func = lambda filename: int(''.join(filter(str.isdigit, filename)))
    return sorted([f for f in os.listdir(path) if f.endswith(file_type)], key=sort_func)

def unify_text_and_labels_multiclass_simplelabels(texts_path, labels_path, save_path):
    """Unify text and labels into a single JSON file.
    The label scheme is one label per line (multiclass).
    We find all block intersections and the line label is given by the innermost block label.
    """
    
    print("\nUnifying text and labels, using a non-hierarchical approach only with simple labels, no B-I-O...")
    text_files = get_files(texts_path, 'txt')
    label_files = get_files(labels_path, 'json')
    
    for i_file in range(len(label_files)):
        # find all block ranges and block label
        with open(labels_path + '/' + label_files[i_file], 'r') as label_file:
            label_file_content = json.load(label_file)
            block_ranges_with_label = {}
            for block in label_file_content:
                label = block['block_type']
                start_line = block['start_line'] - 1
                end_line = block['end_line'] - 1
                if (start_line, end_line) in block_ranges_with_label:
                    print("Exact block intersection found in file:", label_files[i_file])
                else:
                    block_ranges_with_label[(start_line, end_line)] = label
        
        # find the correct label for each text line, by taking the smallest block's label for each line
        df = pd.DataFrame(columns=['sentence', 'label'])
        
        with open(texts_path + '/' + text_files[i_file], 'r', encoding="utf-8-sig", errors='ignore') as content_file:
            lines = content_file.readlines()
            lines = [line.strip() for line in lines]

            for i_line in range(len(lines)):
                blocks_in_line = [key for key in block_ranges_with_label if key[0] <= i_line <= key[1]]
                if len(blocks_in_line) == 0:
                    line_label = 'O'
                else:
                    smallest_block = min(blocks_in_line, key=lambda x: x[1] - x[0])
                    smallest_block_label = block_ranges_with_label[smallest_block]
                    if smallest_block_label == 'Other':
                        line_label = 'O'
                    else:
                        line_label = smallest_block_label
                df = pd.concat([df, pd.DataFrame({'sentence': [lines[i_line]], 'label': line_label})], ignore_index=True)
        
        # save the unified text and labels in a JSON file
        df.to_json(save_path + '/' + text_files[i_file].replace('.txt', '.json'), orient='records', indent=4, force_ascii=False)

def unify_text_and_labels_multiclass_no_hierarchy(texts_path, labels_path, save_path):
    """Unify text and labels into a single JSON file.
    The label scheme is BIO, one label per line (multiclass).
    We find all block intersections and the line label is given by the innermost block label.
    """
    
    print("\nUnifying text and labels, using a non-hierarchical approach...")
    text_files = get_files(texts_path, 'txt')
    label_files = get_files(labels_path, 'json')
    
    for i_file in range(len(label_files)):
        # find all block ranges and block label
        with open(labels_path + '/' + label_files[i_file], 'r') as label_file:
            label_file_content = json.load(label_file)
            block_ranges_with_label = {}
            for block in label_file_content:
                label = block['block_type']
                start_line = block['start_line'] - 1
                end_line = block['end_line'] - 1
                if (start_line, end_line) in block_ranges_with_label:
                    print("Exact block intersection found in file:", label_files[i_file])
                else:
                    block_ranges_with_label[(start_line, end_line)] = label
        
        # find the correct label for each text line, by taking the smallest block's label for each line
        df = pd.DataFrame(columns=['sentence', 'label'])
        
        with open(texts_path + '/' + text_files[i_file], 'r', encoding="utf-8-sig", errors='ignore') as content_file:
            lines = content_file.readlines()
            lines = [line.strip() for line in lines]

            for i_line in range(len(lines)):
                blocks_in_line = [key for key in block_ranges_with_label if key[0] <= i_line <= key[1]]
                if len(blocks_in_line) == 0:
                    line_label = 'O'
                else:
                    smallest_block = min(blocks_in_line, key=lambda x: x[1] - x[0])
                    smallest_block_label = block_ranges_with_label[smallest_block]
                    if smallest_block_label == 'Other':
                        line_label = 'O'
                    elif i_line == smallest_block[0]:
                        line_label = 'B-' + smallest_block_label
                    else:
                        line_label = 'I-' + smallest_block_label
                df = pd.concat([df, pd.DataFrame({'sentence': [lines[i_line]], 'label': line_label})], ignore_index=True)
        
        # save the unified text and labels in a JSON file
        df.to_json(save_path + '/' + text_files[i_file].replace('.txt', '.json'), orient='records', indent=4, force_ascii=False)

def unify_text_and_labels_multiclass(texts_path, labels_path, save_path):
    print("\nUnifying text and labels (version 2)...")
    text_files = get_files(texts_path, 'txt')
    label_files = get_files(labels_path, 'json')
    for i_file in range(len(label_files)):
        dict_label_range = {}
        with open(labels_path + '/' + label_files[i_file], 'r') as file:
            file_content = json.load(file)
            for block in file_content:
                label = block['block_type']
                if label in dict_label_range:
                    dict_label_range[label].append((block['start_line'] - 1, block['end_line'] - 1))
                else:
                    dict_label_range[label] = [(block['start_line'] - 1, block['end_line'] - 1)]
        value_hierarchie_single = [{tup: 1} for sublist in dict_label_range.values() for tup in sublist]
        #value_hierarchie = [{value : 1 for value in lst} for lst in dict_label_range.values()]
        for i in range(len(value_hierarchie_single)):
            current_value = value_hierarchie_single[i]
            current_start, current_end = list(current_value.keys())[0]
            for j in range(len(value_hierarchie_single)):
                if i == j:
                    continue
                potential_value = value_hierarchie_single[j]
                potential_start, potential_end = list(potential_value.keys())[0]
                if (current_start >= potential_start) and (current_end <= potential_end):
                    value_hierarchie_single[i][(current_start, current_end)] += 1
        
        doc = pd.DataFrame(columns=['sentence', 'label'])
        with open(texts_path + '/' + text_files[i_file], 'r', encoding="utf-8-sig", errors='ignore') as file:
            lines = file.readlines()
            line_id = 0
            for line in lines:
                #max_interval = (0,0)
                # check current max 0
                current_max = 0
                current_interval = (0, 0)
                #print(value_hierarchie_single)
                for i in range(len(value_hierarchie_single)):
                    current_value = value_hierarchie_single[i]
                    #print(current_value)
                    current_start, current_end = list(current_value.keys())[0]
                    #print(list(current_value.keys())[0])
                    if (line_id >= current_start) and (line_id <= current_end):
                        if value_hierarchie_single[i][(current_start, current_end)] >= current_max:
                            current_max = value_hierarchie_single[i][(current_start, current_end)]
                            current_interval = (current_start, current_end)
                for label in dict_label_range:
                    if current_interval in dict_label_range[label]:
                        current_label = label
                if current_max == 0 or current_label == 'Other':
                    doc = pd.concat([doc, pd.DataFrame({'sentence': [line], 'label': 'O'})], ignore_index=True)
                else:
                    if line_id == current_interval[0]:
                        doc = pd.concat([doc, pd.DataFrame({'sentence': [line], 'label': f'B-{current_label}-{current_max}'})], ignore_index=True)
                    else:
                        doc = pd.concat([doc, pd.DataFrame({'sentence': [line], 'label': f'I-{current_label}-{current_max}'})], ignore_index=True)
                line_id +=1
        doc.to_json(save_path + '/' + text_files[i_file].replace('.txt', '.json'), orient='records', indent=4, force_ascii=False)

def unify_text_and_labels_multiclass_v2(texts_path, labels_path, save_path):
    print("\nUnifying text and labels (version 2)...")
    text_files = get_files(texts_path, 'txt')
    label_files = get_files(labels_path, 'json')
    for i_file in range(len(label_files)):
        dict_label_range = {}
        with open(labels_path + '/' + label_files[i_file], 'r') as file:
            file_content = json.load(file)
            for block in file_content:
                label = block['block_type']
                if label in dict_label_range:
                    dict_label_range[label].append((block['start_line'] - 1, block['end_line'] - 1))
                else:
                    dict_label_range[label] = [(block['start_line'] - 1, block['end_line'] - 1)]
        value_hierarchie_single = [{tup: 1} for sublist in dict_label_range.values() for tup in sublist]
        #value_hierarchie = [{value : 1 for value in lst} for lst in dict_label_range.values()]
        for i in range(len(value_hierarchie_single)):
            current_value = value_hierarchie_single[i]
            current_start, current_end = list(current_value.keys())[0]
            for j in range(len(value_hierarchie_single)):
                if i == j:
                    continue
                potential_value = value_hierarchie_single[j]
                potential_start, potential_end = list(potential_value.keys())[0]
                if (current_start >= potential_start) and (current_end <= potential_end):
                    value_hierarchie_single[i][(current_start, current_end)] += 1
        
        doc = pd.DataFrame(columns=['sentence', 'label'])
        with open(texts_path + '/' + text_files[i_file], 'r', encoding="utf-8-sig", errors='ignore') as file:
            lines = file.readlines()
            lines = [clean_line(line) for line in lines]
            line_id = 0
            for line in lines:
                #max_interval = (0,0)
                # check current max 0
                current_max = 0
                current_interval = (0, 0)
                #print(value_hierarchie_single)
                for i in range(len(value_hierarchie_single)):
                    current_value = value_hierarchie_single[i]
                    #print(current_value)
                    current_start, current_end = list(current_value.keys())[0]
                    #print(list(current_value.keys())[0])
                    if (line_id >= current_start) and (line_id <= current_end):
                        if value_hierarchie_single[i][(current_start, current_end)] >= current_max:
                            current_max = value_hierarchie_single[i][(current_start, current_end)]
                            current_interval = (current_start, current_end)
                for label in dict_label_range:
                    if current_interval in dict_label_range[label]:
                        current_label = label
                if current_max == 0 or current_label == 'Other':
                    doc = pd.concat([doc, pd.DataFrame({'sentence': [line], 'label': 'O'})], ignore_index=True)
                else:
                    if line_id == current_interval[0]:
                        doc = pd.concat([doc, pd.DataFrame({'sentence': [line], 'label': f'B-{current_label}-{current_max}'})], ignore_index=True)
                    else:
                        doc = pd.concat([doc, pd.DataFrame({'sentence': [line], 'label': f'I-{current_label}-{current_max}'})], ignore_index=True)
                line_id +=1
        doc.to_json(save_path + '/' + text_files[i_file].replace('.txt', '.json'), orient='records', indent=4, force_ascii=False)

def unify_text_and_labels_multiclass(texts_path, labels_path, save_path):
    """Unify text and labels into a single JSON file.
    The labelling approach is a multiclass {BIO + hierarchy}.

    Args:
        texts_path (str): Path to the directory containing the text files.
        labels_path (str): Path to the directory containing the label files.
        save_path (str): Path to the directory where the unified json files will be saved.
    """
    
    print("\nUnifying text and labels...")
    text_files = get_files(texts_path, 'txt')
    label_files = get_files(labels_path, 'json')

    for i in range(len(label_files)):
        doc = pd.DataFrame(columns=['sentence', 'label'])
        # read each line and append it to the dataframe
        with open(texts_path + '/' + text_files[i], 'r', encoding="utf-8-sig", errors='ignore') as file:
            lines = file.readlines()
            lines = [clean_line(line) for line in lines]
            
            for line in lines:
                doc = pd.concat([doc, pd.DataFrame({'sentence': [line], 'label': None})], ignore_index=True)
        
        # read the labels and update the dataframe labels
        with open(labels_path + '/' + label_files[i], 'r') as file:
            file_content = json.load(file)
            for block in file_content:
                class_name = block['block_type']
                start = block['start_line'] - 1
                end = block['end_line'] - 1
                
                if class_name == 'Other':
                    for j in range(start, end + 1):
                        doc.at[j, 'label'] = 'O'
                else:
                    if doc.at[start, 'label'] == None:
                        doc.at[start, 'label'] = 'B-' + class_name + '-1'
                        level = 1
                    else:
                        curr_tag, curr_class_name, curr_h_level = doc.at[start, 'label'].split('-')
                        curr_h_level = int(curr_h_level)
                        level = curr_h_level + 1
                        
                        if curr_tag == 'B':
                            doc.at[start, 'label'] = 'B-' + curr_class_name + '+' + class_name + '-' + str(curr_h_level + 1)
                        else:
                            doc.at[start, 'label'] = 'B-' + class_name + '-' + str(curr_h_level + 1)
                        
                    for j in range(start + 1, end + 1):
                        if doc.at[j, 'label'] == None:
                            doc.at[j, 'label'] = 'I-' + class_name + '-' + str(level)
                        else:
                            curr_tag, curr_class_name, curr_h_level = doc.at[j, 'label'].split('-')
                            curr_h_level = int(curr_h_level)
                            
                            if curr_tag == 'B':
                                doc.at[j, 'label'] = 'B-' + curr_class_name + '-' + str(curr_h_level + 1)
                            else:
                                doc.at[j, 'label'] = 'I-' + class_name + '-' + str(curr_h_level + 1)
                
        # strip sentences and remove empty sentences
        doc['sentence'] = doc['sentence'].str.strip()
        doc = doc[doc['sentence'] != '']
        
        # change any None to 'O' (avoid any missing labels)
        doc['label'] = doc['label'].apply(lambda x: 'O' if x == None else x)
        
        # save the dataframe in a json file
        doc.to_json(save_path + '/' + text_files[i].replace('.txt', '.json'), orient='records', indent=4, force_ascii=False)

def unify_text_and_labels_multilabel(texts_path, labels_path, save_path):
    """Unify text and labels into a single JSON file.
    The label scheme is multilabel with a begin extra label
    We find all block intersections and the line label is given by the innermost block label.
    """
    
    print("\nUnifying text and labels, multilabel...")
    text_files = get_files(texts_path, 'txt')
    label_files = get_files(labels_path, 'json')
    
    for i_file in range(len(label_files)):
        # find all block ranges and block label
        with open(labels_path + '/' + label_files[i_file], 'r') as label_file:
            label_file_content = json.load(label_file)
            block_ranges_with_label = {}
            for block in label_file_content:
                label = block['block_type']
                start_line = block['start_line'] - 1
                end_line = block['end_line'] - 1
                if (start_line, end_line) in block_ranges_with_label:
                    print("Exact block intersection found in file:", label_files[i_file])
                else:
                    block_ranges_with_label[(start_line, end_line)] = label
        
        # find the correct labels for each text line, by taking all block labels for each line, add a begin if it is the first line of a block
        df_rows = []
        with open(texts_path + '/' + text_files[i_file], 'r', encoding="utf-8-sig", errors='ignore') as content_file:
            lines = content_file.readlines()
            lines = [line.strip() for line in lines] # remove newline chars

            for i_line, line in enumerate(lines):
                line_labels = set()
                # get all blocks that intersect with the current line and their labels
                blocks_in_line = [(key, block_ranges_with_label[key]) for key in block_ranges_with_label if key[0] <= i_line <= key[1]]
                if len(blocks_in_line) == 0:
                    line_labels.add('Other')
                else:
                    for key, label in blocks_in_line:
                        if i_line == key[0]: # if it is the first line of the block
                            line_labels.add(f'Begin {label}')
                            line_labels.add(label)
                        else:
                            line_labels.add(label)
                df_rows.append({'sentence': line, 'labels': list(line_labels)})
        
        # save text and labels in a JSON file
        df = pd.DataFrame(df_rows)
        df.to_json(save_path + '/' + text_files[i_file].replace('.txt', '.json'), orient='records', indent=4, force_ascii=False)
    
def create_label_dict(unified_path, save_path, task):
    """
    Create a dictionary of labels and save it to a json file.
    
    Args:
    unified_path: path to the directory containing the unified json files
    save_path: path to the directory where the labels dictionary will be saved
    """
    
    unified_files = get_files(unified_path, 'json')
    labels_set = set()
    
    for file in unified_files:
        with open(unified_path + '/' + file, 'r') as f:
            data = json.load(f)
            for block in data:
                if task == 'multilabel':
                    #ignore 'Other' label
                    labels = block['labels']
                    if 'Other' in labels:
                        labels.remove('Other')
                    labels_set.update(labels)
                elif task == 'multiclass':
                    labels_set.add(block['label'])
    
    print("Labels dictionary: ")
    label_dict = {}
    labels_set = sorted(labels_set)
    for i, label in enumerate(labels_set):
        label_dict[label] = i
        print(f"{label}: {i}")
        
    json.dump(label_dict, open(save_path + '/labels_dict.json', 'w'), indent=4)

def original_dataset_info(texts_path, labels_path):
    print("\n=== Dataset Information ===\n")

    text_files = get_files(texts_path, 'txt')
    label_files = get_files(labels_path, 'json')
    
    print(f"Number of text files: {len(text_files)}")
    print(f"Number of label files: {len(label_files)}\n")

    label_counts = Counter()

    for label_f in label_files:
        with open(os.path.join(labels_path, label_f), 'r') as file:
            file_content = json.load(file)
            for block in file_content:
                label_counts[block['block_type']] += block['end_line'] - block['start_line'] + 1

    label_counts = dict(sorted(label_counts.items(), key=lambda x: x[1], reverse=True))
    
    # display label information
    print("=== Original dataset labels ===")
    for label, count in label_counts.items():
        print(f"{label}: {count}")
    print(f"\nNumber of distinct labels: {len(label_counts)}\n")
    
    # plot documents length distribution
    doc_lengths = []
    for text_f in text_files:
        with open(os.path.join(texts_path, text_f), 'r') as file:
            doc_lengths.append(len(file.readlines()))
    plt.figure(figsize=(10, 6))
    plt.hist(doc_lengths, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Number of lines')
    plt.ylabel('Frequency')
    plt.title('Document Length Distribution', fontsize=16, fontweight='bold')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    
    
    # plot label distribution
    plt.figure(figsize=(10, 6))
    bars = plt.bar(label_counts.keys(), label_counts.values(), color='skyblue', edgecolor='black')
    plt.xlabel('Label', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Label Frequencies', fontsize=16, fontweight='bold')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.xticks(rotation=50, ha="right", fontsize=12)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.show()  
  
def unified_dataset_info(unified_path, task):
    print("\n=== Unified Dataset Information ===\n")
    unified_files = get_files(unified_path, 'json')    
    
    # plot label distribution
    label_counts = Counter()

    for file in unified_files:
        with open(unified_path + '/' + file, 'r') as f:
            data = json.load(f)
            for block in data:
                if task == 'multilabel':
                    #ignore 'Other' label
                    labels = block['labels']
                    if 'Other' in labels:
                        labels.remove('Other')
                    label_counts.update(labels)
                elif task == 'multiclass':
                    label_counts.update([block['label']])
                        
    label_counts = dict(sorted(label_counts.items(), key=lambda item: item[1], reverse=True))

    plt.figure(figsize=(15, 6))
    bars = plt.bar(label_counts.keys(), label_counts.values(), color='skyblue', edgecolor='black')
    plt.xlabel('Label', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Label Frequencies', fontsize=16, fontweight='bold')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.xticks(rotation=50, ha="right", fontsize=12)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.show()

    # display label information
    print("\nFinal labels:\n")
    if 'Other - artificially added' in label_counts:
        del label_counts['Other - artificially added']
    for label, count in label_counts.items():
        print(f"{label}: {count}")
    print(f"\nNumber of distinct labels: {len(label_counts)}")
    
def plot_all_combinations(unified_path):
    print("\n=== Label Combinations ===\n")
    unified_files = get_files(unified_path, 'json')
    all_combinations = Counter()
    
    for file in unified_files:
        with open(unified_path + '/' + file, 'r') as f:
            data = json.load(f)
            for block in data:
                if block['labels'] == None:
                    continue
                else:
                    combination = tuple(sorted(block['labels']))
                    all_combinations.update([combination])
    
    all_combinations = dict(sorted(all_combinations.items(), key=lambda item: item[1], reverse=False))
    
    print("Label combinations:")
    for comb, count in all_combinations.items():
        print(f"{comb}: {count}")
    
    n_combinations = len(all_combinations)
    print(f"\nNumber of distinct label combinations: {n_combinations}")
    
    plt.figure(figsize=(10, int(n_combinations / 4)))
    bars = plt.barh(
        [str(comb) for comb in all_combinations.keys()],
        all_combinations.values(),
        color='skyblue',
        edgecolor='black'
    )
    plt.xlabel('Frequency', fontsize=14)
    plt.ylabel('Label Combination', fontsize=14)
    plt.title('Label Combinations Frequencies', fontsize=16, fontweight='bold')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.yticks(fontsize=12)

    # Add text annotations for the bar values
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width, bar.get_y() + bar.get_height() / 2, 
            f'{int(width)}', ha='left', va='center', fontsize=10
        )

    plt.tight_layout()
    plt.show()
    plt.close()
    
def compare_label_dist(train_docs, val_docs, test_docs, labels_dict, task):
    """
    Count the number of occurrences of each label in the train, validation, and test datasets,
    plot the distribution, and compute similarity metrics.
    
    Args:
        train_docs (list): List of training documents.
        val_docs (list): List of validation documents.
        test_docs (list): List of testing documents.
        labels_dict (dict): A dictionary of label names as keys and initial counts (typically 0) as values.
        task (str): The task type, either 'multiclass' or 'multilabel'.
    """
    def count_labels(documents):
        """Helper function to count label occurrences in a dataset."""
        labels_counter = Counter(labels_dict)
        for doc in documents:
            for sentence in doc:
                if task == 'multiclass':
                    labels_counter.update([sentence["label"]])
                elif task == 'multilabel':
                    #ignore 'Other' label
                    labels = sentence["labels"]
                    if 'Other' in labels:
                        labels.remove('Other')
                    labels_counter.update(labels)
        return labels_counter
    
    # Count labels for each dataset
    train_counts = count_labels(train_docs)
    val_counts = count_labels(val_docs)
    test_counts = count_labels(test_docs)
    
    # Extract sorted labels and proportions
    labels = sorted(labels_dict.keys())
    train_proportions = [train_counts[label] / sum(train_counts.values()) if sum(train_counts.values()) > 0 else 0 for label in labels]
    val_proportions = [val_counts[label] / sum(val_counts.values()) if sum(val_counts.values()) > 0 else 0 for label in labels]
    test_proportions = [test_counts[label] / sum(test_counts.values()) if sum(test_counts.values()) > 0 else 0 for label in labels]
    
    # Compute similarity metrics
    train_val_js = jensenshannon(train_proportions, val_proportions)
    train_test_js = jensenshannon(train_proportions, test_proportions)
    val_test_js = jensenshannon(val_proportions, test_proportions)
    
    # Print similarity metrics
    print("\nJensen-Shannon Divergence:")
    print(f"  Train vs Validation: {train_val_js:.4f}")
    print(f"  Train vs Test: {train_test_js:.4f}")
    print(f"  Validation vs Test: {val_test_js:.4f}")
    
    # plot grouped bar chart
    x = np.arange(len(labels))
    bar_width = 0.25
    plt.figure(figsize=(12, 8))
    plt.bar(x - bar_width, train_proportions, bar_width, label="Train", color="skyblue", edgecolor="black")
    plt.bar(x, val_proportions, bar_width, label="Validation", color="lightgreen", edgecolor="black")
    plt.bar(x + bar_width, test_proportions, bar_width, label="Test", color="salmon", edgecolor="black")
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=10)
    plt.xlabel("Labels", fontsize=12)
    plt.ylabel("Proportion", fontsize=12)
    plt.title("Label Distribution Across Datasets", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # plot train frequencies
    plt.figure(figsize=(10, 6))
    bars = plt.bar(train_counts.keys(), train_counts.values(), color='skyblue', edgecolor='black')
    plt.xlabel('Label', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Train Label Frequencies', fontsize=12, fontweight='bold')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.xticks(rotation=50, ha="right", fontsize=9)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # plot validation frequencies
    plt.figure(figsize=(10, 6))
    bars = plt.bar(val_counts.keys(), val_counts.values(), color='lightgreen', edgecolor='black')
    plt.xlabel('Label', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Validation Label Frequencies', fontsize=12, fontweight='bold')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.xticks(rotation=50, ha="right", fontsize=9)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # plot test frequencies
    plt.figure(figsize=(10, 6))
    bars = plt.bar(test_counts.keys(), test_counts.values(), color='salmon', edgecolor='black')
    plt.xlabel('Label', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Test Label Frequencies', fontsize=12, fontweight='bold')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.xticks(rotation=50, ha="right", fontsize=9)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.show()
    plt.close()

def split_paths(data_path, split_path, test_ratio, val_ratio, seed=None):
    """Split the paths of the documents into train, validation and test sets.

    Args:
        data_path (str): Path to the directory containing the documents.
        test_ratio (float): Ratio of the documents to be used for testing.
        val_ratio (float): Ratio of the documents to be used for validation.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        train_doc_paths (list): List of paths to the training documents.
        val_doc_paths (list): List of paths to the validation documents.
        test_doc_paths (list): List of paths to the testing documents.
    """
    
    doc_names = [f for f in os.listdir(data_path) if f.endswith(".json")]

    full_doc_paths = [data_path + "/" + doc_name for doc_name in doc_names]

    train_doc_paths, test_doc_paths = train_test_split(
        full_doc_paths, test_size=(test_ratio + val_ratio), random_state=seed
    )

    val_doc_paths, test_doc_paths = train_test_split(
        test_doc_paths,
        test_size=(test_ratio / (test_ratio + val_ratio)),
        random_state=seed,
    )
    
    paths_dict = {
        "train": train_doc_paths,
        "val": val_doc_paths,
        "test": test_doc_paths,
    }
    
    json.dump(paths_dict, open(split_path + "/split_paths.json", "w"), indent=4)

    return train_doc_paths, val_doc_paths, test_doc_paths