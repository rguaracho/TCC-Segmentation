import sys
import re
from fuzzywuzzy import fuzz, process
import pandas as pd
import json

def clean_line(line):
    return re.sub(r'<[^>]+>', '', line)

def map_tags(lines):
    mapping = {}
    tag_pattern = re.compile(r'<([^>]+)>')
    for i, line in enumerate(lines, start=1):
        tags = tag_pattern.findall(line)
        mapping[i] = tags if tags else None 
    return mapping

def map_original_to_llm(original_lines, llm_lines, fuzzy_threshold=80):
    orig_stripped = [line.strip() for line in original_lines]
    llm_stripped = [line.strip() for line in llm_lines]
    
    llm_dict = {i: line for i, line in enumerate(llm_stripped, start=1)}
    skipped_lines = 1
    mapping = {}
    for orig_index, orig_line in enumerate(orig_stripped, start=1):
        direct_match = False
        if orig_line == "":
                mapping[orig_index] = None
                continue
        if len(llm_stripped) > orig_index - skipped_lines:
            direct_line = llm_stripped[orig_index - skipped_lines]
            direct_score = fuzz.ratio(orig_line, direct_line)
            if direct_score >= fuzzy_threshold:
                direct_match = True
                mapping[orig_index] = 1 + orig_index - skipped_lines
        if not(direct_match):
            result = process.extractOne(orig_line, llm_dict, scorer=fuzz.ratio, score_cutoff=fuzzy_threshold-20)
            if result is not None:
                _, _, llm_index = result
                mapping[orig_index] = llm_index
            else:
                mapping[orig_index] = None
                skipped_lines+=1
    mapping[len(original_lines)] = len(llm_lines) #attribute last line ref
    return mapping

def merge_tag_mappings(original_to_llm_map, llm_tags_map):
    merged_mapping = {}
    for orig_line, llm_line in original_to_llm_map.items():
        if llm_line is not None:
            merged_mapping[orig_line] = llm_tags_map.get(llm_line, None)
        else:
            merged_mapping[orig_line] = None
    for llm_line, label in llm_tags_map.items():
        if llm_line not in original_to_llm_map.values() and label is not None:
            found = False
            for orig_line, llm_line_j in original_to_llm_map.items():
                if llm_line_j is not None and llm_line_j > llm_line:
                    if orig_line - 1 in merged_mapping:
                        if merged_mapping[orig_line - 1] is not None:
                            for _label in label:
                                merged_mapping[orig_line - 1].append(_label)
                        else:
                            merged_mapping[orig_line - 1] = label
                    else:
                        merged_mapping[orig_line - 1] = label
                    found = True
                    break
            if not found:
                if merged_mapping[orig_line - 1]:
                    for _label in label:
                        merged_mapping[orig_line - 1].append(_label)
                else:
                    merged_mapping[orig_line - 1] = label

    return merged_mapping

def map_tag_intervals(line_tags):
    stacks = {}
    intervals = {}

    for line_no in sorted(line_tags.keys()):
        tags = line_tags[line_no]
        if not tags:
            continue

        i = 0
        while i < len(tags):
            tag = tags[i].strip()
            if not tag.startswith('/'):
                if (i + 1 < len(tags)) and (tags[i + 1].strip() == '/' + tag):
                    intervals.setdefault(tag, []).append([line_no, line_no])
                    i += 2
                else:
                    stacks.setdefault(tag, []).append(line_no)
                    i += 1
            else:
                tag_name = tag[1:].strip()
                if tag_name in stacks and stacks[tag_name]:
                    start_line = stacks[tag_name].pop()
                    intervals.setdefault(tag_name, []).append([start_line, line_no])
                else:
                    pass
                i += 1
    return intervals

def process_output(file_with_tags, original_file):
    # Read file with tags
    with open(file_with_tags, 'r', encoding='utf-8') as f:
        lines_with_tags = f.readlines()
    tag_maps = map_tags(lines_with_tags)
    
    # Read the original file with the correct text
    with open(original_file, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()
    
    # Remove tags
    cleaned_lines = [clean_line(line) for line in lines_with_tags]
    
    mapping = map_original_to_llm(original_lines, cleaned_lines, fuzzy_threshold=80)
    print(mapping)
    print()
    print(tag_maps)
    print()

    merged_map = merge_tag_mappings(mapping, tag_maps)
    print(merged_map)

    tag_intevals_map = map_tag_intervals(merged_map)
    return tag_intevals_map

def generate_LLM_from_txt_json(original_file, json_file, contract, save_path='dataset/LLM_output'):
    # TO DO: optimize (remove hierarchy calculation)
    """Unify text and labels into a single JSON file.
    The labelling approach is a multiclass {BIO}.
    It also generates new .txt files with the treatment that was applied to the text.

    Args:
        texts_path (str): Path to the directory containing the text files.
        labels_path (str): Path to the directory containing the label files.
        save_path (str): Path to the directory where the unified json files will be saved.
    """
    dict_label_range = {}
    with open(json_file, 'r') as file:
        file_content = json.load(file)
        for block in file_content:
            label = block['name']
            if label in dict_label_range:
                dict_label_range[label].append((block['start'] - 1, block['end'] - 1))
            else:
                dict_label_range[label] = [(block['start'] - 1, block['end'] - 1)]
    value_hierarchie_single = [{tup: 1} for sublist in dict_label_range.values() for tup in sublist]
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
    with open(original_file, 'r', encoding="utf-8-sig", errors='ignore') as file:
        lines = file.readlines()
        line_id = 0
        last_label = 'O'
        for line in lines:
            current_max = 0
            current_interval = (0, 0)
            for i in range(len(value_hierarchie_single)):
                current_value = value_hierarchie_single[i]
                current_start, current_end = list(current_value.keys())[0]
                if (line_id >= current_start) and (line_id <= current_end):
                    if value_hierarchie_single[i][(current_start, current_end)] >= current_max:
                        current_max = value_hierarchie_single[i][(current_start, current_end)]
                        current_interval = (current_start, current_end)
            for label in dict_label_range:
                if current_interval in dict_label_range[label]:
                    current_label = label
            if current_max == 0 or current_label == 'Other':
                if last_label.startswith('B'):
                    _label = f'I{last_label[1:]}'
                else:
                    _label = last_label
                doc = pd.concat([doc, pd.DataFrame({'sentence': [line], 'label': _label})], ignore_index=True)
            else:
                if line_id == current_interval[0]:
                    doc = pd.concat([doc, pd.DataFrame({'sentence': [line], 'label': f'B-{current_label}'})], ignore_index=True)
                    last_label = f'B-{current_label}'
                else:
                    doc = pd.concat([doc, pd.DataFrame({'sentence': [line], 'label': f'I-{current_label}'})], ignore_index=True)
                    last_label = f'I-{current_label}'
            line_id +=1
    doc['sentence'] = doc['sentence'].str.strip()
    doc.to_json(save_path + f'/{contract}.json', orient='records', indent=4, force_ascii=False)


def main(file_with_tags, original_file):
    process_output(file_with_tags, original_file)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python LLM_process_output.py <file_with_tags> <original_file>")
        sys.exit(1)
    file_with_tags = sys.argv[1]
    original_file = sys.argv[2]
    main(file_with_tags, original_file)
