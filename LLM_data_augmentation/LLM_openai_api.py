import os
import time
from openai import OpenAI
from LLM_prompt_generator import get_prompts, get_instructions_prompt
from LLM_process_output import process_output, generate_LLM_from_txt_json
from LLM_process_file import process_files
from LLM_retriever_window import Retriever
from sliding_window import get_sliding_window_indexes
import threading
import json
import re

WINDOW_SIZE = 6000
CONTEXT_RATIO = 0.8
SAVE_PATH = "dataset/llm_response"

ORIGINAL_PATH = "dataset/unified_LLM"
ORIGINAL_PATH_BACKUP = "dataset/original_unified_LLM"

#ORIGINAL_PATH = "dataset/batch_norm"
#ORIGINAL_PATH_BACKUP = "dataset/batch_norm"

LABEL_PATTERN = r'^\{(\d+)\}\s*\{([^}]+)\}\{([^}]+)\}.*:'

def process_contract(contract, retriever):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        organization='org-WqUqepXNWZUOI89kQiQ8DlGj',
        project='proj_BSaofHATJzbJDlpysYinBZQx',
    )

    template_messages = []
    template_messages.append({
        "role": "system",
        "content": get_instructions_prompt()
    })

    global_start_time = time.time()
    api_request_deltas = []
    process_output_deltas = []

    query_mission_template = [
        "Now classify the following lines from the contract.",
        "Remember to follow the exact format:",
        "{line_number} {label1}{label2}{ (Begin label) if new}: {original_text}",
        "Make sure to include all applicable labels for nested structures."
    ]

    # Reset LLM_response file
    open(SAVE_PATH + f'/{contract}.txt', 'w', encoding='utf-8-sig')

    with open(f'{ORIGINAL_PATH}/{contract}.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    window_indexes = get_sliding_window_indexes(contract, WINDOW_SIZE, 0, ORIGINAL_PATH)
    print(window_indexes)
    previous_response = None
    last_line = ""
    last_begin_line_1 = ""
    last_begin_line_2 = ""
    for current, window in enumerate(window_indexes):
        i, j = window
        query = "".join(lines[i:j]) if current < len(window_indexes) - 1 else "".join(lines[i:])

        messages = template_messages[:]
        prompts = get_prompts(retriever, query, 2) # maybe get context only with the query / full contract ?
        context_prompts = [
            entry
            for input, expected_output in prompts["context_prompts"]
            for entry in (
                {"role": "user", "content": input},
                {"role": "assistant", "content": expected_output},
            )
        ]

        messages.extend(context_prompts)
        query_mission = query_mission_template[:]

        if previous_response is not None:
            """
            previous_response["assistant"] += "\n"
            previous_response["assistant"] += "".join(lines[i:i+5])
            print(previous_response)
            context_prompts.append(
                #{"role": "user", "content": previous_response["user"]}, 
                {"role": "assistant","content": previous_response["assistant"]}
            )
            """
            query_mission.extend(["Note that the last Begin Articles you classified were at:",
                                  last_begin_line_2,
                                  last_begin_line_1,
                                  "So, only start new Article if it has the same marker level.",
                                  "Also, the last line you classified was:",
                                  last_line,
                                  f"Thus, you might keep this label (do not Begin it again).\n"])
            """
            messages.append({
                "role": "user",
                "content": previous_response["assistant"],
            })
            """

        query = "\n".join(query_mission) + query
        
        messages.append({
            "role": "user",
            "content": query,
        })

        if current == 1:
            with open('dataset/prompt_example/1.txt', 'w', encoding='utf-8-sig') as html_file:
                html_file.write("system prompt\n")
                html_file.write(get_instructions_prompt())
                html_file.write("context prompt\n")
                input, output = context_prompts[0] #prompts["context_prompts"]
                html_file.write(input)
                html_file.write(output)
                html_file.write("query prompt\n")
                html_file.write(query)

        print("FIRST LINES OF NEXT QUERY")
        print(query[:3])

        ts_api = time.time()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0
        )
        tf_api = time.time()
        api_request_deltas.append(tf_api - ts_api)

        raw_content = response.choices[0].message.content
        response_lines = raw_content.split("\n")
        cleaned_lines = [r_line.strip() for r_line in response_lines if r_line.strip()]

        print("START ORIGINAL OUTPUT")
        for x in cleaned_lines[:3]:
            print(x)

        print("END ORIGINAL OUTPUT")
        for x in cleaned_lines[-3:]:
            print(x)

        # Erase label from last line (prevent closing a continuous label)
        """
        if current < len(window_indexes) - 1:
            if not cleaned_lines[-1].startswith("<"):
                #cleaned_lines[-1] += " [CONTINUOUS]"
                cleaned_lines[-1] = re.sub(r'<[^>]+>', '', cleaned_lines[-1])
            elif cleaned_lines[-1].startswith("</"):
                #cleaned_lines[-1] += " [CONTINUOUS]"
                cleaned_lines = cleaned_lines[:len(cleaned_lines) - 1]
        """
        """
        if current < len(window_indexes) - 1:
            if not cleaned_lines[-1].startswith("<"):
                #cleaned_lines[-1] += " [CONTINUOUS]"
                cleaned_lines[-1] = re.sub(r'<[^>]+>', '', cleaned_lines[-1])
            elif cleaned_lines[-1].startswith("</"):
                #cleaned_lines[-1] += " [CONTINUOUS]"
                cleaned_lines = cleaned_lines[:len(cleaned_lines) - 1]
        """
        """
        if current > 0: #and current < len(window_indexes) - 1:
            if last_label:
                first_label_match = START_TAG_PATTERN.search(cleaned_lines[0])
                if first_label_match and first_label_match.group(1) == last_label:
                    print("limpei")
                    print(cleaned_lines[0])
                    cleaned_lines[0] = re.sub(r'<[^>]+>', '', cleaned_lines[0])
                    print(cleaned_lines[0])
        """
        if cleaned_lines[0] == "EXPECTED OUTPUT:":
            cleaned_lines = cleaned_lines[1:]
        cleaned_content = "\n".join(cleaned_lines)
        if current < len(window_indexes) - 1:
            cleaned_content += "\n"

        with open(SAVE_PATH + f'/{contract}.txt', 'a', encoding='utf-8-sig') as html_file:
            html_file.write(cleaned_content)
            html_file.close()

        
        print(" for next resp")
        print(cleaned_lines[-4:])

        if current < len(window_indexes) - 1:    
            print("looking for match")     
            previous_response = True   
            last_line = cleaned_lines[len(cleaned_lines)-1]
            last_begin_line_1 = ""
            last_begin_line_2 = ""

            for line_j in range(len(cleaned_lines) - 1, 0, -1):
                '''
                last_label_match = re.match(LABEL_PATTERN, cleaned_lines[line_j])
                print(last_label_match)
                if last_label_match and last_label == None:
                    _ = last_label_match.group(1)
                    last_label = last_label_match.group(2)
                    last_line = cleaned_lines[line_j]
                    previous_response = True
                '''
                if last_begin_line_1 == "":
                    if "Begin Article" in cleaned_lines[line_j]:
                        last_begin_line_1 = cleaned_lines[line_j]
                elif last_begin_line_2 == "":
                    if "Begin Article" in cleaned_lines[line_j]:
                        last_begin_line_2 = cleaned_lines[line_j]
                        break

        print()
        print("LAST LINE + BEGINS")
        print(last_line)
        print(last_begin_line_1)
        print(last_begin_line_2)
        print(previous_response == None)

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        print("\nToken Usage:")
        print(f"Prompt Tokens: {prompt_tokens}")
        print(f"Completion Tokens: {completion_tokens}")
        print(f"Total Tokens: {total_tokens}")


    llm_output_file = f'{SAVE_PATH}/{contract}.txt'
    original_file = f'{ORIGINAL_PATH_BACKUP}/{contract}.txt'

    ts_output = time.time()
    #tag_intervals_map = process_output(llm_output_file, original_file)
    #results = process_files(llm_output_file, original_file)

    tf_output = time.time()
    process_output_deltas.append(tf_output - ts_output)
    """
    with open(SAVE_PATH + f'/{contract}.json', 'w', encoding='utf-8') as out_file:
        json.dump(results, out_file, indent=4, ensure_ascii=False)
    """
    return

    json_list = []
    slices = [index+1 for _, index in window_indexes][:-1]
    print(slices)
    for tag, intervals in tag_intervals_map.items():
        for start, end in intervals:
            json_list.append({
                "name": tag,
                "start": start,
                "end": end
            })
    result_json = merge_tag_intervals(tag_intervals_map, slices)
    print(result_json)

    with open(SAVE_PATH + f'/{contract}.json', "w") as file:
        json.dump(result_json, file, indent=2)

    generate_LLM_from_txt_json(original_file, f'{SAVE_PATH}/{contract}.json', contract)
    global_end_time = time.time()

    print()
    print("Execution time: {:.6f} seconds".format(global_end_time - global_start_time))
    print(f"API request times: {api_request_deltas}")
    print(f"process output times: {process_output_deltas}")


def get_slice_index(pos, slices):
    """
    Given a position and a sorted list of slice boundaries, return the index
    of the slice in which pos falls. (Assumes slices defines the end-boundaries.)
    """
    for i, boundary in enumerate(slices):
        if pos < boundary:
            return i
    return len(slices) - 1  # fallback: pos is in the last slice

def merge_tag_intervals(tag_intervals_map, slices):
    json_list = []
    
    for tag, intervals in tag_intervals_map.items():
        # Sort intervals by their start value.
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = []
        
        for start, end in intervals:
            if merged:
                last = merged[-1]
                # Check if the previous interval's end equals the current interval's start
                # and that both positions fall into the same slice.
                if last['end'] == start and get_slice_index(last['end'], slices) == get_slice_index(start, slices):
                    # Merge by extending the end.
                    last['end'] = end
                    continue
            # Otherwise, add a new JSON object for this interval.
            merged.append({"name": tag, "start": start, "end": end})
        
        json_list.extend(merged)
    
    return json_list


def main():
    # Load shared resources (for example, the retriever state)
    retriever = Retriever.load("retriever/retriever_state.pkl")

    split_paths = "dataset/LLM_split_paths.json"
    with open(split_paths, 'r') as f:
        config = json.load(f)
    test_files = config.get("test", [])
    test_file_names = [os.path.basename(file_path)[:-5] for file_path in test_files]
    threads = []
    """
    test_file_names = ["contract1", "contract2", "contract3",
                       "contract4", "contract5", "contract6",
                       "contract7", "contract8", "contract9", 
                       "contract10"]
    #test_file_names = ["contract4"]
    threads = []
    all_items = sorted(os.listdir(ORIGINAL_PATH))
    #print(all_items[:10]) # 0 - 9
    #print(all_items[10:20]) # 10 - 19
    test_file_names = [file_path[:-4] for file_path in all_items]
    test_file_names = test_file_names[150:]
    test_file_names = ["EN_AlliedEsportsEntertainmentInc_20190815_8-K_EX-10.34_11788308_EX-10.34_Sponsorship Agreement"]
    """
    test_file_names = ["contract3"]
    for contract in test_file_names:
        t = threading.Thread(target=process_contract, args=(contract, retriever))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("All contracts processed.")

if __name__ == "__main__":
    main()