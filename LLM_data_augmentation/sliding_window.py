import bisect
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(encoding.encode(text))

def get_sliding_window_indexes(contract, token_limit, overlap_fraction, ORIGINAL_PATH):
    file_path = f'{ORIGINAL_PATH}/{contract}.txt'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cumulative = []
    total = 0
    for line in lines:
        tokens = count_tokens(line)
        total += tokens
        cumulative.append(total)
    
    windows = []

    step = token_limit - int(token_limit * overlap_fraction)
    start_token = 0
    
    while start_token < total:
        end_token = start_token + token_limit
        
        start_index = bisect.bisect_left(cumulative, start_token)
        end_index = bisect.bisect_right(cumulative, end_token)

        if start_index < len(lines):
            windows.append((start_index, end_index))
        else:
            break
        
        start_token += step
    
    return windows

def main():
    return 0

if __name__ == '__main__':
    main()