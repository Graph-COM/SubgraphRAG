import numpy as np
from collections import defaultdict


def triplet_to_str(triplet):
    return f"({triplet[0]},{triplet[1]},{triplet[2]})"


def unique_preserve_order(input_list):
    seen = set()
    unique_list = []
    for item in input_list:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)
    return unique_list


def remove_same_head_tail(triplets, mode):
    if 'rmht' not in mode:
        return triplets

    new_triplets = []
    seen = set()
    for triplet in triplets:
        item_1 = ','.join([str(triplet[0]), str(triplet[2])])
        item_2 = ','.join([str(triplet[2]), str(triplet[0])])
        if item_1 not in seen and item_2 not in seen:
            seen.add(item_1)
            seen.add(item_2)
            new_triplets.append(triplet)
    return new_triplets


def merge_tuples(tuple_list, mode=0):
    if mode == 0:
        merged_dict = defaultdict(lambda: [[], None, None])
        for t in tuple_list:
            key = (t[1], t[2])  # Group by the second and third elements
            merged_dict[key][0].append(t[0])  # Append the first element to the list
            merged_dict[key][1] = t[1]  # Set the second element
            merged_dict[key][2] = t[2]  # Set the third element

        # Convert the dictionary back to a list of merged tuples
        return [('[' + ','.join(v[0]) + ']', v[1], v[2]) for v in merged_dict.values()]
    else:
        assert mode == 2
        merged_dict = defaultdict(lambda: [None, None, []])
        for t in tuple_list:
            key = (t[0], t[1])
            merged_dict[key][2].append(t[2])
            merged_dict[key][0] = t[0]
            merged_dict[key][1] = t[1]
        return [(v[0], v[1], '[' + ','.join(v[2]) + ']') for v in merged_dict.values()]


def get_prompts(each_qa, mode, sys_prompt, cot_prompt, thres, seed=0):
    question_prompt = "Question:\n" + each_qa['question']
    if question_prompt[-1] != '?':
        question_prompt += '?'

    if 'rog' in mode:
        num_sampled_triplets = int(mode.split('_')[1])
        good_triplets_rog = each_qa['good_triplets_rog']
        input_triplets = remove_same_head_tail(good_triplets_rog, mode)
        # sampled_triplets = np.array(each_qa[f'sampled_triplets_{num_sampled_triplets}'])
        # input_triplets = np.concatenate([good_triplets_rog, sampled_triplets]) if len(good_triplets_rog) > 0 else sampled_triplets
        input_triplets = [triplet_to_str(triplet) for triplet in input_triplets]
        other_triplets = remove_same_head_tail(each_qa['scored_triplets'], mode)
        other_triplets = [triplet_to_str(triplet) for triplet in other_triplets]
        input_triplets = unique_preserve_order(input_triplets + other_triplets)
        input_triplets = input_triplets[:num_sampled_triplets]
        # input_triplets = np.random.permutation(input_triplets)
        triplet_prompt = "Triplets:\n" + "\n".join(input_triplets)
    elif 'scored' in mode:
        num_sampled_triplets = int(mode.split('_')[1])
        input_triplets = each_qa['scored_triplets']
        if thres:
            input_triplets = [(triplet[0], triplet[1], triplet[2]) for triplet in input_triplets if triplet[3] >= thres]
        else:
            input_triplets = [(triplet[0], triplet[1], triplet[2]) for triplet in input_triplets]

        input_triplets = unique_preserve_order(input_triplets)
        input_triplets = input_triplets[:num_sampled_triplets]
        input_triplets = [triplet_to_str(triplet) for triplet in input_triplets]
        if 'rev' in mode:
            input_triplets.reverse()
        triplet_prompt = "Triplets:\n" + "\n".join(input_triplets)

    elif 'rand' in mode:
        num_sampled_triplets = int(mode.split('_')[1])
        np.random.seed(seed)
        input_triplets = np.random.permutation(np.array(each_qa['graph']))
        if 'randNoA' in mode:
            for each_a in each_qa['a_entity']:
                input_triplets = [triplet for triplet in input_triplets if each_a not in triplet[0] and each_a not in triplet[2]]

        input_triplets = unique_preserve_order([triplet_to_str(triplet) for triplet in input_triplets])
        input_triplets = input_triplets[:num_sampled_triplets]
        triplet_prompt = "Triplets:\n" + "\n".join(input_triplets)
    elif 'noevi' in mode:
        triplet_prompt = ''
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if 'firstq' in mode:
        all_query = "\n\n".join([sys_prompt, question_prompt, triplet_prompt])
        user_query = "\n\n".join([question_prompt, triplet_prompt])
    else:
        all_query = "\n\n".join([sys_prompt, triplet_prompt, question_prompt])
        user_query = "\n\n".join([triplet_prompt, question_prompt])
        if triplet_prompt == '':
            user_query = question_prompt

    each_qa['sys_query'] = sys_prompt
    each_qa['user_query'] = user_query
    each_qa['all_query'] = all_query
    each_qa['cot_query'] = cot_prompt
    return each_qa


def get_prompts_for_data(data, mode, sys_prompt, cot_prompt, thres):
    new_data = []
    for each_qa in data:
        new_data.append(get_prompts(each_qa, mode, sys_prompt, cot_prompt, thres))
    return new_data
