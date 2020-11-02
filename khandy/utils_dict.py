import random
from collections import OrderedDict


def get_dict_first_item(dict_obj):
    for key in dict_obj:
        return key, dict_obj[key]


def sort_dict(dict_obj, key=None, reverse=False):
    return OrderedDict(sorted(dict_obj.items(), key=key, reverse=reverse))


def create_class_dict(name_list, label_list):
    assert len(name_list) == len(label_list)
    class_dict = {}
    for name, label in zip(name_list, label_list):
        class_dict.setdefault(label, []).append(name)
    return class_dict
    
    
def convert_class_dict_to_list(class_dict):
    name_list, label_list = [], []
    for key, value in class_dict.items():
        name_list += value
        label_list += [key] * len(value)
    return name_list, label_list
    
    
def convert_class_dict_to_records(class_dict, label_map=None, raise_if_key_error=True):
    records = []
    if label_map is None:
        for label in class_dict:
            for name in class_dict[label]:
                records.append('{},{}'.format(name, label))
    else:
        for label in class_dict:
            if raise_if_key_error:
                mapped_label = label_map[label]
            else:
                mapped_label = label_map.get(label, label)
            for name in class_dict[label]:
                records.append('{},{}'.format(name, mapped_label))
    return records
    
    
def sample_class_dict(class_dict, num_classes, num_examples_per_class=None):
    num_classes = min(num_classes, len(class_dict))
    sub_keys = random.sample(list(class_dict), num_classes)
    if num_examples_per_class is None:
        sub_class_dict = {key: class_dict[key] for key in sub_keys}
    else:
        sub_class_dict = {}
        for key in sub_keys:
            num_examples_inner = min(num_examples_per_class, len(class_dict[key]))
            sub_class_dict[key] = random.sample(class_dict[key], num_examples_inner)
    return sub_class_dict
    
    
def split_class_dict_on_key(class_dict, split_ratio, use_shuffle=False):
    """Split class_dict on its key.
    """
    assert isinstance(class_dict, dict)
    assert isinstance(split_ratio, (list, tuple))
    
    pdf = [k / float(sum(split_ratio)) for k in split_ratio]
    cdf = [sum(pdf[:k]) for k in range(len(pdf) + 1)]
    indices = [int(round(len(class_dict) * k)) for k in cdf]
    dict_keys = list(class_dict)
    if use_shuffle: 
        random.shuffle(dict_keys)
        
    be_split_list = []
    for i in range(len(split_ratio)):
        #if indices[i] != indices[i + 1]:
        part_keys = dict_keys[indices[i]: indices[i + 1]]
        part_dict = dict([(key, class_dict[key]) for key in part_keys])
        be_split_list.append(part_dict)
    return be_split_list
    
    
def split_class_dict_on_value(class_dict, split_ratio, use_shuffle=False):
    """Split class_dict on its value.
    """
    assert isinstance(class_dict, dict)
    assert isinstance(split_ratio, (list, tuple))
    
    pdf = [k / float(sum(split_ratio)) for k in split_ratio]
    cdf = [sum(pdf[:k]) for k in range(len(pdf) + 1)]
    be_split_list = [dict() for k in range(len(split_ratio))] 
    for key, value in class_dict.items():
        indices = [int(round(len(value) * k)) for k in cdf]
        cloned = value[:]
        if use_shuffle: 
            random.shuffle(cloned)
        for i in range(len(split_ratio)):
            #if indices[i] != indices[i + 1]:
            be_split_list[i][key] = cloned[indices[i]: indices[i + 1]]
    return be_split_list
    
    
def get_class_dict_info(class_dict, with_print=False, desc=None):
    num_list = [len(val) for val in class_dict.values()]
    num_classes = len(num_list)
    num_examples = sum(num_list)
    max_examples_per_class = max(num_list)
    min_examples_per_class = min(num_list)
    if num_classes == 0:
        avg_examples_per_class = 0
    else:
        avg_examples_per_class = num_examples / num_classes
    info = {
        'num_classes': num_classes,
        'num_examples': num_examples,
        'max_examples_per_class': max_examples_per_class,
        'min_examples_per_class': min_examples_per_class,
        'avg_examples_per_class': avg_examples_per_class,
    }
    if with_print:
        desc = desc or '<unknown>'
        print('{} subject number:    {}'.format(desc, info['num_classes']))
        print('{} example number:    {}'.format(desc, info['num_examples']))
        print('{} max number per-id: {}'.format(desc, info['max_examples_per_class']))
        print('{} min number per-id: {}'.format(desc, info['min_examples_per_class']))
        print('{} avg number per-id: {:.2f}'.format(desc, info['avg_examples_per_class']))
    return info
    

def filter_class_dict_by_number(class_dict, lower, upper=None):
    if upper is None:
        return {key: value for key, value in class_dict.items() 
                if lower <= len(value) }
    else:
        assert lower <= upper, 'lower must not be greater than upper'
        return {key: value for key, value in class_dict.items() 
                if lower <= len(value) <= upper }
        
        
def sort_class_dict_by_number(class_dict, num_classes_to_keep=None, reverse=True):
    """
    Args:
        reverse: sort in ascending order when is True.
    """
    if num_classes_to_keep is None: 
        num_classes_to_keep = len(class_dict)
    else:
        num_classes_to_keep = min(num_classes_to_keep, len(class_dict))
    sorted_items = sorted(class_dict.items(), key=lambda x: len(x[1]), reverse=reverse)
    filtered_dict = OrderedDict()
    for i in range(num_classes_to_keep):
        filtered_dict[sorted_items[i][0]] = sorted_items[i][1]
    return filtered_dict

    
def merge_class_dict(*class_dicts):
    merged_class_dict = {}
    for item in class_dicts:
        for key, value in item.items():
            merged_class_dict.setdefault(key, []).extend(value)
    return merged_class_dict
    
    
