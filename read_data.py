from brat_parser import get_entities_relations_attributes_groups
import pandas as pd
from collections import defaultdict
import os
import glob


def read_ann(path):

    entities, relations, attributes, groups = get_entities_relations_attributes_groups(path)

    annotated_note_entity_id = None

    # Find id for "Annotert_Notat" entity, to use when choosing relevant attributes (target values)
    for key, value in entities.items():
        if value.type == "Annotert_Notat":
            annotated_note_entity_id = key
            break
    
    if annotated_note_entity_id is None:
        print("Error: .ann file is missing attribute called Annotert_Notat containing target variables")
        quit()

    return annotated_note_entity_id, attributes


def read_all_ann():
    # synthetic training data
    path1 = r'path/to/path/1'
    # all real labeled training data from annotation sessions 1 and 2
    path2 = r'path/to/path/2'

    all_files = []

    for path in [path1, path2]:
        files = glob.glob(path + "*.ann")
        all_files += files

    # empty dict to hold all annotated files
    ann_dict = {}

    # for each file in the path above ending .ann ...
    for file in all_files:
        annotated_note_entity_id, attribute_dict = read_ann(file)
        file_id = os.path.basename(file).split('.')[0]
        ann_dict[file_id] = [annotated_note_entity_id, attribute_dict]

    return ann_dict


def read_all_raw():
    # synthetic training data
    path1 = r'path/to/path/1'
    # all real labeled training data from annotation sessions 1 and 2
    path2 = r'path/to/path/2'
    
    all_files = []

    for path in [path1, path2]:
        files = glob.glob(path + "*.txt")
        all_files += files

    results = defaultdict(list)

    # for each file in the path above ending .txt ...
    for file in all_files:
        with open(file, "r") as file_open:

            if os.path.basename(file).split('.')[0] in results["filename"]:
                continue
            else:
                results["filename"].append(os.path.basename(file).split('.')[0])
                results["text"].append(file_open.read())

    raw_files = pd.DataFrame(results)

    return raw_files


def get_real_file_ids_to_remove(include_unlabeled=False):
    
    # all real labeled training data from annotation sessions 1 and 2, to remove from further test sets
    path_labeled = r'path/to/path'
    
    # find all labeled files to remove
    labeled_files_to_remove = []
    labeled_files = glob.glob(path_labeled + "*.txt")
    for file in labeled_files:
        labeled_files_to_remove.append(os.path.basename(file).split('.')[0])

    print("labeled files to remove:", len(labeled_files_to_remove))
    
    if(include_unlabeled):
        # all unlabeled real notes used for semi supervised 
        path_unlabeled = r'path/to/path'
        # find all unlabeled files to remove
        unlabeled = pd.read_pickle(path_unlabeled)
        unlabeled_files_to_remove = unlabeled.filename.values.tolist()

        print("unlabeled files to remove:", len(unlabeled_files_to_remove))

        # join the two lists
        files_to_remove = labeled_files_to_remove + unlabeled_files_to_remove
        
    else:
        files_to_remove = labeled_files_to_remove

    print("files to remove:", len(set(files_to_remove)))

    return files_to_remove


def read_all_real():
    path = r'path/to/path'
    all_files = glob.glob(path + "*.txt")

    # Choose subset size
    # subset = all_files[:1000]

    real_files = {}

    # for each file in the path above ending .txt ...
    # for file in subset:
    for file in all_files:
        with open(file, "r") as file_open:
            file_id = os.path.basename(file).split('.')[0]
            real_files[file_id] = file_open.read()

    return real_files


def read_all_predicted_ann():
    # predicted data from final annotation meeting with true labels
    path = r'path/to/path'

    all_files = glob.glob(path + "*.ann")

    # empty dict to hold all annotated files
    ann_dict = {}

    # for each file in the path above ending .ann ...
    for file in all_files:
        annotated_note_entity_id, attribute_dict = read_ann(file)
        file_id = os.path.basename(file).split('.')[0]
        ann_dict[file_id] = [annotated_note_entity_id, attribute_dict]

    return ann_dict


def read_all_predicted_raw():
    # predicted data from final annotation meeting with true labels
    path = r'path/to/path'

    all_files = glob.glob(path + "*.txt")

    results = defaultdict(list)

    # for each file in the path above ending .txt ...
    for file in all_files:
        with open(file, "r") as file_open:

            if os.path.basename(file).split('.')[0] in results["filename"]:
                continue
            else:
                results["filename"].append(os.path.basename(file).split('.')[0])
                results["text"].append(file_open.read())

    raw_files = pd.DataFrame(results)

    return raw_files


def read_all_synthetic_ann():
    """For comparing use of only synthetic data to use of extended data set"""
    # synthetic training data
    path = r'path/to/path/'

    all_files = glob.glob(path + "*.ann")

    # empty dict to hold all annotated files
    ann_dict = {}

    # for each file in the path above ending .ann ...
    for file in all_files:
        annotated_note_entity_id, attribute_dict = read_ann(file)
        file_id = os.path.basename(file).split('.')[0]
        ann_dict[file_id] = [annotated_note_entity_id, attribute_dict]

    return ann_dict


def read_all_synthetic_raw():
    """For comparing use of only synthetic data to use of extended data set"""
    # synthetic training data
    path = r'path/to/path'
    
    all_files = glob.glob(path + "*.txt")

    results = defaultdict(list)

    # for each file in the path above ending .txt ...
    for file in all_files:
        with open(file, "r") as file_open:

            if os.path.basename(file).split('.')[0] in results["filename"]:
                continue
            else:
                results["filename"].append(os.path.basename(file).split('.')[0])
                results["text"].append(file_open.read())

    raw_files = pd.DataFrame(results)

    return raw_files