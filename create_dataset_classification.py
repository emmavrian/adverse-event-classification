from read_data import read_all_raw, read_all_ann, read_all_real, read_all_predicted_ann, read_all_predicted_raw, read_all_synthetic_ann, read_all_synthetic_raw
import pandas as pd
import itertools


def combine_target_with_text(predicted_data=False, only_synthetic=False):

    if(predicted_data):
        ids_and_target_dict = get_target_labels(predicted_data=True)
        raw_data_df = read_all_predicted_raw()
    elif(only_synthetic):
        ids_and_target_dict = get_target_labels(only_synthetic=True)
        raw_data_df = read_all_synthetic_raw()
    else:
        ids_and_target_dict = get_target_labels(predicted_data=False, only_synthetic=False)
        raw_data_df = read_all_raw()

    for index, row in raw_data_df.iterrows():
        raw_filename = row['filename']
        raw_text = row['text']

        # remove problematic unicode space breaks
        raw_text = raw_text.replace(u'\xa0', ' ')

        # add correct text to existing dictionary
        ids_and_target_dict[raw_filename]['text'] = raw_text
    
    final_data = pd.DataFrame.from_dict(ids_and_target_dict, orient='index')
    final_data.index.name = 'note_id'

    return final_data




def get_target_labels(predicted_data=False, only_synthetic=False):

    if(predicted_data):
        ann_dict = read_all_predicted_ann()
    elif(only_synthetic):
        ann_dict = read_all_synthetic_ann()
    else:
        ann_dict = read_all_ann()

    files_with_target_values = {}

    # only these for now. Might add some others later
    all_relevant_target_labels = ["Er_notatet_venekateterrelatert",
                                  "Er_notatet_PVK_relatert",
                                  "Er_notatet_flebittrelatert",
                                  "Er_notatet_infeksjonsrelatert",
                                  "Er_notatet_sepsisrelatert",]
    
    #venous_catheter_related = ["Er_notatet_venekateterrelatert", "Er_notatet_PVK_relatert"]
    #infection_related = ["Er_notatet_flebittrelatert", "Er_notatet_infeksjonsrelatert", "Er_notatet_sepsisrelatert"]

    # if note contains at least one of these positive values - place in venous catheter category
    positive_venous_catheter = ["Venekateter", "Sannsynligvis_venekateter", "PVK", "Sannsynligvis_PVK"]
    #negative_venous_catheter = ["Ikke_venekateter", "Ikke_PVK"]

    # if note contains at least one of these positive values - place in infection category
    positive_infection = ["Flebitt", "Sannsynligvis_flebitt", "Infeksjon", "Sannsynligvis_infeksjon", "Sepsis", "Sannsynligvis_sepsis"]
    #negative_infection = ["Ikke_flebitt", "Ikke_infeksjon", "Ikke_sepsis"]

    for file, ann in ann_dict.items():
        # each file has an ann with this format: [target_id, {'A1': Attribute(id='temp', type='temp', target='temp', values=('Ikke_PVK')), 'A2' Attribute(...)}] ...

        annotated_note_entity_id = ann[0]
        ann_attributes = ann[1]

        target_values = []

        for value in ann_attributes.values():
            
            # skip all attributes that are related to entities other than "Annotert_Notat"
            if value.target != annotated_note_entity_id:
                continue
            
            # find all target values for the note
            if value.type in all_relevant_target_labels:
                #append the relevant target values. Values are saved as a tuple but each tuple can only have one value, so value.values[0] will get it
                target_values.append(value.values[0])

        # place note into target categories by checking if the target labels overlap with ANY of the list of related target labels
        venous_catheter_related = int(bool(set(positive_venous_catheter) & set(target_values)))
        infection_related = int(bool(set(positive_infection) & set(target_values)))
        
        files_with_target_values[file] = {'venous_catheter':venous_catheter_related, 'infection':infection_related}

    return files_with_target_values


