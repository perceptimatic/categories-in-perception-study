import numpy as np
import argparse
import pandas as pd
import csv
import math
import os
from pathlib import Path


WORD_MAPPING = {
    'cake': 'eɪ', #dipthong
    'dodge': 'ɑ', #'ɑ~ɒ'
    'dress': 'ɛ',
    'face': 'eɪ', #dipthong
    'fleece': 'i',
    'foot': 'ʊ',
    'goat': 'oʊ', #dipthong
    'goose': 'u',
    'kit': 'ɪ',
    'look': 'ʊ',
    'loop': 'u',
    'lot': 'ɔ',
    'stop': 'ɔ',
    'strut': 'ʌ',
    'tape': 'eɪ', #dipthong
    'trap': 'æ'
}

AI_MAPPING = {
    'xq': 'æ',
    'xa': 'ʌ',
    #'xc': 'tʃ',
    #'xd': 'ð',
    'xe': 'ɛ',
    #'xg': 'ŋ',
    'xi': 'ɪ',
    #'xj': 'dʒ',
    'xo': 'ɔɪ', #dipthong
    'xr': 'ɹ', #syllabified consonant
    #'xs': 'ʃ',
    #'xt': 'θ',
    'xu': 'ʊ',
    'xw': 'aʊ', #dipthong
    'xy': 'aɪ', #dipthong
    #'xz': 'ʒ',
    #'b': 'b',
    'a': 'ɑ', #'ɑ~ɒ'
    'c': 'ɔ',
    #'d': 'd',
    'e': 'eɪ', #dipthong - keep
    #'f': 'f',
    #'g': 'g',
    #'h': 'h',
    'i': 'i',
    #'k': 'k',
    #'l': 'l',
    #'m': 'm',
    #'n': 'n',
    'o': 'oʊ', #dipthong - keep
    #'p': 'p',
    #'r': 'ɹ',
    #'s': 's',
    #'t': 't',
    'u': 'u',
    #'v': 'v',
    #'w': 'w',
    #'y': 'j',
    #'z': 'z'
}

WV_MAPPING = {
    'a': 'a',
    'a:': 'aː',
    'aː': 'aː',
    'e': 'e',
    'eɪ': 'eɪ',
    'ẽ': 'ẽ',
    'i': 'i',
    'i:': 'iː',
    'in': 'in',
    'iː': 'iː',
    'ĩ': 'ĩ',
    'o': 'o',
    'oʊ': 'oʊ',
    'õ': 'õ',
    'u': 'u',
    'un': 'un',
    'uː': 'uː',
    'ũ': 'ũ',
    'y': 'y',
    'y:': 'yː',
    'yː': 'yː',
    'æ': 'æ', 
    'æ:': 'æː', 
    'ø': 'ø',
    'ø:': 'øː',
    'øː': 'øː',
    'œ': 'œ',
    'œ̃': 'œ̃',
    'ɐ̃': 'ɐ̃',
    'ɑ': 'ɑ',
    'ɑ̃': 'ɑ̃',
    'ɔ': 'ɔ',
    'ɔ̃': 'ɔ̃',
    'ɛ': 'ɛ',
    'ɛ̃': 'ɛ̃',
    'ɤ:': 'ɤː', 
    'ɪ': 'ɪ',
    'ɯ': 'ɯ',
    'ʊ': 'ʊ',
    'ʌ': 'ʌ',
    'ʏ': 'ʏ'
}

LANG_CODES = {
    'GL': '(de)',
    'FR': '(fr)',
    'EN': '(en-us)',
    'TU': '(tr)',
    'GR': '(de-mun)',
    'ES': '(et)',
    'BR': '(pt-br)'
}

FRENCH_WANTED_VOWELS = ('a', 'ɑ̃', 'e', 'ɛ', 'ɛ̃', 'i', 'o', 'ø', 'œ', 'ɔ', 'ɔ̃', 'u', 'y')
WANTED_VOWELS = ('ʌ','ɛ','ɪ','ʊ','ɑ','eɪ','i','oʊ','u','æ')
#WANTED_VOWELS = ('ʌ','ɛ','ɪ','ʊ','ɑ~ɒ','ɔ','eɪ','i','oʊ','u','æ')
WANTED_VOWELS_TEST = ('a', 'aː', 'e', 'eɪ', 'ẽ', 'i', 'iː', 'ĩ', 'o', 'oʊ', 'õ', 'u', 'uː', 'ũ', 
                        'y', 'yː', 'æ', 'æː', 'ø', 'øː', 'œ', 'ɐ̃', 'ɑ', 'ɑ̃', 'ɔ', 'ɔ̃', 'ɛ', 
                        'ɛ̃', 'ɤː', 'ɪ', 'ɯ', 'ʊ', 'ʌ', 'ʏ')
TEST_LABELS_SORTED = ['a (de)', 'a (fr)', 'a (pt-br)', 'aː (de-mun)', 'aː (de)',
    'aː (et)', 'æ (en-us)', 'æː (et)', 'ɐ̃ (pt-br)', 'ɑ (en-us)', 'ɑ̃ (fr)', 'e (de)',
    'e (fr)', 'e (pt-br)', 'ẽ (pt-br)', 'eɪ (en-us)', 'ɛ (de)', 'ɛ (en-us)', 'ɛ (fr)',
    'ɛ̃ (fr)', 'ɛ (pt-br)', 'ɤː (et)', 'i (de)', 'i (en-us)', 'i (fr)', 'i (pt-br)',
    'ĩ (pt-br)', 'i (tr)', 'iː (de-mun)', 'iː (et)', 'ɪ (de-mun)', 'ɪ (en-us)',
    'o (de)', 'o (fr)', 'ø (fr)', 'o (pt-br)', 'õ (pt-br)', 'øː (de-mun)', 'øː (et)',
    'œ (fr)', 'œ (tr)', 'oʊ (en-us)', 'ɔ (de)', 'ɔ (fr)', 'ɔ̃ (fr)', 'ɔ (pt-br)',
    'u (en-us)', 'u (fr)', 'u (pt-br)', 'ũ (pt-br)', 'u (tr)', 'uː (de-mun)', 'ɯ (tr)',
    'ʊ (de-mun)', 'ʊ (en-us)', 'ʌ (en-us)', 'y (fr)', 'y (tr)', 'yː (de-mun)', 'yː (et)',
    'ʏ (de-mun)']

def probability(train_labels, point, klabels, k):
    num_labels = len(klabels)
    probabilities = {}
    # add a key for every possible label
    for label in train_labels: 
        probabilities[label] = 0.0
    # assign the actual ratios for each label
    for label in klabels:
        probabilities[label] = klabels[label]/k
    return (point, probabilities)


def label_just_vowel(just_sound_name):
    wait = False
    for char in just_sound_name:
        if wait:
            wait = False
            if 'x'+char in AI_MAPPING:
                return AI_MAPPING['x'+char]
        else:
            if char in AI_MAPPING:
                return AI_MAPPING[char]
            elif char == 'x':
                wait = True

def average_probability(test_labels, compiled_probabilities):
    average_probabilities = {}
    count = {}

    for label in test_labels: 
        count[label] = 0
        
    for item in compiled_probabilities:
        if item[0] not in average_probabilities:
            average_probabilities[item[0]] = item[1]
        else:
            for label_i, label in enumerate(item[1]):
                average_probabilities[item[0]][label] += item[1][label]
                #[(xa, {a: 0.6, xa: 0.4, i: 0.0}),
                #(xa, {a: 0.2, xa: 0.4, i: 0.4})]
        count[item[0]] += 1
    for vowel in average_probabilities:
        for key in average_probabilities[vowel]:
            average_probabilities[vowel][key] /= count[vowel]
    return average_probabilities


def load_training_data(train_dir, language, stimuli_csv):
    print('loading data for training set')
    # DATA LOADING
    vowels_data = []
    # load the data from the vowels from words (strut, cake, etc .)
    for filename in (train_dir.glob("**/*.npy")): 
        array = np.load(filename)
        # Compute the average of each column across all frames
        average_array = np.mean(array, axis=0)
        vowels_data.append(np.array(average_array))

    # LABEL LOADING
    vowels_labels = []
    count = 0

    if language == 'eng':
        # add the labels of the vowels from words (strut, cake, etc.)
        for filename in cut_model_dir.glob("**/*.npy"):
            file_no_path = Path(filename).name
            vowels_labels.append(WORD_MAPPING[file_no_path[file_no_path.find('_')+1:file_no_path.find('.')]]) 
            count += 1
            if (count % 500 == 0):
                print(count)

        # add the labels of the vowels from the ai corpus
        for filename in ai_cut_model_dir.glob("**/*.npy"):
            file_no_path = Path(filename).name
            just_sound_name = file_no_path[file_no_path.find('_', -9)+1:file_no_path.find('.')]
            file_no_sound_name = file_no_path[0:file_no_path.find('_',-9)+1]

            vowels_labels.append(label_just_vowel(just_sound_name))
            count += 1
            if (count % 500 == 0):
                print(count)

        wanted_vowels = WANTED_VOWELS
    
    if language == 'fre':
        csvfile = pd.read_csv(stimuli_csv, index_col = "unique_filename")
        for filename in train_dir.glob("**/*.npy"):
            file_no_path = Path(filename).name.removesuffix('.npy')
            vowels_labels.append(csvfile.loc[file_no_path, "vowel"])
            count += 1
            if (count % 500 == 0):
                print(count)
        wanted_vowels = FRENCH_WANTED_VOWELS
    
    # filter out unwanted vowels
    filtered_vowels_data = []
    filtered_vowels_labels = []
    for index, label in enumerate(vowels_labels):
        for v in wanted_vowels:
                if v == label:
                    filtered_vowels_labels.append(label)
                    filtered_vowels_data.append(vowels_data[index])

    print('done loading training data')
    distribution = {}
    for vowel in wanted_vowels:
        distribution[vowel] = filtered_vowels_labels.count(vowel)
    print("vowel distribution:", distribution)

    # create dataset
    X, y = filtered_vowels_data, filtered_vowels_labels

    return X, y # these will be X_train, y_train for the rest of the study

def kNN_regression(X_train, X_test, y_train, y_test, k, image_name, stimulus_ABX_code_test, class_vec_wv_dir):
    train_labels = set(y_train)
    test_labels = set(y_test)
    compiled_probabilities = []
    # make my own nearest neighbours:
    '''Calculate and store the distance between each test point and every training point as tuples, then sort,
    then take the top k values, then find the majority label among the k points. save to a dictionary.
    '''
    # end result
    labeling_output = []
    percents = []
    count = 0
    pacer = 0
    # for each test item, calculate distances between each training item and store the label
    for test_index in range(len(X_test)):
        distances = []
        for train_index in range(len(X_train)):
            distances.append((np.linalg.norm(X_train[train_index] - X_test[test_index]), y_train[train_index]))
        sorted_distances = sorted(distances)
        # find the plurality among the closest k neighbours
        klabels = {}
        for i in range(k):
            if sorted_distances[i][1] not in klabels:
                klabels[sorted_distances[i][1]] = 0
            klabels[sorted_distances[i][1]] += 1
        # show probabilities:
        percents.append(probability(train_labels, X_test[test_index], klabels, k))
        compiled_probabilities.append((y_test[test_index], percents[-1][-1])) # append what the label actually is, and what we thought it was.

        # store the test point and label together
        labeling_output.append((X_test[test_index], max(klabels, key=klabels.get)))
        # self test
        # print(y_test[test_index] == max(klabels, key=klabels.get), y_test[test_index], max(klabels, key=klabels.get))
        pacer += 1
        if pacer % 500 == 0:
            print(pacer)
        if y_test[test_index] == max(klabels, key=klabels.get):
            count += 1

    average_probabilities = average_probability(test_labels, compiled_probabilities)
    # some_file = heatmap(train_labels, test_labels, average_probabilities, image_name)
    print('finished kNN regression')

    print('exporting classification vectors as .npy files')
    
    for vector_i, vector in enumerate(compiled_probabilities):
        vector_as_array = []
        for label in train_labels:
            vector_as_array.append(vector[1].get(label, 0))  # Safely get value for label
        
        # Build the output file path safely
        output_file = os.path.join(
            class_vec_wv_dir,
            stimulus_ABX_code_test[vector_i].removesuffix('.wav') + '.npy'
        )
        
        # Save the vector array
        np.save(output_file, vector_as_array)
    
    return average_probabilities, compiled_probabilities


def load_testing_data(cut_wv_model_dir, stimuli_csv):
    print('loading data for test set')
    # DATA LOADING
    X_test = []
    y_test = []
    stimulus_ABX_code_test = []
    languages = []
    count = 0

    csvfile = pd.read_csv(stimuli_csv, index_col='index')    
    for filename in cut_wv_model_dir.glob("**/*.npy"):
        basename = Path(filename).name.removesuffix('.npy')
        if len(basename.split('_')) == 3:
            stimulus_code = basename.split('_')[0] + '_' + basename.split('_')[1]
        else:
            stimulus_code = basename.split('_')[0]
        array = np.load(filename)
        # Compute the average of each column across all frames
        average_array = np.mean(array, axis=0)   
        phone = WV_MAPPING[csvfile.loc[stimulus_code, '#phone']]
        lang = LANG_CODES[csvfile.loc[stimulus_code, 'language']]
        
        # filter data
        if phone in WANTED_VOWELS_TEST:
            X_test.append(average_array)
            y_test.append(phone + ' ' + lang) #label loading
            stimulus_ABX_code_test.append(csvfile.loc[stimulus_code, '#file_extract'].removesuffix('.wav'))
        
        count += 1
        if count % 500 == 0:
            print(count)
    print('done loading data for test set')

    return X_test, y_test, stimulus_ABX_code_test


def row_distance(row, cache, train_labels, class_vec_or_X_test_model, stimulus_ABX_code_test):
    TGT_filename = stimulus_ABX_code_test.index(row['TGT_item'])
    OTH_filename = stimulus_ABX_code_test.index(row['OTH_item'])
    X_filename = stimulus_ABX_code_test.index(row['X_item'])
    
    TGT_item = class_vec_or_X_test_model[TGT_filename]
    OTH_item = class_vec_or_X_test_model[OTH_filename]
    X_item = class_vec_or_X_test_model[X_filename]

    #calculate deltas and place winner in new column
    # calc TGT:
    if (TGT_filename, X_filename) not in cache:
        if train_labels:
            X_rep = []
            TGT_rep = []
            for label in train_labels:
                X_rep.append(X_item[1][label])
                TGT_rep.append(TGT_item[1][label])
        else:
            X_rep = X_item
            TGT_rep = TGT_item
        
        cache[(TGT_filename, X_filename)] = distance(np.array(TGT_rep), np.array(X_rep))
    # calc OTH:
    if (OTH_filename, X_filename) not in cache:
        if train_labels:
            X_rep = []
            OTH_rep = []
            for label in train_labels:
                X_rep.append(X_item[1][label])
                OTH_rep.append(OTH_item[1][label])
        else:
            X_rep = X_item
            OTH_rep = OTH_item

        cache[(OTH_filename, X_filename)] = distance(np.array(OTH_rep), np.array(X_rep))

    return cache[(OTH_filename, X_filename)] - cache[(TGT_filename, X_filename)]

def distance(AB_item, X_item):
    num = 0
    denom1 = 0
    denom2 = 0
    
    for i in range(AB_item.shape[0]):
        num += (X_item[i]*AB_item[i])
        denom1 += AB_item[i]**2
        denom2 += X_item[i]**2

    return 1 - ( num / ( math.sqrt(denom1)*math.sqrt(denom2) ) )
        

def ABX_task(train_labels, class_vec_or_X_test_model, stimulus_ABX_code_test, model, infile, outfile):
    exp_csvfile = pd.read_csv(infile)
    
    cache = {}
    exp_csvfile[model+'delta'] = exp_csvfile.apply(row_distance, axis=1, args=(cache, train_labels, class_vec_or_X_test_model, stimulus_ABX_code_test))
        
    exp_csvfile.to_csv(outfile)


def export_identification_data(average_classification_vectors, train_labels, test_labels, doc_directory, model):
    with open(doc_directory + model + "class_vecs_avgs.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
    
        # Write header
        writer.writerow(['Phone (Language)', 'Response', 'value'])
    
        # Iterate through the outer and inner dictionaries
        for test_label in TEST_LABELS_SORTED:
            for train_label in FRENCH_WANTED_VOWELS:
                value = average_classification_vectors[test_label][train_label]
                writer.writerow([test_label, train_label, value])       


def load_classification_vectors(classification_vector_dir):
    classification_vectors = {}
    for filename in (classification_vector_dir.glob("**/*.npy")):
        array = np.load(filename)
        stimuli_code = filename.name.removesuffix('.npy')
        classification_vectors[stimuli_code] = array

    return classification_vectors

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate deltas for the model")
    
    parser.add_argument("-c", "--compute", type=bool, nargs="+", required=True, help="True if computing classification vectors is needed, False if already computed")
    parser.add_argument("-m", "--model", type=str, nargs="+", required=True, help="The model of the representation files for the experiment: mfcc, deepspeech, wav2vec")
    parser.add_argument("--train_dir", type=str, nargs="+", required=True)
    parser.add_argument("--test_dir", type=str, nargs="+", required=True)
    parser.add_argument("--language", type=str, nargs="+", required=True)
    parser.add_argument("--cvec_dir", type=str, nargs="+", required=True)
    parser.add_argument("--stim_data_test", type=str, nargs="+", required=True)
    parser.add_argument("--stim_data_train", type=str, nargs="+", required=True)
    parser.add_argument("--exp_data", type=str, nargs="+", required=True)
    parser.add_argument("--outfile", type=str, nargs="+", required=True)
    parser.add_argument("--doc_directory", type=str, nargs="+", required=True)

    commandline_input = parser.parse_args()

    compute = commandline_input.compute[0]
    model = commandline_input.model[0]
    train_dir = Path(commandline_input.train_dir[0])
    test_dir = Path(commandline_input.test_dir[0])
    language = commandline_input.language[0]
    classification_vector_dir = Path(commandline_input.cvec_dir[0])
    stimuli_data_test = Path(commandline_input.stim_data_test[0])
    stimuli_data_train = Path(commandline_input.stim_data_train[0])
    experimental_data = Path(commandline_input.exp_data[0])
    outfile = Path(commandline_input.outfile[0])
    doc_directory = commandline_input.doc_directory[0]
    
    print("starting identification experiment...")
    X_test, y_test, stimulus_ABX_code_test = load_testing_data(test_dir, stimuli_data_test)
    if compute:
        print(f"creating classification vectors for {model} data...") 
        X_train, y_train = load_training_data(train_dir, language, stimuli_data_train)
        average_classification_vectors, classification_vectors = kNN_regression(X_train, X_test, y_train, y_test, int(math.sqrt(len(y_train))), model, stimulus_ABX_code_test, classification_vector_dir)
        print(f"done creating classification vectors for {model} data.")
    else:
        print(f"loading classification vectors from {classification_vector_dir}...")
        classification_vectors = load_classification_vectors(classification_vector_dir)
        print(f"done loading classification vectors from {classification_vector_dir}.")

    print("exporting identification results...")
    export_identification_data(average_classification_vectors, set(y_train), set(y_test), doc_directory, model)
    print("done exporting identification results.")

    print("starting kNN experiment for discrimination data")
    ABX_task([], X_test, stimulus_ABX_code_test, model+"_raw_", experimental_data, Path("temp"))
    ABX_task(FRENCH_WANTED_VOWELS, classification_vectors, stimulus_ABX_code_test, model+"_", Path("temp"), outfile)
    print("experiment complete.")

    # example usage:
    # python general_NN_vowels.py -c True -m wav2vec --train_dir /home/paulie/scratch/categorical_analysis/model_reps/reference/cut_wav2vec/ --test_dir /home/paulie/scratch/categorical_analysis/model_reps/test/cut_wav2vec/ --language fre --cvec_dir /home/paulie/scratch/categorical_analysis/classification_vectors/wav2vec/ --stim_data_test /home/paulie/scratch/categorical_analysis/docs/WorldVowels_stimuli\ \(1\).csv --stim_data_train /home/paulie/scratch/categorical_analysis/docs/all_timestamps_reference.csv --exp_data /home/paulie/scratch/categorical_analysis/docs/all_discrimination.csv --outfile /home/paulie/scratch/categorical_analysis/docs/wav2vec_identification.csv --doc_directory /home/paulie/scratch/categorical_analysis/docs/

    # python general_NN_vowels.py -c True -m mfcc --train_dir /home/paulie/scratch/categorical_analysis/model_reps/reference/cut_mfcc/ --test_dir /home/paulie/scratch/categorical_analysis/model_reps/test/cut_mfcc/ --language fre --cvec_dir /home/paulie/scratch/categorical_analysis/classification_vectors/mfcc/ --stim_data_test /home/paulie/scratch/categorical_analysis/docs/WorldVowels_stimuli\ \(1\).csv --stim_data_train /home/paulie/scratch/categorical_analysis/docs/all_timestamps_reference.csv --exp_data /home/paulie/scratch/categorical_analysis/docs/all_discrimination.csv --outfile /home/paulie/scratch/categorical_analysis/docs/mfcc_identification.csv --doc_directory /home/paulie/scratch/categorical_analysis/docs/
