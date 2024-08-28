from pathlib import Path
from textgrids import TextGrid
import pandas as pd
import os
import subprocess

WANTED_VOWELS = ['a', 'ɔ', 'œ', 'e', 'i', 'o', 'u', 'y', 'ø', 'ɑ̃', 'ɔ̃', 'ɛ', 'ɛ̃']
UNWANTED_CONTEXTS = ['nan', '', 'spn', 'ʁ', 'l', 'a', 'ɔ', 'œ', 'e', 'i', 'o', 'u', 'y', 'ø', 'ɑ̃', 'ɔ̃', 'ɛ', 'ɛ̃']
OE_CONTEXTS = ["b_f", "b_j", "b_s", "b_v", "b_ɟ", "b_ɡ", "d_f", "d_j", "d_m", "d_v", "f_j", "f_n", "f_ʃ", "j_v", "k_j", "k_k", "k_v", "m_b", "m_p", "m_ɡ", "n_b", "n_f", "n_j", "n_s", "n_v", "p_b", "p_p", "p_v", "p_ŋ", "p_ɡ", "p_ʒ", "s_f", "s_j", "s_m", "t_f", "t_j", "t_v", "v_f", "v_j", "v_v", "v_ɡ", "ə_j", "ɡ_f", "ɡ_j", "ʃ_t", "ʒ_f", "ʒ_n", "ʒ_ɲ"]
nasE_CONTEXTS = ["b_f",
"b_s",
"b_v",
"d_f",
"d_m",
"d_v",
"f_n",
"j_v",
"k_k",
"k_v",
"m_b",
"m_p",
"m_ɡ",
"n_b",
"n_f",
"n_s",
"n_v",
"p_b",
"p_p",
"p_v",
"p_ɡ",
"s_f",
"s_m",
"t_f",
"t_j",
"t_v",
"v_f",
"v_v",
"ʃ_t",
"ʒ_f",
"ʒ_n"]


def load_textgrid(textgrid: Path):
    """
    Example .ali file:
    ...
    """
    tg = TextGrid(textgrid)
    tier = tg["phones"]

    onset = [p.xmin for p in tier]
    offset = [p.xmax for p in tier]
    phone = [p.text for p in tier]
    speaker = [textgrid.stem for _ in tier]
    
    cols = {"onset": onset, "offset": offset, "speaker": speaker, "phone": phone}
    df = pd.DataFrame(cols)
    df[["onset", "offset"]] = df[["onset", "offset"]].astype(float)
    df["file"] = ali.stem

    return df

def organize_df(data: str):
    #initialize df
    df = pd.read_csv(data, delimiter='\t')

    # add columns
    phone = df['phone'].tolist()
    onset = df['onset'].tolist()
    offset = df['offset'].tolist()

    if not phone[0]:
        phone[0] = '#'
        previous_phone = ['#', '#'] + phone[1:-1]
        previous_phone_onset = ['#', '#'] + onset[1:-1]
    else:
        previous_phone = ['#'] + phone[:-1]
        previous_phone_onset = ['#'] + onset[:-1]
    
    if not phone[-1]:
        phone[-1] = '#'
        next_phone = phone[1:-1] + ['#', '#']
        next_phone_offset = offset[1:-1] + ['#', '#']
    else:
        next_phone = phone[1:] + ['#']
        next_phone_offset = offset[1:] + ['#']
    
    new_cols = {"previous_phone": previous_phone, 
                "next_phone": next_phone, 
                "previous_phone_onset": previous_phone_onset, 
                "next_phone_offset": next_phone_offset}
    df = df.assign(**new_cols)
    
    # filter rows so its only the rows that have the vowels we want as the phone
    df = df[df['phone'].isin(WANTED_VOWELS)]
    df = df[df['previous_phone'].isin(UNWANTED_CONTEXTS) == False]
    df = df[df['next_phone'].isin(UNWANTED_CONTEXTS) == False]
    
    df['context'] = df['previous_phone'].astype(str) + '_' + df['next_phone'].astype(str)
    df = df[~df['context'].str.contains('nan', case=False, na=False)]

    print(df.columns)

    df.to_csv("filtered_data.tsv", sep='\t', index=False)
    
    return df


def merge_count_tables(filtered_data: str, vowel_table: str, file_to_write: str):
    file1 = pd.read_csv(filtered_data, sep='\t')
    file2 = pd.read_csv(vowel_table, sep='\t')

    # Perform the left merge on context
    merged = pd.merge(file1, file2, on='context', how='inner', suffixes=(None, '_'))

    # Write the result to a new TSV file
    merged.to_csv('comparing_counts.tsv', sep='\t', index=False)


def count_phone_instances_from_not_summarized(filtered_data: str):
    # count the number of instances of a given phone from a file where each instance is separate (not summarized)
    df = pd.read_csv(filtered_data, delimiter='\t')

    return df['phone'].value_counts()


def count_instances_from_summarized(filtered_data: str, column_to_count: str):
    # summarize the total number of instances of a specific column for each vowel (from a summarized or not summarized table)
    df = pd.read_csv(filtered_data, delimiter='\t')
    summary = []
    for phone in WANTED_VOWELS:
        summary.append(df[df['phone'].eq(phone) == True][column_to_count].sum())

    cols = {'phone': WANTED_VOWELS, 'summary_of_column': summary}

    return pd.DataFrame(cols)


def summarize_contexts(filtered_data: str, file_to_write: str):
    # make a table with the counts of each context for every phone listed
    df = pd.read_csv(filtered_data, delimiter='\t')

    summary = df.groupby(['phone', 'context']).size().reset_index(name='count')
    print(summary)

    summary.to_csv(file_to_write, sep='\t', index=False)


def filter_to_chosen_contexts(filtered_data: str, file_to_write: str):
    # filter to only contain the contexts found in certain phones (or any other filtering needed)
    df = pd.read_csv(filtered_data, delimiter='\t')
    
    df = df[df['context'].isin(OE_CONTEXTS) == True]
    df = df[df['context'].isin(nasE_CONTEXTS) == True]
    df = df[~df['context'].eq('j_v')]

    df.to_csv(file_to_write, sep='\t', index=False)

    return df


def make_counts_table_for_one_vowel(filtered_data: str, vowel: str, file_to_write: str):
    # make count table for just one vowel (made for nasE/5 vowel originally) to be able to merge with the overall context summary table
    df = pd.read_csv(filtered_data, delimiter='\t')

    df = df[df['phone'].eq(vowel) == True]
    df = df[['context', 'count']]

    df.to_csv(file_to_write, sep='\t', index=False)

    return df


def what_if_only_specific_contexts(summarized_data: str, vowel_specific_table_counts: str, file_to_write: str):
    # write a new csv that counts how many instances of vowels would be lost if we went with the contexts picked
    df_data = pd.read_csv(summarized_data, delimiter='\t')
    df_counts = pd.read_csv(vowel_specific_table_counts, delimiter='\t')
    
    df_joined = df_data.merge(df_counts, how='inner', on='context', suffixes=(None,'_'))
    df_joined['lost_examples'] = df_joined.apply(determine_occurrence, axis=1)

    df_joined.to_csv(file_to_write, sep='\t', index=False)

def determine_occurrence(row):
    # helper function
    if row['count'] >= row['count_']:
        return 0
    else:
        return row['count_'] - row['count']

def summarize_lost_examples(joined_filtered_data: str):
    # summarize the total number of lost examples overall (not phone or context specific)
    df = pd.read_csv(joined_filtered_data, delimiter='\t')
    
    print("total lost:", df['lost_examples'].sum())


def add_current_number_picked_column(filtered_data: str, file_to_write: str):
    df = pd.read_csv(filtered_data, delimiter='\t')

    df['current_number_picked'] = df.apply(determine_number_picked, axis=1)
    
    df.to_csv(file_to_write, sep='\t', index=False)

def determine_number_picked(row):
    # helper function
    if row['lost_examples'] == 0:
        return row['count_']
    else:
        return row['count']

def add_total_picked_column(filtered_data: str, file_to_write: str):
    df = pd.read_csv(filtered_data, delimiter='\t')

    df['total_number_picked'] = df.apply(sum_number_picked, axis=1)

    df.to_csv(file_to_write, sep='\t', index=False)

def sum_number_picked(row):
    # helper function
    if row['extras'] + row['current_number_picked'] <= row['count']:
        return row['extras'] + row['current_number_picked']
    else:
        return False

def check_only_possible_ones_included(counts_table: str):
    df = pd.read_csv(counts_table, delimiter='\t')

    df.apply(check_each_row_so_only_possible_ones, axis=1)


def check_each_row_so_only_possible_ones(row):
    if row['lost_examples'] > 0:
            if row['total_number_picked'] != row['count']:
                print(row['phone'], row['context'])

def add_yes_to_first_n_of_each_vowel(experimental_data: str, total_picked_counts: str, file_to_write: str):
    
    count_table = pd.read_csv(total_picked_counts, delimiter='\t')
    experimental_table = pd.read_csv(experimental_data, delimiter='\t')
    experimental_table['chosen'] = 'no'

    for vowel in WANTED_VOWELS:
        relevant_vowel_counts = count_table[count_table['phone'] == vowel]
        for context in relevant_vowel_counts['context']:
            count = int(relevant_vowel_counts[relevant_vowel_counts['context'] == context]['total_number_picked'].iloc[0])
            filtered_indices = experimental_table[(experimental_table['phone'] == vowel) &
                                                  (experimental_table['context'] == context)].index[:count]
            experimental_table.loc[filtered_indices, 'chosen'] = 'yes'

    only_yes = experimental_table[experimental_table['chosen'] == 'yes']
    
    only_yes = only_yes.drop(columns=['chosen'])

    only_yes.to_csv(file_to_write, sep='\t', index=False)
    print('size of new table:', only_yes.shape)


def rename_rows(input_file: str, output_file: str):
    df = pd.read_csv(input_file, delimiter='\t')

    # Add a new column 'FileNumber' with numbers from 1 to the number of rows in the DataFrame
    df['FileNumber'] = range(1, len(df) + 1)

    # Write the updated DataFrame to a new TSV file
    df.to_csv(output_file, sep='\t', index=False)

    print(f"Updated file saved as {output_file}")


def cut_wavs(experimental_data: str, wav_directory: str, output_directory: str):
    # cuts/makes the wav files from the picked contexts/phones
    experimental_data_table = pd.read_csv(experimental_data, delimiter='\t')

    os.makedirs(output_directory, exist_ok=True)

    for index, row in experimental_data_table.iterrows():
        # Extract relevant data from each row
        wav_file = str(row['file']) + '.wav'  # Column containing the name of the wav file
        start_time = row['previous_phone_onset']  # Column containing the start interval
        end_time = row['next_phone_offset']  # Column containing the end interval
        phone_onset = row['onset'] - row['previous_phone_onset']
        phone_offset = row['next_phone_offset'] - row['offset']
        output_name = f"{row['FileNumber']}_{row['phone']}_{row['context']}_{phone_onset}_{phone_offset}.wav"

        # Calculate duration from start and end times
        duration = end_time - start_time

        # Construct the full path to the wav file in the wav_directory
        full_wav_path = os.path.join(wav_directory, wav_file)

        # Build the sox command
        sox_command = [
            'sox', full_wav_path, os.path.join(output_directory, output_name),
            'trim', str(start_time), str(duration)
        ]

        # Run the sox command
        subprocess.run(sox_command)

    print(f"Processed {len(experimental_data_table)} files.")
        
        
if __name__ == "__main__":
    # make empty df

    for textgrid_path in directory:
        # make df for each textgrid containing base columns
        df_unformatted = load_textgrid(textgrid_path)
        # add relevant columns and filter to only contain relevant contexts
        df = organize_df(df_unformatted)
        # add df to main df

        # cut wav files and store them in end directory
        cut_wavs(df)
    
    # write main df to csv
    # return csv.
