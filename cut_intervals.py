import os
import csv
import wave
from statistics import mean
import librosa
import numpy as np

# this code includes:
# predict_vowel_timestamps_per_word: meant to predict the timestamps of a vowel once some have already been predicted by hand
# convert_timestamps_to_frames: takes the timsetamps for a given file and converts them to frames for both mfcc and wav2vec, and gives them a new unique filename (in case there are two vowels in the same file).
# filter_data_to_mapping: filters the timestamp data to only include rows where the word and vowel match a vowel and word from the training data
# add_word_column: adds a column to the csv with the word, extracted from the filename

def predict_vowel_timestamps_per_word(original_csv_path: str, wav_dir_path: str, word_vowel_csv_path: str, output_csv_path: str):

    # Load word-to-vowel mappings
    word_vowel_map = {}
    with open(word_vowel_csv_path, 'r') as word_vowel_file:
        reader = csv.reader(word_vowel_file)
        next(reader)
        for row in reader:
            word_vowel_map[row[0]] = row[1]

    # Load original CSV
    original_data = []
    with open(original_csv_path, 'r') as original_csv_file:
        reader = csv.reader(original_csv_file)
        next(reader)
        for row in reader:
            original_data.append(row)

    # Calculate average onset and offset percentages for each word
    word_onset_offsets = {}
    for row in original_data:
        filename, onset, offset = row[0], row[1], row[2]
        word = filename.split('_')[1].replace('.wav', '')

        # Skip if onset or offset is empty
        if onset and offset:
            wav_path = os.path.join(wav_dir_path, filename)
            with wave.open(wav_path, 'r') as wav_file:
                duration = wav_file.getnframes() / wav_file.getframerate()
                onset_percent = float(onset) / duration
                offset_percent = float(offset) / duration

            if word not in word_onset_offsets:
                word_onset_offsets[word] = {'onsets': [], 'offsets': []}

            word_onset_offsets[word]['onsets'].append(onset_percent)
            word_onset_offsets[word]['offsets'].append(offset_percent)

    # Compute averages
    word_averages = {}
    for word, times in word_onset_offsets.items():
        avg_onset = mean(times['onsets'])
        avg_offset = mean(times['offsets'])
        word_averages[word] = (avg_onset, avg_offset)

    # Fill missing values in the original CSV
    updated_data = []
    for row in original_data:
        filename, onset, offset = row[0], row[1], row[2]
        word = filename.split('_')[1].replace('.wav', '')

        if not onset or not offset:
            if word in word_averages:
                wav_path = os.path.join(wav_dir_path, filename)
                with wave.open(wav_path, 'r') as wav_file:
                    duration = wav_file.getnframes() / wav_file.getframerate()

                avg_onset, avg_offset = word_averages[word]
                if not onset:
                    onset = round(avg_onset * duration, 3)
                if not offset:
                    offset = round(avg_offset * duration, 3)

        updated_data.append([filename, onset, offset])

    # Write updated data to a new CSV
    with open(output_csv_path, 'w', newline='') as output_csv_file:
        writer = csv.writer(output_csv_file)
        writer.writerows(updated_data)

    print("done predicting timestamps.\n")


def convert_timestamps_to_frames(input_csv, output_csv, dataset):
    """
    Process a CSV with timestamps and convert them to frame numbers for both MFCC and wav2vec representations.

        input_csv (str): Path to the input CSV file containing filename, vowel, onset, and offset timestamps.
        output_csv (str): Path to the output CSV file.
    """
    sampling_rate=16000
    mfcc_stride=0.01
    wav2vec_stride=0.02
    mfcc_frames = 5
    wav2vec_frames = 3

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Open the input and output CSV files
    with open(input_csv, mode='r', encoding='utf-8') as infile, open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)

        # Write header
        if dataset == "reference":
            writer.writerow(["unique_filename", "original_filename", "vowel", 
                             "timestamp_onset", "timestamp_offset", 
                             "frame_number_onset_mfcc", "frame_number_offset_mfcc", 
                             "frame_number_onset_wav2vec", "frame_number_offset_wav2vec"])
        else:
            writer.writerow(["unique_filename", "original_filename",
                             "timestamp_onset", "timestamp_offset",
                             "frame_number_onset_mfcc", "frame_number_offset_mfcc",
                             "frame_number_onset_wav2vec", "frame_number_offset_wav2vec"])

        for wav_id, row in enumerate(reader, start=1):
            wav_filename = row["filename"]
            if dataset=="reference":
                vowel = row["vowel"]
            onset = float(row["onset"])
            offset = float(row["offset"])

            # Generate unique filename for the output
            unique_filename = f"{os.path.splitext(wav_filename)[0]}_{wav_id}"

            # Convert timestamps to frame numbers for both representations
            onset_frame_mfcc = int(onset / mfcc_stride)
            onset_frame_wav2vec = int(onset / wav2vec_stride)
            
            if dataset == "reference":
                offset_frame_mfcc = int(offset / mfcc_stride)
                offset_frame_wav2vec = ont(offset / wav2vec_stride)
            else:
                offset_frame_mfcc = onset_frame_mfcc + mfcc_frames
                offset_frame_wav2vec = onset_frame_wav2vec + wav2vec_frames

            
            if offset_frame_wav2vec == onset_frame_wav2vec or offset_frame_mfcc == onset_frame_mfcc:
                print(f"Notice: {unique_filename} has zero frames.")

            # Write to CSV
            if dataset == "reference":
                writer.writerow([
                    unique_filename,
                    wav_filename,
                    vowel,
                    onset,
                    offset,
                    onset_frame_mfcc,
                    offset_frame_mfcc,
                    onset_frame_wav2vec,
                    offset_frame_wav2vec
                ])
            else:
                writer.writerow([
                    unique_filename,
                    wav_filename,
                    onset,
                    offset,
                    onset_frame_mfcc,
                    offset_frame_mfcc,
                    onset_frame_wav2vec,
                    offset_frame_wav2vec
                ])
        print("done converting timestamps to frames")

def filter_data_to_mapping(data_csv, mapping_csv, output_csv):
    """
    Filter rows in the data CSV where both the word and vowel columns are present in the mapping CSV.
    """
    # Load the mapping into a set for quick lookup
    valid_pairs = set()
    with open(mapping_csv, mode='r', encoding='utf-8') as mapping_file:
        mapping_reader = csv.DictReader(mapping_file)
        for row in mapping_reader:
            valid_pairs.add((row['word'], row['vowel']))

    # Filter the data CSV
    with open(data_csv, mode='r', encoding='utf-8') as data_file, open(output_csv, mode='w', newline='', encoding='utf-8') as output_file:
        data_reader = csv.DictReader(data_file)
        fieldnames = data_reader.fieldnames
        if not fieldnames:
            raise ValueError("The input data CSV is missing headers.")

        data_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        data_writer.writeheader()

        for row in data_reader:
            word = row.get('word')
            vowel = row.get('vowel')
            if (word, vowel) in valid_pairs:
                data_writer.writerow(row)

def add_word_column(input_csv, output_csv):
    """
    Add a new column 'word' to the CSV by extracting the second field from the 'filename' column, 
    where fields are separated by underscores ('_').
    """
    with open(input_csv, mode='r', encoding='utf-8') as infile, open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile, delimiter='\t')
        fieldnames = reader.fieldnames + ['word']  # Add the new column

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        print(fieldnames)

        for row in reader:
            filename = row.get('filename')
            fields = filename.split('_')
            print(filename, fields)
            row['word'] = fields[1]
            writer.writerow(row)


def cut_representations(input_csv, mfcc_input_dir, wav2vec_input_dir, mfcc_output_dir, wav2vec_output_dir):
    """
    Cut representations based on frame ranges specified in a CSV file.
    """
    os.makedirs(mfcc_output_dir, exist_ok=True)
    os.makedirs(wav2vec_output_dir, exist_ok=True)

    with open(input_csv, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)

        count = 0
        for row in reader:
            unique_filename = row['unique_filename']
            original_filename = row['original_filename']
            onset_mfcc = int(row['frame_number_onset_mfcc'])
            offset_mfcc = int(row['frame_number_offset_mfcc'])
            onset_wav2vec = int(row['frame_number_onset_wav2vec'])
            offset_wav2vec = int(row['frame_number_offset_wav2vec'])

            # Load MFCC representation and cut
            mfcc_path = os.path.join(mfcc_input_dir, f"{original_filename}.npy")
            if os.path.exists(mfcc_path):
                mfcc_data = np.load(mfcc_path)
                mfcc_cut = mfcc_data[onset_mfcc:offset_mfcc]
                output_mfcc_path = os.path.join(mfcc_output_dir, f"{unique_filename}.npy")
                np.save(output_mfcc_path, mfcc_cut)
            else:
                print(f"Warning: MFCC file {mfcc_path} does not exist. Skipping.")

            # Load wav2vec representation and cut
            wav2vec_path = os.path.join(wav2vec_input_dir, f"{original_filename}.npy")
            if os.path.exists(wav2vec_path):
                wav2vec_data = np.load(wav2vec_path)
                wav2vec_cut = wav2vec_data[onset_wav2vec:offset_wav2vec]
                output_wav2vec_path = os.path.join(wav2vec_output_dir, f"{unique_filename}.npy")
                np.save(output_wav2vec_path, wav2vec_cut)
            else:
                print(f"Warning: wav2vec file {wav2vec_path} does not exist. Skipping.")
            
            count += 1
            if count % 500 == 0:
                print(count)

    print("done cutting representations.")

def generate_vowel_csv(wav_dir, output_csv):
    """
    Generate a CSV file with onset and offset times for the middle vowel segment in each audio file.

    Parameters:
        wav_dir (str): Path to the directory containing .wav files.
        output_csv (str): Path to the output CSV file.
    """
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]

    if not wav_files:
        print("No .wav files found in the directory.")
        return

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'onset', 'offset'])
        
        count = 0
        for wav_file in wav_files:
            wav_path = os.path.join(wav_dir, wav_file)

            # Strip .wav extension
            file_base_name = os.path.splitext(wav_file)[0]

            # Use wave module to determine audio duration
            with wave.open(wav_path, 'r') as sound_file:
                frame_rate = sound_file.getframerate()
                n_frames = sound_file.getnframes()
                duration = n_frames / frame_rate

            # Calculate onset and offset for the middle vowel segment (middle 5 hundredths of a second)
            segment_duration = 0.05  # 5 hundredths of a second
            onset_time = round(max(0, (duration - segment_duration) / 2), 2)
            offset_time = round(min(duration, onset_time + segment_duration), 2)

            if duration < segment_duration:
                print(f"Warning: {wav_file} has less than 0.05 seconds available.", duration, onset_time, offset_time)
            count += 1
            if count % 100 == 0:
                print(count)

            # Write to CSV
            writer.writerow([file_base_name, onset_time, offset_time])
    print("done getting timestamps.")

if __name__ == "__main__":
    cut_representations('/home/paulie/scratch/categorical_analysis/docs/all_framestamps_test.csv', '/home/paulie/scratch/categorical_analysis/model_reps/test/mfcc/', '/home/paulie/scratch/categorical_analysis/model_reps/test/wav2vec', '/home/paulie/scratch/categorical_analysis/model_reps/test/cut_mfcc/', '/home/paulie/scratch/categorical_analysis/model_reps/test/cut_wav2vec/')
