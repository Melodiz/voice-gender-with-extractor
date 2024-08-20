import glob
import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import logging
import warnings

# Set up logging to save error and warning messages to 'dev/errors_warnings.log'
log_file_path = os.path.join('dev', 'errors_warnings.log')
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(filename=log_file_path, level=logging.WARNING,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Redirect warnings to the logger
def warn_with_log(message, category, filename, lineno, file=None, line=None):
    logging.warning(f'{filename}:{lineno}: {category.__name__}: {message}')

warnings.showwarning = warn_with_log

class AudioFeatureExtractor:
    def __init__(self, audio_folder_path, labels_file_path, limit_rows=None):
        """
        Initialize the AudioFeatureExtractor with paths and optional row limit.
        """
        self.audio_folder_path = audio_folder_path
        self.labels_file_path = labels_file_path
        self.limit_rows = limit_rows
        self.labels_dict = self._load_labels()
        self.features = ["duration", "meanfreq", "sd", "median", "Q25",
                         "Q75", "IQR", "skew", "kurt", "sp_ent", "sfm",
                         "mode", "centroid", "peakf", "meanfun", "minfun",
                         "maxfun", "meandom", "mindom", "maxdom", "dfrange",
                         "modindx", "label"]

    def _load_labels(self):
        """
        Load labels from the labels file and return as a dictionary.
        """
        return pd.read_csv(self.labels_file_path, sep='\t', header=None).set_index(0)[1].to_dict()

    def extract_features(self, file_path):
        """
        Extract audio features from a given file path.
        """
        try:
            y, sr = librosa.load(file_path, sr=None)

            # Frequency spectrum analysis
            S = np.abs(librosa.stft(y, n_fft=2048))
            # Replace zeros with a very small number to avoid log(0)
            S = np.where(S == 0, np.finfo(float).eps, S)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

            # Filter frequencies to human voice range (0-320 Hz)
            valid_idx = np.where((freqs >= 0) & (freqs <= 320))[0]
            freqs = freqs[valid_idx]
            S = S[valid_idx, :]
            meanfreq = np.mean(freqs) / 1000
            sd = np.std(freqs) / 1000
            median = np.median(freqs) / 1000
            Q25 = np.percentile(freqs, 25) / 1000
            Q75 = np.percentile(freqs, 75) / 1000
            IQR = Q75 - Q25
            skew = pd.Series(freqs).skew()
            kurt = pd.Series(freqs).kurt()
            sp_ent = -np.sum(S * np.log(S))
            sfm = np.mean(S) / np.std(S)
            mode = pd.Series(freqs).mode()[0] / 1000
            centroid = np.mean(S) / 1000

            # Fundamental frequency parameters
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            meanfun = np.nanmean(f0)
            minfun = np.nanmin(f0)
            maxfun = np.nanmax(f0)

            # Dominant frequency parameters
            y_harm = librosa.effects.harmonic(y)
            meandom = np.mean(y_harm)
            mindom = np.min(y_harm)
            maxdom = np.max(y_harm)
            dfrange = maxdom - mindom
            duration = len(y) / sr

            # Modulation index calculation
            changes = np.abs(np.diff(y_harm))
            modindx = np.mean(changes) / dfrange if dfrange != 0 else 0

            return [duration, meanfreq, sd, median, Q25, Q75, IQR, skew, kurt, sp_ent, sfm, mode, centroid, 0, meanfun, minfun, maxfun, meandom, mindom, maxdom, dfrange, modindx]
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            return [None] * len(self.features)

    def audio_to_row(self, file_path):
        """
        Convert audio file to a row of features with the corresponding label.
        """
        file_name = os.path.basename(file_path).split('.')[0]
        label = self.labels_dict.get(file_name)
        features = self.extract_features(file_path)
        features.append(label)
        return features

    def build_dataframe(self):
        """
        Build a DataFrame from the extracted features of all audio files.
        """
        filenames = glob.glob(f"{self.audio_folder_path}/*.wav")
        if self.limit_rows:
            filenames = filenames[:self.limit_rows]

        with ProcessPoolExecutor(max_workers=4) as executor:
            rows = list(tqdm(executor.map(self.audio_to_row, filenames),
                        total=len(filenames), desc="Extracting features"))

        return pd.DataFrame(rows, columns=self.features)

    def save_to_csv(self, output_file):
        """
        Save the extracted features to a CSV file.
        """
        df = self.build_dataframe()
        df.to_csv(output_file, index=False)


def main():
    """
    Main function to execute the feature extraction and save to CSV.
    """
    audio_folder_path = "data/"
    labels_file_path = "data/targets.tsv"
    limit_rows = None  # Set to None to process all rows
    output_file = "row.csv"

    extractor = AudioFeatureExtractor(
        audio_folder_path, labels_file_path, limit_rows)
    extractor.save_to_csv(output_file)


if __name__ == "__main__":
    main()