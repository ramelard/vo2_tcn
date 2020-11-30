import pandas as pd
import os
import re
import numpy as np

# For each file in ../data/backup_noVO2filtering/{train,test,validation}, filter and interpolate the corresponding raw
# VO2 file and save it to our main ../data/{train,test,validation} folders.
folders = ['test', 'train', 'validation']
for dataset in folders:
    # Load old data and overwrite VO2 columns
    data_dir = f'../data/backup_noVO2filtering/{dataset}/'
    csv_files = os.listdir(data_dir)

    window_breaths = 5
    for filename in csv_files:
        df = pd.read_csv(data_dir + filename)
        match = re.search('(high|mid|low|max)(\d+)(_\d)*.csv', filename)
        if match is None:
            print(f'Could not interpret data file {filename}. Skipping.')
            continue
        protocol = match.group(1)
        pid = int(match.group(2))

        file_raw = f'{protocol}{pid}_rawVO2.csv'
        try:
            df_raw = pd.read_csv(f'../data/breathbybreath/{file_raw}')
        except FileNotFoundError:
            print(f'../data/breathbybreath/{file_raw} not found. Skipping.')
            continue
        vo2_1hz = df_raw.VO2.rolling(window_breaths, min_periods=1, center=True).median().to_numpy()
        vo2rel_1hz = df_raw.VO2_rel.rolling(window_breaths, min_periods=1, center=True).median().to_numpy()
        t = df_raw['time'].to_numpy()
        # Use 1Hz interpolation time to match existing data
        xx_1hz = df['Time'].to_numpy()
        vo2_1hz = np.interp(xx_1hz, t, vo2_1hz)
        vo2rel_1hz = np.interp(xx_1hz, t, vo2rel_1hz)

        assert_msg = f'{filename} df.shape[0]={df.shape[0]}, vo2.size={vo2_1hz.size}'
        assert df.shape[0] == vo2_1hz.size, assert_msg

        df['VO2'] = vo2_1hz
        df['VO2rel'] = vo2rel_1hz
        print(f'Writing ../data/{dataset}/{filename}')
        df.to_csv(f'../data/{dataset}/{filename}', index=False)
