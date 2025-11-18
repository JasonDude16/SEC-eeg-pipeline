import re
import os
import mne
import yasa
import os.path as op
from pathlib import Path
from src.util.process import split_raw_by_annotation
from src.util.features import *

fif_path = Path('/Users/jasondude/Library/Mobile Documents/com~apple~CloudDocs/Desktop/SEC_EEG/fif')
out_path = '/Users/jasondude/Library/Mobile Documents/com~apple~CloudDocs/Desktop/SEC_EEG/features'

##################################################################################

# get all fif files
all_fifs = sorted(fif_path.glob('*fif.gz'))

for f in all_fifs:

  subj = str(f).replace('_raw.fif.gz', '').split('/')[-1]

  out_file = op.join(out_path, f'{subj}.csv')
  if op.isfile(out_file):
    print(f'{subj} | files exist, skipping...') 
    continue
  
  raw = mne.io.read_raw_fif(f, preload=True)
  raw_segments = split_raw_by_annotation(raw, ann_text=['night_1', 'night_2', 'night_3'], epoch_length=30)

  nights = []
  for key,raw_night in raw_segments.items():
    
    df_feat_night = yasa.compute_features_stage(raw_night.copy().pick('EEG'), hypno=raw_night['hypno'][0][0], do_1f=False)
    df_feat_night.reset_index(inplace=True)
    df_feat_night['night'] = key
    
    df_ap_night = compute_aperiodics(raw_night, picks='EEG', stage_name='hypno', include_stages=[0,2,3,4])
    df_ap_night['stage'] = df_ap_night['stage'].map({0: 'WN', 2: 'N2', 3: 'N3', 4: 'REM'})
    df_ap_night['night'] = key
    
    df_all_night = pd.merge(df_feat_night, df_ap_night, how='outer')
    df_all_night.insert(0, 'night', df_all_night.pop('night'))
    nights.append(df_all_night)
    
  df_all = pd.concat(nights)
  df_all.insert(0, 'subj', subj)
  df_all.to_csv(out_file, index=False)
  print(f'{subj} | done \n')
