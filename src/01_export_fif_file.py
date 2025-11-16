import re
import mne
import yasa
import pandas as pd
from pathlib import Path
import os.path as op
from src.util.process import add_night_annotations

# setup
edf_path = Path('/Users/jasondude/Library/Mobile Documents/com~apple~CloudDocs/Desktop/SEC_EEG/edf')
hyp_path = Path('/Users/jasondude/Library/Mobile Documents/com~apple~CloudDocs/Desktop/SEC_EEG/hypnogram')
fif_path = Path('/Users/jasondude/Library/Mobile Documents/com~apple~CloudDocs/Desktop/SEC_EEG/fif')
downsample_rate = 100
low_freq_filt = 0.3
high_freq_filt = 35
reprocess = False
stage_map = {
  0: 0, 
  1: 1, 
  2: 2, 
  3: 3, 
  4: 3, 
  5: 4, 
  9: -2
} 

###################################################################################################

#  get list of all edfs
all_edfs = sorted(edf_path.glob('*.edf'))

for f in all_edfs:

    # get id, find associated hypnogram and epoch report
    subj = re.search(r'_(\d+)_Export\.edf$', str(f)).group(1)

    # skip subject if fif file already exists
    fif_file = list(fif_path.glob(subj + '*.fif.gz'))
    fif_file = fif_file[0] if len(fif_file) else None
    if fif_file is not None and reprocess is False:
        print(f"{subj} fif already exists and reprocess is set to False, skipping..")
        continue

    hyp_file = list(hyp_path.glob('*' + subj + '*.csv'))
    hyp_file = hyp_file[0] if len(hyp_file) else None

    if hyp_file is None:
        print(f"{subj} | No hypnogram found. SKIPPING SUBJECT.")
        continue

    raw = mne.io.read_raw_edf(f, preload=True)
    raw.filter(l_freq=low_freq_filt, h_freq=high_freq_filt)
    raw.resample(downsample_rate)

    df_hypno = pd.read_csv(hyp_file, index_col=False)
    df_hypno.PrimaryAutoStage = df_hypno.PrimaryAutoStage.map(stage_map)
    hypno = yasa.hypno_upsample_to_data(df_hypno.PrimaryAutoStage, sf_hypno=1/30, sf_data=raw.info['sfreq'], data=raw)

    info = mne.create_info(ch_names=['hypno'], ch_types=['misc'], sfreq=raw.info['sfreq'])
    hypno_arr = mne.io.RawArray([hypno], info, verbose=False)
    raw.add_channels([hypno_arr], force_update_info=True)

    ann = add_night_annotations(df_hypno)
    raw.set_annotations(ann)

    # export
    outname = op.join(fif_path, f'SEC_{subj}_raw.fif.gz')
    raw.save(outname, overwrite=True, verbose=False)
