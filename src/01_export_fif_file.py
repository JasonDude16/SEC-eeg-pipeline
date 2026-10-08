import re
import mne
import pandas as pd
from src.config import paths
import os.path as op
from src.util.process import add_night_annotations, align_hypnogram

# setup
edf_path = paths['edf']
hyp_path = paths['hypnogram']
fif_path = paths['fif']
downsample_rate = 100
low_freq_filt = 0.3
high_freq_filt = 35
reprocess = False
stage_map = {
  0: 0, 
  1: 1, 
  2: 2, 
  3: 3, 
  4: 2,  
  5: 4, 
  9: -2,
  6: -2
} 

###################################################################################################

#  get list of all edfs
all_edfs = sorted(edf_path.glob('*.edf'))
if not all_edfs:
    raise FileNotFoundError(f'No EDF files found in {edf_path}. Check config.toml or config.local.toml.')
if not hyp_path.is_dir():
    raise FileNotFoundError(f'Hypnogram directory not found: {hyp_path}')
fif_path.mkdir(parents=True, exist_ok=True)

for f in all_edfs:

    # get id, find associated hypnogram and epoch report
    match = re.search(r'_(\d+)_Export\.edf$', f.name)
    if match is None:
        raise ValueError(f'Unexpected EDF filename: {f.name}. Expected <prefix>_<subject>_Export.edf.')
    subj = match.group(1)

    # skip subject if fif file already exists
    fif_file = fif_path / f'SEC_{subj}_raw.fif.gz'
    if fif_file.is_file() and reprocess is False:
        print(f"{subj} fif already exists and reprocess is set to False, skipping..")
        continue

    hyp_file = sorted(p for p in hyp_path.glob('*.csv')
                      if re.search(r'(?<!\d)' + re.escape(subj) + r'(?!\d)', p.stem))
    if len(hyp_file) > 1:
        raise ValueError(f'{subj} | Multiple hypnograms found: {hyp_file}')
    hyp_file = hyp_file[0] if len(hyp_file) else None

    if hyp_file is None:
        print(f"{subj} | No hypnogram found. SKIPPING SUBJECT.")
        continue

    raw = mne.io.read_raw_edf(f, preload=True)
    raw.filter(l_freq=low_freq_filt, h_freq=high_freq_filt)
    raw.resample(downsample_rate)

    df_hypno = pd.read_csv(hyp_file, index_col=False)
    df_hypno.PrimaryAutoStage = df_hypno.PrimaryAutoStage.map(stage_map)
    if df_hypno.PrimaryAutoStage.isna().any():
        raise ValueError(f'{subj} | Missing or unmapped PrimaryAutoStage values.')
    hypno = align_hypnogram(df_hypno, raw.info['sfreq'], raw.n_times)

    info = mne.create_info(ch_names=['hypno'], ch_types=['misc'], sfreq=raw.info['sfreq'])
    hypno_arr = mne.io.RawArray([hypno], info, verbose=False)
    raw.add_channels([hypno_arr], force_update_info=True)

    ann = add_night_annotations(df_hypno)
    raw.set_annotations(ann)

    # export
    outname = op.join(fif_path, f'SEC_{subj}_raw.fif.gz')
    raw.save(outname, overwrite=True, verbose=False)
