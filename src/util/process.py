import mne
import pandas as pd
import warnings
import numpy as np
from mne import concatenate_raws, Annotations

def align_hypnogram(hypno, sf, n_times, elapsed_col="ElapsedTime(sec)", epoch_dur=30):

    elapsed = pd.to_numeric(hypno[elapsed_col], errors="raise").to_numpy()
    if len(elapsed) == 0 or not np.isfinite(elapsed).all():
        raise ValueError('Hypnogram elapsed times must be finite and nonempty.')
    starts = np.rint(elapsed * sf).astype(int)
    ends = np.rint((elapsed + epoch_dur) * sf).astype(int)
    if (starts < 0).any() or (starts >= n_times).any():
        raise ValueError('Hypnogram epoch starts fall outside the recording; check alignment.')
    if (starts[1:] < ends[:-1]).any():
        raise ValueError('Hypnogram epochs overlap or are not in chronological order.')

    # Preserve initial offsets and gaps instead of shifting later stages earlier.
    staging = np.full(n_times, -2, dtype=int)
    for start,end,stage in zip(starts, ends, hypno.PrimaryAutoStage):
        staging[start:min(end, n_times)] = stage
    return staging


def add_night_annotations(hypno, night_col="Night", elapsed_col="ElapsedTime(sec)",
    time_offset_sec=0.0, epoch_dur=30):

    # Ensure numeric elapsed times
    hypno[elapsed_col] = pd.to_numeric(hypno[elapsed_col], errors="coerce")
    hypno = hypno.dropna(subset=[elapsed_col]).sort_values(elapsed_col)

    onsets, durations, labels = [], [], []
    for night, g in hypno.groupby(night_col):
        g = g.sort_values(elapsed_col)

        start = float(g[elapsed_col].min()) + time_offset_sec
        last_epoch_start = float(g[elapsed_col].max()) + time_offset_sec
        duration = (last_epoch_start - start) + epoch_dur

        onsets.append(start)
        durations.append(duration)
        labels.append(f"night_{int(night)}")

    # Build annotation object
    ann = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=labels,
    )
    return ann

def split_raw_by_annotation(raw, ann_text, epoch_length, concat_multiple=False):
  
  ann_df = raw.annotations.to_data_frame()
  
  raw_subset = {}
  for text in ann_text:
  
    ann_match = ann_df['description'].eq(text)
    ann_match_loc = ann_match[ann_match].index
    
    if len(ann_match_loc) == 0:
      warnings.warn(f'No text matching {text}, skipping...') 
      continue

    raw_concat = []
    for i in range(len(ann_match_loc)):
      start_time = raw.annotations.onset[ann_match_loc[i]] - raw.first_time
      end_time = start_time + raw.annotations.duration[ann_match_loc[i]]
  
      # anchor epochs to the annotated start, which need not be a multiple of 30
      start_epoch = start_time
      end_epoch = end_time - 1 / raw.info['sfreq']
      
      # if the annotation extends beyond recording use the last sample
      if end_epoch > raw.times[len(raw.times)-1]:
        end_epoch = raw.times[len(raw.times)-1]
    
      if len(ann_match_loc) > 1:
        if concat_multiple: 
          raw_concat.append(raw.copy().crop(start_epoch, end_epoch))
          if i == (len(ann_match_loc) - 1):
            raw_subset[text] = concatenate_raws(raw_concat)
        else:
          raw_subset[text + '_' + str(i + 1)] = raw.copy().crop(start_epoch, end_epoch)
      else:
        raw_subset[text] = raw.copy().crop(start_epoch, end_epoch)
        
  return raw_subset
