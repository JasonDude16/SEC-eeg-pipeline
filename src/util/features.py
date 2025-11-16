import mne
import yasa
import fooof
import numpy as np
import pandas as pd
import scipy.stats as sp_stats
import scipy.signal as sp_sig
from scipy.integrate import simpson

def _get_fif_data(raw, picks=None, stage_name=None):

    raw_copy = raw.copy()
    if picks is not None:
        raw_arr = raw_copy.pick(picks).get_data() * 1e6
    else:
        raw_arr = raw_copy.pick_types(eeg=True).get_data() * 1e6
        
    if stage_name is not None:
        staging = raw[stage_name][0][0].astype(int)
    else:
        staging = None

    chans = raw_copy.ch_names
    sf = raw.info['sfreq']

    return raw_arr, staging, chans, sf


# identical to yasa.bandpower with default arguments
def _compute_psd_base(raw_arr, sf, win_sec=4, average='median', window='hamming'):
    freqs, psd = sp_sig.welch(raw_arr, fs=sf, nperseg=win_sec*sf, average=average, window=window)
    return freqs, psd


def _compute_checks(raw, picks, stage_name, include_stages):
    assert isinstance(raw, mne.io.BaseRaw), 'raw must be an instance of class mne.io.BaseRaw'
    if picks is not None:
        assert isinstance(picks, (str, list))
    if stage_name is not None:
        assert stage_name in raw.ch_names 
        assert include_stages is not None, 'include_stages cannot be None if stage_name is given'


def compute_psd(raw, picks=None, stage_name=None, include_stages=None, **kwargs):
    
    _compute_checks(raw, picks, stage_name, include_stages)
    raw_arr, staging, chans, sf = _get_fif_data(raw, picks, stage_name)

    if stage_name is None:
        freqs, psd = _compute_psd_base(raw_arr, sf, **kwargs)
        df_psd = pd.DataFrame(psd)
        df_psd.columns = freqs
        df_psd.insert(loc=0, column='chan', value=chans)
    else:
        psd_dict = dict()
        psd_list = []
        for stg in include_stages:
            data_stg = raw_arr[:, staging == stg]
            if data_stg.shape[1] == 0:
                continue
            freqs, psd = _compute_psd_base(data_stg, sf, **kwargs)
            psd_dict.update({stg:psd})
            df_psd = pd.DataFrame(psd)
            df_psd.columns = freqs
            df_psd.insert(loc=0, column='chan', value=chans)
            df_psd.insert(loc=1, column='stage', value=stg)
            psd_list.append(df_psd)

        if len(psd_list) > 0:
            df_psd = pd.concat(psd_list)
            df_psd.reset_index(inplace=True, drop=True)
        else:
            return None

    return df_psd


def compute_bandpower_from_psd(psd, freqs, bands, relative=False):

    # restrict freq range and psd according to min and max freqs in bands 
    fmin = min([b[0] for b in bands])
    fmax = max([b[1] for b in bands])
    freq_range = np.logical_and(freqs >= fmin, freqs <= fmax)
    psd = psd[:, freq_range]
    freqs = freqs[freq_range]

    freq_res = freqs[1] - freqs[0]

    bp_dict = {}
    for band in bands:
        low, high, name = band
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        bp = simpson(psd[:, idx_band], dx=freq_res)

        if relative:
            bp /= simpson(psd, dx=freq_res)

        bp_dict[name] = bp
    
    bp_dict['total_power'] = simpson(psd, dx=freq_res)
    df_bp = pd.DataFrame(bp_dict)
    df_bp.columns = ['bp_' + c for c in df_bp.columns]

    return df_bp


def compute_bandpower(raw, picks=None, stage_name=None, include_stages=None, relative=False,
    bands=[(0.5, 1, 'delta_slow'), (1, 4, 'delta_fast'), (4, 8, 'theta'), (8, 12,  'alpha'), (12, 16,  'sigma'), (16, 30,  'beta'), (30, 45,  'gamma')]):

    _compute_checks(raw, picks, stage_name, include_stages)
    df_psd = compute_psd(raw, picks, stage_name, include_stages)

    if df_psd is None:
        return None

    merge_cols = np.isin(df_psd.columns, ['stage', 'chan', 'epoch'])
    freq_cols = np.isin(df_psd.columns, ['stage', 'chan', 'epoch'], invert=True)
    
    df_bp = compute_bandpower_from_psd(np.array(df_psd.iloc[:, freq_cols]), np.array(df_psd.columns[freq_cols]), bands, relative=relative)
    df_info = df_psd.iloc[:, merge_cols].reset_index(drop=True)
    df_all = pd.concat([df_info, df_bp], axis=1)

    return df_all


def compute_aperiodics(raw, picks=None, stage_name=None, include_stages=None, freq_range=[1, 30]):

    _compute_checks(raw, picks, stage_name, include_stages)
    df_psd = compute_psd(raw, picks, stage_name, include_stages)

    if df_psd is None:
        return None

    specparam_list = []
    for r in range(df_psd.shape[0]):
        df_info = df_psd.iloc[r, np.isin(df_psd.columns, ['stage', 'chan', 'epoch'])]
        df_arr = df_psd.iloc[r, np.isin(df_psd.columns, ['stage', 'chan', 'epoch'], invert=True)]
        try:
            fm = fooof.FOOOF()
            fm.fit(freqs=np.array(df_arr.index), power_spectrum=np.array(df_arr), freq_range=freq_range)
            df_info['fooof_exponent'] = fm.get_params('aperiodic_params', 'exponent')
            df_info['fooof_offset'] = fm.get_params('aperiodic_params', 'offset')
            df_info['fooof_rsq'] = fm.get_params('r_squared')
            df_info['fooof_error'] = fm.get_params('error')
            specparam_list.append(df_info)
        except:
            pass

    df_specparam = pd.concat(specparam_list, axis=1).T
    df_psd.reset_index(inplace=True, drop=True)

    return df_specparam
