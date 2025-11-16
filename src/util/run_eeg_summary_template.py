import os
import re
import numpy as np
import nbformat
import os.path as op
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path
from src.util.process import split_raw_by_annotation

# setup
template_path = 'src/util/eeg_summary_template.ipynb'
output_dir = 'report/eeg_summary'
fif_path = Path('/Users/jasondude/Library/Mobile Documents/com~apple~CloudDocs/Desktop/SEC_EEG/fif')

if not op.exists(output_dir):
  os.mkdir(output_dir)

files = sorted(fif_path.glob("*.fif.gz"))
ids = [str(f).replace('_raw.fif.gz', '').split('/')[-1] for f in files]

# set up dictionary of params to modify in notebook
info = []
for i in range(len(ids)):
  subj = {}
  subj['idx'] = ids[i]
  subj['fif_path'] = fif_path
  info.append(subj)

# replace placeholders in the notebook
def replace_placeholders(notebook, subj):
  for cell in notebook.cells:
    if cell.cell_type == 'markdown' or cell.cell_type == 'code':
      for k,v in subj.items():
        placeholder = f'{{{{{k}}}}}'
        cell.source = cell.source.replace(placeholder, str(v))
  return notebook

# generate notebook for each participant
def generate_report(template_path, info, output_dir):
  with open(template_path) as f:
    template_nb = nbformat.read(f, as_version=4)
  
  subj_nb = replace_placeholders(template_nb, info)
  subj_nb_path = os.path.join(output_dir, f"SEC_{info['idx']}_report.ipynb")
  
  if os.path.exists(subj_nb_path):
    return (f"SEC {info['idx']} already exists and reprocess=False, skipping...")
  
  # execute 
  ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
  ep.preprocess(subj_nb, {'metadata': {'path': './'}})
  
  # save
  with open(subj_nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(subj_nb, f)
  
  # convert to HTML
  html_exporter = HTMLExporter()
  body, _ = html_exporter.from_notebook_node(subj_nb)
  html_path = os.path.join(output_dir, f"SEC_{info['idx']}_report.html")
  with open(html_path, 'w', encoding='utf-8') as f:
    f.write(body)

for i in range(len(info)):
  print(f"Generating report for SEC {info[i]['idx']}...")
  generate_report(template_path, info[i], output_dir)
  print('Done')
