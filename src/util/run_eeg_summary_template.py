import os
import re
import numpy as np
import nbformat
import os.path as op
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from src.config import repo_path, paths
from src.util.process import split_raw_by_annotation

# setup
template_path = repo_path / 'src/util/eeg_summary_template.ipynb'
output_dir = paths['reports']
fif_path = paths['fif']

output_dir.mkdir(parents=True, exist_ok=True)

files = sorted(fif_path.glob("*_raw.fif.gz"))
if not files:
  raise FileNotFoundError(f'No FIF files found in {fif_path}. Run the export step first.')
ids = [f.name.replace('_raw.fif.gz', '') for f in files]

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
        if cell.cell_type == 'code':
          cell.source = cell.source.replace(repr(placeholder), repr(str(v)))
        else:
          cell.source = cell.source.replace(placeholder, str(v))
  return notebook

# generate notebook for each participant
def generate_report(template_path, info, output_dir):
  with open(template_path) as f:
    template_nb = nbformat.read(f, as_version=4)
  
  subj_nb = replace_placeholders(template_nb, info)
  subj_nb_path = os.path.join(output_dir, f"{info['idx']}_report.ipynb")
  
  html_path = os.path.join(output_dir, f"{info['idx']}_report.html")
  if os.path.exists(subj_nb_path) and os.path.exists(html_path):
    return (f"{info['idx']} reports already exist, skipping...")
  
  # execute 
  ep = ExecutePreprocessor(timeout=600, kernel_name='sec-eeg')
  ep.preprocess(subj_nb, {'metadata': {'path': str(repo_path)}})
  
  # save
  with open(subj_nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(subj_nb, f)
  
  # convert to HTML
  html_exporter = HTMLExporter()
  body, _ = html_exporter.from_notebook_node(subj_nb)
  with open(html_path, 'w', encoding='utf-8') as f:
    f.write(body)

for i in range(len(info)):
  print(f"Generating report for {info[i]['idx']}...")
  result = generate_report(template_path, info[i], output_dir)
  print(result or 'Done')
