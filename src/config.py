from pathlib import Path
import tomllib

# setup
repo_path = Path(__file__).resolve().parents[1]
config_path = repo_path / 'config.local.toml'
if not config_path.is_file():
    config_path = repo_path / 'config.toml'

with open(config_path, 'rb') as f:
    config = tomllib.load(f)

paths = {key: (repo_path / Path(value).expanduser()).resolve()
         for key,value in config['paths'].items()}
