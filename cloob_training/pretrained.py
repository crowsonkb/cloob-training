from hashlib import sha256
from pathlib import Path

from torch import hub

from .config import load_config


CONFIG_DIR = Path(__file__).resolve().parent / 'pretrained_configs'


def list_configs():
    return sorted(path.stem for path in CONFIG_DIR.glob("*.json"))


def get_config(name):
    config_path = CONFIG_DIR / (name + '.json')
    if config_path.is_file():
        return load_config(config_path)


def download_checkpoint(config):
    dest_dir = Path(hub.get_dir()) / 'cloob'
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = config['url']
    filename = url.rpartition('/')[2]
    model_hash = filename.rpartition('-')[2].partition('.')[0]
    dest_file = dest_dir / filename
    if not dest_file.exists():
        hub.download_url_to_file(url, str(dest_file), hash_prefix=model_hash)
    if sha256(open(dest_file, 'rb').read()).hexdigest() == model_hash:
        return str(dest_file)
    raise RuntimeError(f'Model has been downloaded to {dest_file!s} but the SHA256 checksum does not not match')
