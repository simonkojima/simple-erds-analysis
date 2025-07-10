import os
import sys
from pathlib import Path
import socket

try:
    import tomllib
except:
    import toml

hostname = socket.gethostname()

path = os.path.abspath(__file__)

while True:
    path_base = os.path.dirname(path)
    if os.path.exists(os.path.join(path_base, "config.toml")):
        sys.path.append(path_base)
        path = path_base
        break
    
    if path_base == path:
        raise RuntimeError("'config.toml' was not found...")
    else:
        path = path_base

if "plafrim" in hostname:
    fname = "config_plafrim.toml"
else:
    fname = "config.toml"

try:
    with open(os.path.join(path, fname), "rb") as f:
        config = tomllib.load(f)
except:
    with open(os.path.join(path, fname), "r") as f:
        config = toml.load(f)
    
    
for key, val in config['dir'].items():
    config['dir'][key] = Path(val).expanduser()