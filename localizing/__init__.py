from pathlib import Path

import lmwrapper
import lmwrapper.caching

cur_path = Path(__file__).parent.absolute()

set_cache_dir = cur_path / "../.lmwrapper_cache"
print("Setting cache dir to", set_cache_dir.resolve())
lmwrapper.caching.set_cache_dir(set_cache_dir)
