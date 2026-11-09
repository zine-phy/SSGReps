"""Load the bundled, trusted SSG database, including directly from its archive."""
from functools import lru_cache
from pathlib import Path
import pickle
import tarfile


@lru_cache(maxsize=1)
def load_database():
    directory = Path(__file__).resolve().parent.parent/'ssg_data'
    unpacked = directory/'identify.pkl'
    archive = directory/'identify.pkl.tar.gz'
    if unpacked.exists():
        with unpacked.open('rb') as stream:
            records = pickle.load(stream)
    elif archive.exists():
        # Read only the known member; do not extract paths from the archive.
        with tarfile.open(archive, 'r:gz') as tar:
            member = next((m for m in tar.getmembers() if Path(m.name).name == 'identify.pkl' and m.isfile()), None)
            if member is None:
                raise ValueError('The database archive does not contain identify.pkl.')
            with tar.extractfile(member) as stream:
                records = pickle.load(stream)
    else:
        raise FileNotFoundError('Missing bundled database: ssg_data/identify.pkl.tar.gz')
    return {record['ssgNum']: record for record in records}
