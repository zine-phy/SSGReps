"""Repository-root command-line entry point."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent/'SSGReps'))
from SSGReps import main

if __name__ == '__main__':
    main()
