#!/usr/bin/env python3
"""SpinDoctor - entry point.

Measures basketball spin from shooting clips, using rigid-body matching of the
markers on the ball surface.

Examples:
    python main.py --video shot1.MOV --output ./output
    python main.py --folder ./videos --output ./output
    python main.py --player-folder ./gym_videos --output ./output
"""

from spindoctor.cli import main

if __name__ == '__main__':
    main()
