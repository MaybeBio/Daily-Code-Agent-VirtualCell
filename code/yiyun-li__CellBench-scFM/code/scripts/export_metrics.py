#!/usr/bin/env python3
"""Export versioned metrics; do not combine native-scale errors with correlations."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bio.io.export import export_task, export_panels, PRIMARY
from bio.data.catalog import PANEL_ORDER

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,choices=list(PRIMARY))
    parser.add_argument('--root',default='output')
    parser.add_argument('--panel',choices=list(PANEL_ORDER),
                        help='One main-text panel. Omit to write every panel that has results, never a mixed table.')
    args=parser.parse_args()
    if args.panel:
        export_task(args.task,args.root,panel=args.panel)
    else:
        export_panels(args.task,args.root)
