from pathlib import Path
import argparse
from pathsnap.matcher import train_compare

if __name__ == "__main__":
    ap=argparse.ArgumentParser(description="Train and compare PathSnap phase-1 candidate rankers.")
    ap.add_argument("--data-dir",type=Path,default=Path("data/Student folder"))
    ap.add_argument("--output-dir",type=Path,default=Path("outputs"))
    ap.add_argument("--candidates",type=int,default=12)
    args=ap.parse_args()
    rows,selected,_=train_compare(args.data_dir,args.output_dir,k=args.candidates)
    print(f"Selected from training-trip validation: {selected}")
    for row in rows: print(row["model"],row.get("validation_top1_accuracy"),row.get("student_test_top1_accuracy"))
