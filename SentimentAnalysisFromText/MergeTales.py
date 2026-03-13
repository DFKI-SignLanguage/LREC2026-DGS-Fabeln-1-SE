import os
import pandas as pd
import numpy as np

from pathlib import Path

#
# python MergeTales.py -d ../ -o testall.csv
#  

EVALUATED_POSTFIX = "-Evaluated.csv"
 
def merge_tales(base_dir: Path) -> pd.DataFrame:

    # label_data, feature_data = {}, {}

    all_dfs = []

    sorted_stories = sorted(os.listdir(base_dir))
 
    for folder in sorted_stories:

        # story_path = os.path.join(base_dir, folder)
        lab_csv = base_dir / folder / (folder + EVALUATED_POSTFIX)
        print(f"Looking for file '{lab_csv}'")

        if not lab_csv.exists():
            print("Doesn't exist. Skipping...")
            continue

        if not lab_csv.is_file():
            print("Not a file. Skipping...")
            continue

        # Read the CSV file into a DataFrame
        lab_df = pd.read_csv(lab_csv, sep=';')
        # Add a column with the story name
        # Insert at the beginning
        # The same value for all rows
        lab_df.insert(0, "Story", folder)
    
        # Add each DataFrame to the list
        all_dfs.append(lab_df)

    
    # Merge all dataframes
    merged_df = pd.concat(all_dfs, ignore_index=True)  # Concatenate the list of DataFrames into a single DataFrame

    return merged_df


#
# MAIN
if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description='Aggregates data from multiple CSV files')
    parser.add_argument('--tales-dir', '-d', type=str, required=True,
                        help='Path to the directory containing the tale files (AllSentences_Sentiment.csv and AllFront_features.csv)')
    parser.add_argument('--out-csv', '-o', type=str, required=True,
                        help='Output path for aggregated data (merged CSV file)')

    args = parser.parse_args()

    base_dir = Path(args.tales_dir)
    output_csv = Path(args.out_csv)

    if not base_dir.exists():
        raise ValueError(f"Directory does not exist: {base_dir}")

    # Merge all CSV sentiment files in the directory
    out_df: pd.DataFrame = merge_tales(base_dir)

    # Save the DataFrame to a CSV file
    out_df.to_csv(output_csv, header=True, index=False, sep=',')

    unique_stories = out_df["Story"].unique()
    print(f"Stories ({len(unique_stories)}):", unique_stories)
    print(f"Output {len(out_df)} rows.")

    print("All done.")
