import os
import pandas as pd
import re

def create_master_csv(root_folder):
    output_folder = os.path.join(root_folder, "Output")
    aggregated_folder = os.path.join(output_folder, "Aggregated")
    os.makedirs(aggregated_folder, exist_ok=True)

    for top_folder in sorted(os.listdir(output_folder)):
        top_folder_path = os.path.join(output_folder, top_folder)
        if not os.path.isdir(top_folder_path):
            continue

        all_features = []
        for subfolder in sorted(os.listdir(top_folder_path)):  
            subfolder_path = os.path.join(top_folder_path, subfolder)
            feature_csv = os.path.join(subfolder_path, "Front_features.csv")

            if os.path.exists(feature_csv):
                feature_df = pd.read_csv(feature_csv)

                match = re.match(r'([A-Za-z]+)-([S\d+]+)', subfolder)
                if match:
                    tale = match.group(1)
                    segment = match.group(2)
                else:
                    tale = "Unknown"
                    segment = "Unknown"

                feature_df.insert(0, 'Segment', segment)
                feature_df.insert(0, 'Story', tale)
                all_features.append(feature_df)

        if all_features:
            master_df = pd.concat(all_features, ignore_index=True)

            master_df['Segment_numeric'] = master_df['Segment'].apply(lambda x: int(re.search(r'(\d+)', x).group()))
            master_df = master_df.sort_values(by='Segment_numeric')
            master_df = master_df.drop(columns=['Segment_numeric'])

            aggregated_subfolder = os.path.join(aggregated_folder, top_folder)
            os.makedirs(aggregated_subfolder, exist_ok=True)

            aggregated_csv = os.path.join(aggregated_subfolder, 'AllFront_features.csv')
            master_df.to_csv(aggregated_csv, index=False)
            print(f"Master CSV saved to: {aggregated_csv}")
        else:
            print(f"No valid feature files found in {top_folder_path}")

if __name__ == "__main__":
    root_folder = "/project/" # Path to root directory with Output folder
    create_master_csv(root_folder)
