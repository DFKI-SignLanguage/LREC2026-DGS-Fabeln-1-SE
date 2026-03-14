from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

FEATURES = ["dist_elbows_lr_avg",
    # "mouthSmileRight_mean",
    #         "pose_LEFT_HIP_x_mean", "pose_LEFT_HIP_y_std",
    #         "pose_LEFT_HIP_y_velocity_std",
    #         "pose_RIGHT_HIP_y_velocity_std", "pose_RIGHT_HIP_y_acceleration_std",
    #         "pose_LEFT_ELBOW_z_mean", "pose_RIGHT_ELBOW_y_std",
    #         "pose_LEFT_SHOULDER_z_mean",
    #         "torso_yaw_mean",
    #         "eyeLookDownLeft_mean",
            "jawRight_mean"
            ]

def plot_correlations(in_df: pd.DataFrame, out_folder: Path) -> None:
    """For each of the FEATURES, take the column from the input dataframe and plot a a graph where:
      - on the horizontal axis, there are the labels in order negative, neutral, positive, mapped to values 0, 1, 2.
      - On the vertical axis there is the feature value
      - values are scattered to allows visibility for close values
     """
    for feature in FEATURES:
        print("Feature: ", feature)

        # Map sentiment labels to numbers
        sentiment_mapping = {"negative": 0, "neutral": 1, "positive": 2}
        in_df["SentimentNum"] = in_df["Sentiments-Aggregated"].str.lower().map(sentiment_mapping)
        x_values = in_df["SentimentNum"].to_numpy(dtype=np.float64)
        y_values = in_df[feature].to_numpy(dtype=np.float64)
        print("x_values", len(x_values), x_values.min(), x_values.max(), x_values.mean())
        print("y_values", len(y_values), y_values.min(), y_values.max(), y_values.mean())

        #
        # Plot a scatterplot with sentiments as x and feature values as y
        plt.figure(figsize=(6, 4))
        # plt.xlabel("Sentiment (0=negative, 1=neutral, 2=positive)")
        # plt.ylabel(feature)
        plt.title(f"{feature}")
        plt.xticks([0, 1, 2], ["negative", "neutral", "positive"])
        plt.xlim(-0.2, 2.2)
        plt.ylim((y_values.min(), y_values.max()))
        # Scatter plot
        assert len(in_df["SentimentNum"]) == len(y_values)
        x_values_jittered = x_values + np.random.normal(0, 0.02, size=len(x_values))
        plt.scatter(x=x_values_jittered, y=y_values, c='blue', alpha=0.2)

        # Measure the linear correlation between x and y values
        correlation = np.corrcoef(x_values, y_values)[0, 1]
        # print(f"Correlation between {feature} and sentiment: {correlation:.4f}")
        # Plot a line fitting the linear correlation, using bias and a slope
        slope, intercept = np.polyfit(x_values, y_values, 1)
        x_fit = np.array([-0.2, 2.2])
        y_fit = intercept + slope * x_fit
        plt.plot(x_fit, y_fit, color='red', linestyle='--', label='Linear fit')
        # Report values on plot
        plt.figtext(0.20, 0.75, f"Intercept: {intercept:.3f}\nSlope: {slope:.3f}\nCorrelation: {correlation:.3f}", fontsize=10, ha="left")

        # Finish up
        plt.grid(True)
        plt.tight_layout()
        plot_out_path = out_folder / f"Correlation_{feature}_Sentiment.pdf"
        print("Saving to", plot_out_path)
        plt.savefig(plot_out_path, bbox_inches='tight')
        plt.close()


        #
        #
        # Plot an histogram with the distribution of the values of the features (y_values)
        # Three histograms must be overlapped, with different color according to the associated sentiment (x_values) and the mapping {"Negative": 0, "Neutral": 1, "Positive": 2}
        plt.figure(figsize=(6, 4))
        bins = 30
        plt.hist(y_values[x_values == 0], bins=bins, alpha=0.5, label='negative', color='red')
        plt.hist(y_values[x_values == 1], bins=bins, alpha=0.5, label='neutral', color='gray')
        plt.hist(y_values[x_values == 2], bins=bins, alpha=0.5, label='positive', color='green')
        # plt.xlabel(feature)
        # plt.ylabel("Count")
        plt.title(f"{feature}")
        # Finish up
        plt.legend()
        plt.tight_layout()
        plot_out_path = out_folder / f"Histogram_{feature}_by_Sentiment.pdf"
        print("Saving to", plot_out_path)
        plt.savefig(plot_out_path, bbox_inches='tight')
        plt.close()


#
#
#
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plot correlations between a feature and the sentiment values')
    parser.add_argument('--labels-dataframe', '-l', type=str, help="Input dataframe with all segment-level sentiments.")
    parser.add_argument('--features-dataframe', '-f', type=str, help="Input dataframe with all segment-level motion features.")
    parser.add_argument('--out-dir', '-o', required=True, help="The output directory for all stats and plots")

    args = parser.parse_args()

    labels_path = Path(args.labels_dataframe)
    features_path = Path(args.features_dataframe)
    out_path = Path(args.out_dir)

    if not labels_path.exists():
        raise ValueError(f"Input dataframe file does not exist: {labels_path}")

    if not features_path.exists():
        raise ValueError(f"Input dataframe file does not exist: {features_path}")

    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(labels_path)
    print("Labels shape:", labels_df.shape)
    features_df = pd.read_csv(features_path)
    print("Features shape:", features_df.shape)

    # Drop rows where the aggregated sentiment is "multi"
    labels_df = labels_df[~ (labels_df["Sentiments-Aggregated"] == "multi")]
    print("Labels no-multi shape:", labels_df.shape)

    # Merge the two datasets using jointly Story and id columns as key
    merged_df = pd.merge(labels_df, features_df, on=["Story", "id"], how="inner")
    dropped_labels = len(labels_df) - len(merged_df)
    dropped_features = len(features_df) - len(merged_df)
    if dropped_labels > 0:
        print(f"WARNING: {dropped_labels} row(s) from labels had no match in features.")
    if dropped_features > 0:
        print(f"WARNING: {dropped_features} row(s) from features had no match in labels.")
    print(f"Merged rows: {len(merged_df)}")

    print("Merged DF shape: ", merged_df.shape)
    print(merged_df.head(5))

    plot_correlations(in_df=merged_df, out_folder=out_path)

    print("All done.")
