from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


MODELS = ["GPT5", "Perplexity", "Mistral", "GPTOSS20B", "Aggregated"]


def plot_tales_sentiment_stats(in_df: pd.DataFrame, out_path: Path, model: str, prefix: str) -> None:

    # Count the occurrences of each sentiment in the model column
    sentiment_counts = in_df["Sentiments-" + model].value_counts().sort_index()
    # print(sentiment_counts)

    # Summarize multi vs. single prediction
    n_rows = len(in_df)
    stats = f"Rows: {n_rows}\n"
    model_multi_col = in_df["Multi-" + model]
    multi_count = (model_multi_col == "yes").sum()
    stats = stats + f"Rows with multiple predictions: {multi_count}\n"
    stats = stats + f"Rows with single prediction: {n_rows - multi_count}"
    
    with open(out_path / (prefix + "SentimentCounts.txt"), "w") as f:
        f.write(str(sentiment_counts))
        f.write("\n======\n")
        f.write(stats)


    # Plot a bar chart with the sentiment counts
    plt.figure(figsize=(5, 3))
    sentiment_counts.plot(kind="bar")
    # plt.xlabel("Sentiment")
    plt.xlabel(None)
    plt.ylabel(None)  # "Count")
    plt.title("Sentiments count after majority vote")
    plt.xticks(rotation=30)  # Rotate x-axis labels for better readability
    plt.grid(axis='y')
    plt.tight_layout()
    plot_out_path = out_path / (prefix + "SentimentCounts.pdf")
    print("Saving to", plot_out_path)
    plt.savefig(plot_out_path, bbox_inches='tight')
    plt.close()


def plot_sentiment_progression(in_df: pd.DataFrame, out_path: Path, sentiment_col: str) -> None:

    stories = in_df["Story"].unique()
    print(stories)

    for story in stories:
        print(f"Plotting progression for {story}...")
        # Filter entries only for this story
        story_df: pd.DataFrame = in_df[in_df["Story"] == story].copy()
        print(f"{len(story)} segments")

        # in story_df, for column sentiment_col, substitute labels negative, neutral and positive with 0, 1, and 2 respectively
        story_df["SentimentStr"] = story_df[sentiment_col].str.lower()
        story_df["Sentiment"] = story_df["SentimentStr"].map({"negative": 0, "neutral": 1, "positive": 2})

        print(story_df["Sentiment"].value_counts())

        # Plot setup
        plt.figure(figsize=(4, 3))

        # Compute how many tick to plot on the horizontal axis
        MAX_HORIZ_TICKS = 7
        tick_step = len(story_df["id"]) // MAX_HORIZ_TICKS
        ticks = story_df["id"][::tick_step].values.tolist()  # Get every Nth index

        #
        # Plot the sentiment scatter line
        # Plot a line graph using the column "id" for the horizonal axis, and "Sentiment" for the vertical
        plt.plot(story_df["id"], story_df["Sentiment"], marker='o', linestyle='-')

        # 
        # Plot smoothed sentiment curve
        window_size = 7
        story_df["SentimentSmooth"] = story_df["Sentiment"].rolling(window=window_size, center=True, min_periods=1).mean()
        plt.plot(story_df["id"], story_df["SentimentSmooth"], color='orange', linestyle='--', label=f'Moving avg (window={window_size})')

        # plt.xlabel(None)
        # plt.ylabel(None)

        plt.title(f"{story}")
        plt.ylim(-0.2, 2.2)
        plt.yticks([0, 1, 2], ['Negative', 'Neutral', 'Positive'])
        plt.xticks(ticks)  # Use the generated ticks
        plt.grid()

        # plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
        plt.tight_layout()

        # Save tghe plot
        plot_out_path = out_path / f"SentimentProgression-{story}.pdf"
        print("Saving to", plot_out_path)
        plt.savefig(plot_out_path, bbox_inches='tight')
        plt.close()

        # story_df["Sentiment"] = np.where(story_df["Sentiment"].str.contains("negative"), -1, 0)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Ollama Script')
    parser.add_argument('--in-dataframe', '-i', type=str, help="Input dataframe with all sentiment information.")
    parser.add_argument('--out-dir', '-o', required=True, help="The output directory for all stats and plots")
    parser.add_argument('--model', '-m', type=str, required=False, choices=MODELS, help="Model to use (default: Perplexity)")

    args = parser.parse_args()

    in_path = Path(args.in_dataframe)
    out_path = Path(args.out_dir)
    model = args.model

    print("Available models:", MODELS)

    SENTIMENT_COLUMN = "Sentiments"
    MULTI_COLUMN = "Multi"
    if model is not None:
        SENTIMENT_COLUMN = SENTIMENT_COLUMN + '-' + model
        MULTI_COLUMN = MULTI_COLUMN + '-' + model

    print("Using columns '{SENTIMENT_COLUMN}' and '{MULTI_COLUMN}'.")

    if not in_path.exists():
        raise ValueError(f"Input dataframe file does not exist: {in_path}")
    
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)

    in_df = pd.read_csv(in_path, sep=',')
    print("Rows:", len(in_df))

    cols = in_df.columns
    print(f"Columns ({len(cols)})", cols)

    # Plot all the data stats
    print("Plotting stats...")
    plot_tales_sentiment_stats(in_df=in_df, out_path=out_path, model=model, prefix="Full-")

    #
    # Filter away rows where the MULTI_COLUMN is yes
    print("Removing multiple sentiment rows ...")
    clean_multi = in_df[MULTI_COLUMN].str.strip()  # Remove leading and trailing spaces from strings
    in_df = in_df[clean_multi != "yes"]
    print("Rows:", len(in_df))
    print(in_df[SENTIMENT_COLUMN].value_counts())
    print(in_df[MULTI_COLUMN].value_counts())
    print([f"'{c}'" for c in in_df[MULTI_COLUMN].value_counts().index])

    # Plot the data stats for predictions without ambiguities
    print("Plotting stats...")
    plot_tales_sentiment_stats(in_df=in_df, out_path=out_path, model=model, prefix="NoMulti-")

    # Plot the sentiment progression for each story
    print("Plotting tale progressions ...")
    plot_sentiment_progression(in_df=in_df, out_path=out_path, sentiment_col=SENTIMENT_COLUMN)


    print("All done.")


