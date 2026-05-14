from pathlib import Path
import pandas as pd

MODELS = ["GPT5", "Perplexity", "Mistral", "GPTOSS20B"]

VOTE_COLUMNS = [("Sentiments-" + m) for m in MODELS]

AGGREGATION_POSTFIX = "Aggregated"


def _normalize_label(l: str) -> str:
    """Normalize the input label.
    - Normalize spaces
    - If a dash is present
      - Break on dash
      - strip spaces
      - sort alphabetically
      - rejoin with dash"""

    l = l.strip()  # remove surrounding spaces
    if '-' in l:
        # split by dash and remove surrounding spaces
        parts = [part.strip() for part in l.split('-')]
        # sort labels
        parts = sorted(parts)
        # rejoin
        l = '-'.join(parts)

    return l

def aggregate(in_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the sentiment predictions of all models using majority voting.
    For each row, determine the most common sentiment among all models and set that as the final sentiment.
    If there is a tie, set the sentiment to 'multi' and set the Multi column to 'yes'.
    """

    # Create a new DataFrame to store the aggregated results
    agg_df = in_df.copy()

    # Initialize the new columns
    #agg_df["Sentiment"] = ""
    #agg_df["Multi"] = "no"

    # Apply the normalization function to the labels
    for m in MODELS:
        sentiment_column_name = "Sentiments-" + m
        sentiment_column = agg_df[sentiment_column_name]
        print("Normalizing column ", sentiment_column_name, sentiment_column.dtype)

        # Apply the normalization function to the column labels
        sentiment_column = sentiment_column.apply(_normalize_label)
        agg_df[sentiment_column_name] = sentiment_column
        

    # Iterate over each row in the DataFrame
    sentiment_columns = [("Sentiments-" + m) for m in MODELS]

    for idx, row in agg_df.iterrows():
        # print("==> IDX", idx)
        sentiments = row[sentiment_columns]
        sentiments_tab = sentiments.value_counts()

        # Majority vote
        most_common_count = int(sentiments_tab.iloc[0])
        most_common_sentiment = sentiments_tab.index[0]

        if len(sentiments_tab) > 1:
            if sentiments_tab.iloc[1] == most_common_count:
                # There is a competing sentiment with the same count
                # Let's mark it in the result
                most_common_count = -1
                most_common_sentiment = 'mixed'

        # If the most common label is mixed
        if '-' in most_common_sentiment:
            most_common_count = -1
            most_common_sentiment = 'mixed'
                
    
        assert len(sentiments_tab) > 0
        # assert (len(sentiments_tab) == 1) and (most_common_sentiment == sentiments_tab.index[0]), f"len {len(sentiments_tab)}, '{most_common_sentiment}', '{sentiments_tab.index[0]}'"
    
        if len(sentiments_tab) > 1:
            # print(sentiments_tab, "\nLOC", sentiments_tab.index[0])
            # print("Most common", most_common_sentiment)
            pass
    
        if most_common_count > 0:
            # Take not of the most common sentiment
            agg_df.at[idx, "Sentiments-" + AGGREGATION_POSTFIX] = most_common_sentiment
            agg_df.at[idx, "Multi-" + AGGREGATION_POSTFIX] = "no"
        else:
            # There are at lest two sentiments with the same count. Set result to 'multi' and 'yes'
            agg_df.at[idx, "Sentiments-" + AGGREGATION_POSTFIX] = "multi"
            agg_df.at[idx, "Multi-" + AGGREGATION_POSTFIX] = "yes"

    return agg_df


def compute_agreement_lib(in_df: pd.DataFrame) -> dict:
    import krippendorff

    # Map in_df labels to unique integers
    unique_labels = set()
    for col in in_df.columns:
        unique_labels.update(in_df[col].dropna().unique())

    # Create a deterministic integer mapping (sorted for reproducibility)
    mapping_dict = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    print(mapping_dict)

    votes_ordinal_df = in_df.replace(mapping_dict)
    # print( votes_ordinal_df.head())

    votes_ordinal_values = votes_ordinal_df.values
    print("Ordinal values shape ", type(votes_ordinal_values), votes_ordinal_values.shape)
    # print(votes_ordinal_values)

    a_nominal = krippendorff.alpha(reliability_data=votes_ordinal_values.T, level_of_measurement='nominal')

    return {
        'n_items': len(in_df),
        'n_annotators': len(in_df.columns),
        'alpha_nominal': a_nominal,
        #'alpha_ordinal_str': a_ordinal_str,
        #'alpha_ordinal': a_ordinal
    }


#
# 
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Aggregate the sentiment predictions of all models')
    parser.add_argument('--input-csv', '-i', required=True, help='Input file path to the fairy tale CVS')
    parser.add_argument('--output-csv', '-o', required=True, help='Output file path to the CSV with the aggregated predictions')
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)

    print(f"Reading '{in_path}' ...")
    in_df = pd.read_csv(in_path)
    
    print("Aggregating ...")
    out_df = aggregate(in_df=in_df)

    print(f"Writing '{out_path}' ({len(out_df)} rows) ...")
    out_df.to_csv(out_path, header=True, index=False)

    #
    # Some stats
    print("==" * 40)
    print(out_df["Multi-Aggregated"].value_counts())
    for m in MODELS:
        print(out_df["Sentiments-" + m].value_counts())



    #
    # Computing inter-annotator agreement on original dataset ...
    print("==" * 40)
    print("Computing inter-annotator agreement stats before majority vote ...")
    agreement_info = compute_agreement_lib(out_df[VOTE_COLUMNS])
    print(agreement_info)

    #
    # Drop rows without agreement
    # agreed_df = out_df[out_df["Multi-Aggregated"] == "no"]
    print("==" * 40)
    agreed_df = out_df[out_df["Multi-Aggregated"] == "no"]
    # agreed_df = out_df
    print(f"AGREED VOTE: {len(agreed_df)}")


    #
    # Computing inter-annotator agreement after majority vote
    print("=" * 20 + " VOTES " + "=" * 20)

    votes_df = agreed_df[VOTE_COLUMNS]
    assert len(votes_df.columns) == len(MODELS)
    print(votes_df.head())

    for c in votes_df.columns:
        print(votes_df[c].value_counts())


    # Compute inter-annotator agreement
    print("Computing inter-annotator agreement stats after majority vote ...")
    agreement_info = compute_agreement_lib(votes_df)
    print(agreement_info)

    print("All done.")
