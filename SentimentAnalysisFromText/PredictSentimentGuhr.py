from pathlib import Path

# https://huggingface.co/oliverguhr/german-sentiment-bert
from germansentiment import SentimentModel



import pandas as pd


def main(in_file: Path, out_file: Path):
    in_df: pd.DataFrame = pd.read_csv(in_file)

    sentences = in_df["text_original"]
    print(f"Loaded {len(sentences)} sentences.")

    # DEBUG --  Use only some sentences
    # sentences = sentences[:5]

    # Convert the list of sentences into a list of strings each surrounded by double quotes and wiht each line.
    sentences = [f'"{sentence}"' for sentence in sentences]

    # all_sentences = "\n\n".join(sentences)

    model = SentimentModel()

       
    result = model.predict_sentiment(sentences)
    print("Result size", len(result))
    print(result)


    
    # # Convert the response (a 3-column CSV formatted as a single string) to a pandas dataframe.
    # response_list = response.strip().split('\n')
    # columns_str = response_list[0]
    # columns = columns_str.split(',')
    # # Be sure to remove double quotes from the header
    # columns = [c.replace('"', '') for c in columns]
    # print("COLS", columns)
    # if not "Sentiments" in columns:
    #     raise ValueError(f"Response does not contain Sentiments column: {columns}")
    # if not "Multi" in columns:
    #     raise ValueError(f"Response does not contain Multi column: {columns}")

    # df_data = [row.split(',') for row in response_list[1:]]
    # df_sentiment = pd.DataFrame(df_data, columns=columns)

    # # Append the columns "Sentiment" and "Multi" to the input dataframe
    out_df = in_df.copy()
    out_df["Sentiments"] = result
    # out_df["Multi"] = df_sentiment["Multi"]

    # # Print some info about the output dataframe
    # print("Output dataframe size: ", len(out_df))

    # Write the response in the output file
    out_path = Path(out_file).with_suffix(".csv")
    out_df.to_csv(out_path, header=True, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Ollama Script')
    parser.add_argument('--input', '-i', required=True, help='Input file path to the fairy tale CVS')
    parser.add_argument('--out', '-o', required=True, help='Output file path to the CSV with the predictions')
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)

    # Check that the input file exists
    if not in_path.exists():
        raise ValueError(f"Input file {in_path} does not exist")
    
    # 
    main(in_file=in_path, out_file=out_path)

    print("All done.")
