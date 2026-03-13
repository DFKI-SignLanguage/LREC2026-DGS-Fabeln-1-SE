from pathlib import Path

import ollama
# from ollama import generate

import pandas as pd

# MODEL_NAME = "llama3.1:8b"
MODEL_NAME = "gpt-oss:20b"

def main(base_prompt:str, in_file: Path, out_file: Path):
    in_df: pd.DataFrame = pd.read_csv(in_file)

    sentences = in_df["text_original"]
    print(f"Loaded {len(sentences)} sentences.")

    # DEBUG --  Use only some sentences
    # sentences = sentences[:5]

    # Convert the list of sentences into a list of strings each surrounded by double quotes and wiht each line.
    sentences = [f'"{sentence}"' for sentence in sentences]
    all_sentences = "\n\n".join(sentences)

    #
    # Compose the prompt
    final_prompt = base_prompt\
        + "\n"\
        + all_sentences

    print("======= FINAL PROMPT =======")
    print(final_prompt)
    print("============================")

    #
    # Invoke ollama
    # Provide the prompt and read the answer
    print("Generating...")
    gen_response: ollama.GenerateResponse = ollama.generate(model=MODEL_NAME, prompt=final_prompt)
    print("========= RESPO ========")
    print(gen_response.response)
    print("==========================")
    print(f"{gen_response.done} - {gen_response.done_reason}")

    print("==== DURATIONS =====")
    durations = [("Load", gen_response.load_duration),
                 ("Prompt Eval", gen_response.prompt_eval_duration),
                 ("Eval", gen_response.eval_duration),
                 ("Tot", gen_response.total_duration)]
    for label, d in durations:
        # From nanoseconds to seconds
        d = d / 1000 / 1000 / 1000
        print(f"{label}: {d} secs")
    print("====================")

    #
    # Format the response        
    # print("Response", type(response), len(response), response.keys())
    response = gen_response.response

    # Convert the response (a 3-column CSV formatted as a single string) to a pandas dataframe.
    response_list = response.strip().split('\n')
    columns_str = response_list[0]
    columns = columns_str.split(',')
    # Be sure to remove double quotes from the header
    columns = [c.replace('"', '') for c in columns]
    print("COLS", columns)
    if not "Sentiments" in columns:
        raise ValueError(f"Response does not contain Sentiments column: {columns}")
    if not "Multi" in columns:
        raise ValueError(f"Response does not contain Multi column: {columns}")

    df_data = [row.split(',') for row in response_list[1:]]
    df_sentiment = pd.DataFrame(df_data, columns=columns)

    # Append the columns "Sentiment" and "Multi" to the input dataframe
    out_df = in_df.copy()
    out_df["Sentiments"] = df_sentiment["Sentiments"]
    out_df["Multi"] = df_sentiment["Multi"]

    # Print some info about the output dataframe
    print("Output dataframe size: ", len(out_df))

    # Write the response in the output file
    out_path = Path(out_file).with_suffix(".csv")
    out_df.to_csv(out_path, header=True, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Ollama Script')
    parser.add_argument('--base-prompt', '-p', type=str, required=True,
                        help='Base prompt to use for ollama')
    parser.add_argument('--input', '-i', required=True, help='Input file path to the fairy tale CVS')
    parser.add_argument('--out', '-o', required=True, help='Output file path to the CSV with the predictions')
    args = parser.parse_args()

    base_prompt_path = Path(args.base_prompt)
    in_path = Path(args.input)
    out_path = Path(args.out)

    if not base_prompt_path.exists():
        raise ValueError(f"Base prompt file {base_prompt_path} does not exist")

    if not in_path.exists():
        raise ValueError(f"Input file {in_path} does not exist")
    
    # Read the file base_prompt_path in a string
    prompt_str = base_prompt_path.read_text()
    print("===== Base promt: =====")
    print(prompt_str)
    print("=======================")

    main(base_prompt=prompt_str, in_file=in_path, out_file=out_path)

    print("All done.")
