# DGS-Fabeln-1-SE: sentiment analysis on DGS-Fabeln-1

Scripts used to peforme sentiment analysis on DGS-Fabeln-1 and aggregate results, compute stats, plot, ... .


## Predict sentiment using online models

For GPT5 and Perplexity (Sonic), we used the Perplexity Pro GUI.
For Mistral, we used its web interface.

The prompts were composed by using the text in `Prompt-1_2.txt` followed by the sentences of the tale, one per line, separated by an empty line, closed by double-quotes.
For an example, see `data/1-DHUDI/1-DHUDI-Sentences-noid.txt`

Results were copied back and everything was verified and copy-pasted manually in 7 files `<id>-<talename>-Evaluated.csv`.


## Predict sentiments using GPTOSS:20B through ollama running locally

For GPTOSS20B we used a local ollama installation and the script `PredictSentiment.py`.

    python PredictSentiment.py -p Prompt-1_2.txt -i data/2-FrauHolle/2-FrauHolle-Sentences.csv -o 2-predictions.csv 

## Test: predict sentiments with the Guhr library

    python PredictSentimentGuhr.py -i data/1-DHUDI/1-DHUDI-Sentences.csv -o testguhr.csv
    python PredictSentimentGuhr.py -i data/2-FrauHolle/2-FrauHolle-Sentences.csv -o testguhr.csv

This library was dropped because it was predicting mostly "Neutral". Likely, this library is not appropriate for simplified German text.

# Merge all tales in a single file

    python MergeTales.py -d data -o AllTales-Evaluated.csv

Output header:

    Story,id,text_original,Sentiments-GPT5,Multi-GPT5,Sentiments-Perplexity,Multi-Perplexity,Sentiments-Mistral,Multi-Mistral,Sentiments-GPTOSS20B,Multi-GPTOSS20B
    

# Generate stats and plots for each tale

 ```
 python PlotTalesSentimentStats.py -i AllTales-Evaluated.csv -o Plots-GPT5 -m GPT5
 python PlotTalesSentimentStats.py -i AllTales-Evaluated.csv -o Plots-Perplexity -m Perplexity
 python PlotTalesSentimentStats.py -i AllTales-Evaluated.csv -o Plots-Mistral -m Mistral
 python PlotTalesSentimentStats.py -i AllTales-Evaluated.csv -o Plots-GPTOSS20B -m GPTOSS20B
 ```

# Aggregate sentiments

Aggregate the sentiment predictions of all models using "majority voting":

    python AggregateVotes.py -i AllTales-Evaluated.csv -o DGS-Fabeln-1-SE-Labels.csv

# Plotting

Plot the stats for the aggregated votes:

    python PlotTalesSentimentStats.py -i DGS-Fabeln-1-SE-Labels.csv -o Plots-Aggregated -m Aggregated

Plot the correlations between features and predicted sentiment:

    python PlotFeatureSentimentCorrelations.py -l DGS-Fabeln-1-SE-Labels.csv -f DGS-Fabeln-1-SE-MotionFeatures.csv -o Correlation

# Stats

Writing 'AllTales-Aggregated.csv' ...
Computing inter-annotator agreement stats...
{'n_items': 574, 'n_annotators': 13, 'alpha_nominal': 0.6942246909485004, 'alpha_ordinal': 0.7739714138884802}
All done.
