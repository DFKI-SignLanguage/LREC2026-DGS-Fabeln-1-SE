# Code for the creation of the DGS-Fabeln-1-SE dataset

This repository accompanies the paper presenting the DGS-Fabeln-1-SE dataset, presented the the LREC 2026 conference in Palma, Spain, May 2026.

Folders:
* `SentimentAnalysisFromText` - The code analysing the text of the DGS-Fabeln-1 dataset to produce sentiment labels.
* `VideoDataPreperation` - The code extracting raw data from videos using MediaPipe (for both body landmarks and facial blndshapes) and computing extra features (velocities, accellerations, distances, means, standard deviations, peak frequency, ...) and producing the feature vectors for each video segment.
* `TrainingAndEvaluation` - The code training and evaluating sentiment prediction from the video features.


## Reference

Nunnari, F., Jain, S. and Gebhard, P. (2026) “Sentiment Analysis of German Sign Language Fairy Tales.” The Fifteenth Language Resources and Evaluation Conference (LREC 2026), Palma, Mallorca, Spain, pp. 9525–9534. Available at: https://doi.org/10.63317/3cyfzw6vs9oe.

```
@inproceedings{nunnari_sentiment_2026,
	address = {Palma, Mallorca, Spain},
	title = {Sentiment {Analysis} of {German} {Sign} {Language} {Fairy} {Tales}},
	url = {https://lrec.elra.info/lrec2026-main-748},
	doi = {10.63317/3cyfzw6vs9oe},
	urldate = {2026-05-11},
	author = {Nunnari, Fabrizio and Jain, Siddhant and Gebhard, Patrick},
	month = may,
	year = {2026},
	pages = {9525--9534},
}
```
