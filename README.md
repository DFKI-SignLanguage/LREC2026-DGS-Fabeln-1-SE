# Code for the creation of the DGS-Fabeln-1-SE dataset

This repository accompanies the paper presenting the DGS-Fabeln-1-SE dataset, presented the the LREC 2026 conference in Palma, Spain, May 2026.

Folders:
* `SentimentAnalysisFromText` - The code analysing the text of the DGS-Fabeln-1 dataset to produce sentiment labels.
* `VideoDataPreperation` - The code extracting raw data from videos using MediaPipe (for both body landmarks and facial blndshapes) and computing extra features (velocities, accellerations, distances, means, standard deviations, peak frequency, ...) and producing the feature vectors for each video segment.
* `TrainingAndEvaluation` - The code training and evaluating sentiment prediction from the video features.


## Reference

```
Nunnari, Fabrizio, Siddhant Jain, Patrick Gebhard. 2026. “Sentiment Analysis of German Sign Language Fairy Tales.”
Proceedings of the fifteenth biennial Language Resources and Evaluation Conference (LREC 2026) (Palma, Mallorca, Spain), May.
URL: https://lrec.elra.info/lrec2026-main-748
```
