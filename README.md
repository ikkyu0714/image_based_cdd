# Cultural Difference Detector

Japanese and English image search results are compared using
visual feature vectors extracted by VGG16.

The project is a cleaned and modularized version of the original
cultural difference detection scripts.

## Pipeline

The processing pipeline is:

1. Read Japanese and English keywords/hypernyms from a Synset Excel file.
2. Generate image search queries.
3. Download images for Japanese and English queries.
4. Extract VGG16 feature vectors.
5. Generate representative feature vectors.
6. Compare Japanese and English vectors using cosine similarity.
7. Save the results to CSV.

## Directory Structure

```text
cultural_difference_clean/
├── main.py
├── config.yaml
├── requirements.txt
├── README.md
└── cultural_difference/
    ├── __init__.py
    ├── config.py
    ├── synset_repository.py
    ├── image_downloader.py
    ├── feature_extractor.py
    ├── aggregators.py
    ├── metrics.py
    ├── detector.py
    └── result_writer.py
