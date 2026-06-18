# tfm-media-violence-analysis
> Project by @karenL26 :elf:


## Pipeline

Example execution for one date range:

```
python -m src.acquisition.build_cdx_index --config configs/sources_argentina_mexico.yml --from-date 20150101 --to-date 20150103
python -m src.acquisition.build_homepage_candidates --config configs/sources_argentina_mexico.yml
python -m src.filtering.build_clean_candidates --config configs/sources_argentina_mexico.yml
python -m src.acquisition.build_article_texts --config configs/sources_argentina_mexico.yml
python -m src.acquisition.retry_failed_article_texts --config configs/sources_argentina_mexico.yml
python -m src.filtering.build_relevance_dataset --config configs/sources_argentina_mexico.yml
```bash