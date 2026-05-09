# Generator Data Quality Report

Tokenizer: `google/flan-t5-small`

## train

| Metric | Value |
|---|---:|
| rows | 10080 |
| broken_spacing | 48 |
| broken_spacing_rate | 0.004761904761904762 |
| target_lt_6_words | 23 |
| target_lt_6_words_rate | 0.002281746031746032 |
| target_lt_8_words | 167 |
| target_lt_8_words_rate | 0.016567460317460318 |
| target_starts_continuation | 2 |
| target_starts_continuation_rate | 0.0001984126984126984 |
| avg_query_words | 12.206051587301587 |
| avg_evidence_words | 68.06736111111111 |
| avg_target_words | 21.410714285714285 |
| avg_target_tokens | 29.36468253968254 |
| evidence_target_overlap_avg | 0.29686532692949724 |
| query_target_overlap_avg | 0.23908184738925392 |
| evidence_query_overlap_avg | 0.2595154786142226 |
| duplicate_query_target_pairs | 476 |

Target token lengths: `{'min': 10, 'p25': 18.0, 'median': 26.0, 'p75': 37.0, 'max': 96}`

## val

| Metric | Value |
|---|---:|
| rows | 960 |
| target_lt_6_words | 2 |
| target_lt_6_words_rate | 0.0020833333333333333 |
| target_lt_8_words | 17 |
| target_lt_8_words_rate | 0.017708333333333333 |
| target_starts_continuation | 1 |
| target_starts_continuation_rate | 0.0010416666666666667 |
| avg_query_words | 11.664583333333333 |
| avg_evidence_words | 54.338541666666664 |
| avg_target_words | 21.686458333333334 |
| avg_target_tokens | 29.551041666666666 |
| evidence_target_overlap_avg | 0.2453456216535052 |
| query_target_overlap_avg | 0.250070936388782 |
| evidence_query_overlap_avg | 0.2621380859914867 |
| duplicate_query_target_pairs | 10 |

Target token lengths: `{'min': 10, 'p25': 18.0, 'median': 25.0, 'p75': 38.0, 'max': 96}`

## test

| Metric | Value |
|---|---:|
| rows | 960 |
| target_lt_6_words | 1 |
| target_lt_6_words_rate | 0.0010416666666666667 |
| target_lt_8_words | 12 |
| target_lt_8_words_rate | 0.0125 |
| avg_query_words | 13.30625 |
| avg_evidence_words | 57.123958333333334 |
| avg_target_words | 23.994791666666668 |
| avg_target_tokens | 33.02916666666667 |
| evidence_target_overlap_avg | 0.3046818072687773 |
| query_target_overlap_avg | 0.2680557731290974 |
| evidence_query_overlap_avg | 0.28667061103231345 |
| duplicate_query_target_pairs | 37 |

Target token lengths: `{'min': 10, 'p25': 19.0, 'median': 29.0, 'p75': 42.0, 'max': 96}`
