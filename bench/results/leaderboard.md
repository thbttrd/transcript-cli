# Benchmark leaderboard (tier 2, median)


## AMI

| Rank | Config | cpWER | WER | DER | Speaker-err | Runtime |
|------|--------|-------|-----|-----|-------------|---------|
| 1 | merge=hard_boundary, align=False, sortformer=very_high_lat, no_fallback=True, suppress_nst=False | 174.7 | 175.4 | 33.2 | 0.0 | 0.3s |
| 2 | merge=hard_boundary, align=False, sortformer=low_lat, no_fallback=True, suppress_nst=False | 174.8 | 175.4 | 35.3 | 0.0 | 0.3s |
| 3 | merge=hard_boundary, align=True, sortformer=low_lat, no_fallback=True, suppress_nst=False | 175.4 | 175.4 | 35.0 | 0.0 | 290.6s |
| 4 | merge=hard_boundary, align=True, sortformer=very_high_lat, no_fallback=True, suppress_nst=False | 175.5 | 175.4 | 33.7 | 0.0 | 239.4s |
| 5 | merge=prob_based, align=True, sortformer=very_high_lat, no_fallback=True, suppress_nst=False | 232.8 | 175.4 | 80.6 | 44.4 | 12.4s |
| 6 | merge=prob_based, align=False, sortformer=very_high_lat, no_fallback=True, suppress_nst=False | 232.8 | 175.4 | 80.6 | 44.4 | 0.1s |
| 7 | merge=prob_based, align=True, sortformer=low_lat, no_fallback=True, suppress_nst=False | 232.8 | 175.4 | 80.6 | 44.4 | 289.4s |
| 8 | merge=prob_based, align=False, sortformer=low_lat, no_fallback=True, suppress_nst=False | 232.8 | 175.4 | 80.6 | 44.4 | 0.1s |

## SUMM-RE

| Rank | Config | cpWER | WER | DER | Speaker-err | Runtime |
|------|--------|-------|-----|-----|-------------|---------|
| 1 | merge=hard_boundary, align=True, sortformer=very_high_lat, no_fallback=True, suppress_nst=False | 44.4 | 39.0 | 39.5 | 5.6 | 179.3s |
| 2 | merge=hard_boundary, align=True, sortformer=low_lat, no_fallback=True, suppress_nst=False | 45.0 | 39.0 | 39.5 | 5.3 | 290.8s |
| 3 | merge=hard_boundary, align=False, sortformer=very_high_lat, no_fallback=True, suppress_nst=False | 45.8 | 39.0 | 38.4 | 6.7 | 0.2s |
| 4 | merge=hard_boundary, align=False, sortformer=low_lat, no_fallback=True, suppress_nst=False | 46.3 | 39.0 | 39.5 | 7.4 | 0.2s |
| 5 | merge=prob_based, align=False, sortformer=very_high_lat, no_fallback=True, suppress_nst=False | 112.8 | 39.0 | 87.3 | 70.3 | 0.1s |
| 6 | merge=prob_based, align=False, sortformer=low_lat, no_fallback=True, suppress_nst=False | 112.8 | 39.0 | 87.3 | 70.3 | 0.1s |
| 7 | merge=prob_based, align=True, sortformer=very_high_lat, no_fallback=True, suppress_nst=False | 112.8 | 39.0 | 86.2 | 70.4 | 12.1s |
| 8 | merge=prob_based, align=True, sortformer=low_lat, no_fallback=True, suppress_nst=False | 112.8 | 39.0 | 86.2 | 70.4 | 290.3s |
