# Abacus-409: how many field changes reproduce the wrong answer?

397 released failures with per-field gold labels (12 rows have no vignette).

| fields changed | n | % |
|---|---|---|
| 1 field | 293 | 73.8 |
| 2 fields | 45 | 11.3 |
| >2 fields | 35 | 8.8 |
| no numeric answer | 24 | 6.0 |

Control: 246 of 373 numeric failures (66.0%) would also be reproduced by one field change if the error had the same size in the opposite direction.

| calculator | n | observed 1 field | mirrored 1 field |
|---|---|---|---|
| ariscat | 50 | 29 (58%) | 0 (0%) |
| caprini | 45 | 36 (80%) | 35 (78%) |
| cci | 47 | 41 (87%) | 36 (77%) |
| gad7 | 20 | 14 (70%) | 13 (65%) |
| hasbled | 30 | 24 (80%) | 24 (80%) |
| meldna | 16 | 8 (50%) | 4 (25%) |
| nihss | 45 | 45 (100%) | 44 (98%) |
| psiport | 34 | 14 (41%) | 13 (38%) |
| sofa | 44 | 41 (93%) | 41 (93%) |
| wellsdvt | 42 | 41 (98%) | 36 (86%) |

## By annotated error class

| error class | n | 1 field | 2 fields | >2 fields | no numeric answer |
|---|---|---|---|---|---|
| Interpretation Error | 142 | 126 (89%) | 7 (5%) | 9 (6%) | 0 (0%) |
| Assignment Error | 89 | 63 (71%) | 19 (21%) | 7 (8%) | 0 (0%) |
| Incorrect Formula | 80 | 43 (54%) | 7 (9%) | 10 (12%) | 20 (25%) |
| Incorrect Criteria | 41 | 33 (80%) | 4 (10%) | 4 (10%) | 0 (0%) |
| Calculation Error | 27 | 18 (67%) | 5 (19%) | 0 (0%) | 4 (15%) |
| Indeterminate | 10 | 4 (40%) | 1 (10%) | 5 (50%) | 0 (0%) |
| Incorrect Reporting | 5 | 3 (60%) | 2 (40%) | 0 (0%) | 0 (0%) |
| Software Error/Other | 3 | 3 (100%) | 0 (0%) | 0 (0%) | 0 (0%) |

## Fields most often implicated in single-field attributions

| calculator | field | n |
|---|---|---|
| nihss | motor_leg_left | 36 |
| nihss | motor_arm_left | 36 |
| nihss | facial_palsy | 33 |
| nihss | motor_arm_right | 32 |
| nihss | motor_leg_right | 31 |
| nihss | visual | 27 |
| sofa | platelets | 26 |
| nihss | sensory | 25 |
| nihss | questions | 25 |
| cci | age | 24 |
| nihss | limb_ataxia | 23 |
| wellsdvt | previous_dvt | 23 |
| sofa | glasgow_coma_scale | 23 |
| sofa | creatinine | 23 |
| nihss | extinction | 22 |

A single-field attribution means some one-field misreading yields exactly the model's answer;
it is consistent with, not proof of, an extraction error. In additive scores several fields can
explain the same shift, and each of them is counted in the field table.
