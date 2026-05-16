# TF-WM blind_retrieve Benchmark

Episodes: 30
Confidence: 95%
Bootstrap samples: 1000

## Metrics

| Metric | Mean | CI Low | CI High |
| --- | ---: | ---: | ---: |
| `success_rate` | 1.0000 | 1.0000 | 1.0000 |
| `steps_to_success` | 48.0000 | 48.0000 | 48.0000 |
| `average_return` | 14.0209 | 13.9491 | 14.0914 |
| `slip_rate` | 0.0333 | 0.0264 | 0.0410 |
| `camera_query_rate` | 0.1153 | 0.1118 | 0.1187 |
| `force_violation_rate` | 0.0000 | 0.0000 | 0.0000 |

## Claim Checks

| Metric | Target | Value | Result |
| --- | ---: | ---: | --- |
| `success_rate` | >= 0.7500 | 1.0000 | pass |
| `camera_query_rate` | <= 0.2000 | 0.1153 | pass |
| `slip_rate` | <= 0.0800 | 0.0333 | pass |
| `force_violation_rate` | <= 0.0100 | 0.0000 | pass |
