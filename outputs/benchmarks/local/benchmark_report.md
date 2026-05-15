# TF-WM Local Benchmark

Episodes: 12
Confidence: 95%
Bootstrap samples: 100

## Metrics

| Metric | Mean | CI Low | CI High |
| --- | ---: | ---: | ---: |
| `success_rate` | 1.0000 | 1.0000 | 1.0000 |
| `steps_to_success` | 48.0000 | 48.0000 | 48.0000 |
| `average_return` | 14.1827 | 14.0269 | 14.3412 |
| `slip_rate` | 0.0278 | 0.0130 | 0.0426 |
| `camera_query_rate` | 0.1111 | 0.1059 | 0.1181 |
| `force_violation_rate` | 0.0000 | 0.0000 | 0.0000 |

## Claim Checks

| Metric | Target | Value | Result |
| --- | ---: | ---: | --- |
| `success_rate` | >= 0.7500 | 1.0000 | pass |
| `camera_query_rate` | <= 0.2000 | 0.1111 | pass |
| `slip_rate` | <= 0.0800 | 0.0278 | pass |
| `force_violation_rate` | <= 0.0100 | 0.0000 | pass |
