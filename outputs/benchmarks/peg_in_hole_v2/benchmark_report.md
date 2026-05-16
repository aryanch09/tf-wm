# TF-WM peg_in_hole Benchmark

Episodes: 30
Confidence: 95%
Bootstrap samples: 1000

## Metrics

| Metric | Mean | CI Low | CI High |
| --- | ---: | ---: | ---: |
| `success_rate` | 0.9667 | 0.9000 | 1.0000 |
| `steps_to_success` | 48.0000 | 48.0000 | 48.0000 |
| `average_return` | 13.9963 | 13.9180 | 14.0790 |
| `slip_rate` | 0.0375 | 0.0285 | 0.0465 |
| `camera_query_rate` | 0.1201 | 0.1146 | 0.1271 |
| `force_violation_rate` | 0.0000 | 0.0000 | 0.0000 |

## Claim Checks

| Metric | Target | Value | Result |
| --- | ---: | ---: | --- |
| `success_rate` | >= 0.7500 | 0.9667 | pass |
| `camera_query_rate` | <= 0.2000 | 0.1201 | pass |
| `slip_rate` | <= 0.0800 | 0.0375 | pass |
| `force_violation_rate` | <= 0.0100 | 0.0000 | pass |
