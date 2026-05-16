# TF-WM tool_use Benchmark

Episodes: 30
Confidence: 95%
Bootstrap samples: 1000

## Metrics

| Metric | Mean | CI Low | CI High |
| --- | ---: | ---: | ---: |
| `success_rate` | 1.0000 | 1.0000 | 1.0000 |
| `steps_to_success` | 48.0000 | 48.0000 | 48.0000 |
| `average_return` | 14.1401 | 14.0643 | 14.2133 |
| `slip_rate` | 0.0326 | 0.0250 | 0.0403 |
| `camera_query_rate` | 0.1160 | 0.1118 | 0.1201 |
| `force_violation_rate` | 0.0000 | 0.0000 | 0.0000 |

## Claim Checks

| Metric | Target | Value | Result |
| --- | ---: | ---: | --- |
| `success_rate` | >= 0.7500 | 1.0000 | pass |
| `camera_query_rate` | <= 0.2000 | 0.1160 | pass |
| `slip_rate` | <= 0.0800 | 0.0326 | pass |
| `force_violation_rate` | <= 0.0100 | 0.0000 | pass |
