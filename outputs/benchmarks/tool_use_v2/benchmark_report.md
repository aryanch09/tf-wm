# TF-WM tool_use Benchmark

Episodes: 30
Confidence: 95%
Bootstrap samples: 1000

## Metrics

| Metric | Mean | CI Low | CI High |
| --- | ---: | ---: | ---: |
| `success_rate` | 0.9667 | 0.9000 | 1.0000 |
| `steps_to_success` | 48.0000 | 48.0000 | 48.0000 |
| `average_return` | 14.0895 | 13.9953 | 14.1926 |
| `slip_rate` | 0.0396 | 0.0299 | 0.0500 |
| `camera_query_rate` | 0.1201 | 0.1153 | 0.1250 |
| `force_violation_rate` | 0.0000 | 0.0000 | 0.0000 |

## Claim Checks

| Metric | Target | Value | Result |
| --- | ---: | ---: | --- |
| `success_rate` | >= 0.7500 | 0.9667 | pass |
| `camera_query_rate` | <= 0.2000 | 0.1201 | pass |
| `slip_rate` | <= 0.0800 | 0.0396 | pass |
| `force_violation_rate` | <= 0.0100 | 0.0000 | pass |
