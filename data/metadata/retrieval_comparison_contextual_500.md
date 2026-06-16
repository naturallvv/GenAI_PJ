| name | qa_count | retrieval_mode | hit_at_1 | hit_at_3 | hit_at_5 | mrr | dense_weight | candidate_k | elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline Dense | 500 | dense | 0.9000 | 0.9880 | 0.9980 | 0.9444 | - | 50 | 10.8580 |
| Contextual Dense | 500 | dense | 0.9020 | 0.9860 | 0.9980 | 0.9435 | - | 50 | 10.0440 |
| Contextual Hybrid | 500 | hybrid | 0.9060 | 0.9920 | 0.9980 | 0.9478 | 0.9000 | 50 | 18.5220 |
| Contextual Hybrid + Reranker | - | - | - | - | - | - | - | - | - |
