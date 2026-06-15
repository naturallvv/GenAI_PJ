| name | qa_count | retrieval_mode | hit_at_1 | hit_at_3 | hit_at_5 | mrr | dense_weight | candidate_k | elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline Dense | 100 | - | 0.8800 | 0.9700 | 1.0000 | 0.9292 | - | - | 3.5890 |
| Contextual Dense | 100 | dense | 0.8800 | 0.9600 | 1.0000 | 0.9262 | - | 50 | 3.6570 |
| Contextual Hybrid | 100 | hybrid | 0.8800 | 0.9800 | 1.0000 | 0.9300 | 0.9000 | 50 | 4.1070 |
| Contextual Hybrid + Reranker | - | - | - | - | - | - | - | - | - |
