# Final Threshold Sweep

Strict feasible candidates: `0`
Selection note: No candidate satisfied all constraints by thresholding only; selected best safety-preserving config. ESA did not exceed current 0.5300; AQS did not exceed current 0.6733; selected config changes more than 2% of answerable rows away from ANSWER

## Selected Config
`{'answerability_threshold': 0.45, 'esa_accept_threshold': 0.0, 'ticket_threshold': 0.3, 'reject_threshold': 0.25, 'nearest_kb_similarity_threshold': 0.25, 'centroid_similarity_threshold': 0.3, 'fallback_score_threshold': 0.65, 'Recall@1': 0.156, 'Recall@5': 0.362, 'MRR@10': 0.23196666666666663, 'EvidenceHit@5': 0.362, 'CitationPrecision': 1.0, 'GroundedAnswerRate': 1.0, 'UnsupportedClaimRate': 0.0, 'ESA': 0.472, 'AQS': 0.6246666666666666, 'Tool Decision Accuracy': 0.689, 'ANSWER F1': 0.8088235294117647, 'ANSWER Precision': 0.7932692307692307, 'ANSWER Recall': 0.825, 'TICKET F1': 0.4990176817288802, 'TICKET Precision': 0.4110032362459547, 'TICKET Recall': 0.635, 'REJECT F1': 0.50187265917603, 'REJECT Precision': 1.0, 'REJECT Recall': 0.335, 'confusion_matrix': {'ANSWER': {'ANSWER': 495, 'TICKET': 105, 'REJECT': 0}, 'TICKET': {'ANSWER': 73, 'TICKET': 127, 'REJECT': 0}, 'REJECT': {'ANSWER': 56, 'TICKET': 77, 'REJECT': 67}}, 'Macro-F1': 0.603237956772225, 'FalseRejectRate': 0.0, 'FalseAcceptRate': 0.665, 'OODAnswerRate': 0.28, 'TicketMissRate': 0.365, 'UnsupportedAnswerRate': 0.3225, 'UnsupportedAnswerCount': 129, 'UnsupportedAnswerPreventionCount': 271, 'UnsupportedAnswerPreventionRate': 0.6775, 'SafeActionRate': 0.6775, 'FalseRejectOnAnswerableRate': 0.0, 'FalseTicketOnAnswerableRate': 0.175, 'FalseNonAnswerOnAnswerableRate': 0.175}`

## Updated Proposed Metrics After Thresholding

| Metric | Current | Tuned |
|---|---:|---:|
| Recall@5 | 0.3620 | 0.3620 |
| ESA | 0.5300 | 0.4720 |
| AQS | 0.6733 | 0.6247 |
| UnsupportedAnswerRate | 0.5525 | 0.3225 |
| TicketMissRate | 0.5550 | 0.3650 |
| OODAnswerRate | 0.5500 | 0.2800 |
| FalseRejectOnAnswerableRate | 0.0000 | 0.0000 |
| FalseNonAnswerOnAnswerableRate | 0.1333 | 0.1750 |
| Macro-F1 | 0.5772 | 0.6032 |

No models were retrained. Reranker remains off.