# Threshold Sweep Summary

Candidates evaluated: `392`
Strict feasible candidates: `98`

## Selected
`{'nearest_kb_similarity_threshold': 0.3, 'centroid_similarity_threshold': 0.25, 'lexical_gate_required': True, 'ticket_threshold': 0.4, 'Tool Decision Accuracy': 0.727, 'ANSWER F1': 0.8136986301369863, 'ANSWER Precision': 0.6906976744186046, 'ANSWER Recall': 0.99, 'TICKET F1': 0.6137931034482759, 'TICKET Precision': 0.9888888888888889, 'TICKET Recall': 0.445, 'REJECT F1': 0.352, 'REJECT Precision': 0.88, 'REJECT Recall': 0.22, 'confusion_matrix': {'ANSWER': {'ANSWER': 594, 'TICKET': 0, 'REJECT': 6}, 'TICKET': {'ANSWER': 111, 'TICKET': 89, 'REJECT': 0}, 'REJECT': {'ANSWER': 155, 'TICKET': 1, 'REJECT': 44}}, 'Macro-F1': 0.5931639111950874, 'FalseRejectRate': 0.0075, 'FalseAcceptRate': 0.78, 'OODAnswerRate': 0.775, 'TicketMissRate': 0.555}`

## Selection Policy
Primary constraints prioritize ANSWER recall, low false rejects, high reject precision, and citation-preserving behavior before Macro-F1.