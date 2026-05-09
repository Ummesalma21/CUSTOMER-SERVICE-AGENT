# Generator Batch Debug

Model/tokenizer: `google/flan-t5-small`
Batch size checked: `4`
Max input length: `512`
Max target length: `96`

## Train
- Rows checked: `24`
- All ignored labels: `0`
- All ignored label rate: `0.0`
- Average non-ignored label tokens: `27.5`
- Average label -100 rate: `0.7135416666666669`
- Invalid input ID examples: `0`

## Validation
- Rows checked: `24`
- All ignored labels: `0`
- All ignored label rate: `0.0`
- Average non-ignored label tokens: `33.375`
- Average label -100 rate: `0.6523437499999999`
- Invalid input ID examples: `0`

Detailed decoded examples are saved to `outputs/reports/generator_batch_debug_examples.jsonl`.