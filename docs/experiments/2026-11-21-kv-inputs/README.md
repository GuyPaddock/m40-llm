# November KV input freeze

Two separate CPU preparations produced the same 48 cases byte for byte using the pinned Qwen Q8_0 file. The preparer enforces the model SHA-256. The short case uses raw ChatML. In each long case the complete task ends outside the recent 1024 tokens. lengths.tsv records case ID, actual prompt length, task-end token count, archive-note count and output budget. The manifest retains the exact text, IDs, scoring and preparer identities.

To replay, run the release kv_quality example with prepare-november, --model and --output pointing to a new file. Decompress manifest.json.gz and compare both bytes and manifest.sha256. Eight CPU tests executed. These are input controls; model outputs have not been evaluated.
