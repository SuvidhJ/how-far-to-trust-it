# Data licence and attribution

The code in this repository is MIT (see `LICENSE`). The data is not.

Every tweet in this repository is derived from **Customer Support on Twitter**
(Thought Vector, on Kaggle), which is published under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/):
<https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter>.

That licence covers, and applies to, every file below:

| file | what it is |
| --- | --- |
| `data/golden/*.jsonl`, `data/golden/labels/*` | 220 hand-labelled customer messages, with my intent labels |
| `data/processed/americanair_index.jsonl` | 3,000 past customer/brand exchanges used for retrieval |
| `data/processed/americanair_demo_cases.jsonl` | the same messages with the agent's replayed output |
| `data/golden/americanair_judge_*` | drafts and evidence quoted for the judge study |

Re-use of those files must keep the attribution, stay non-commercial, and carry
the same licence. The intent labels are my own work and are released on the same
terms, so that the labelled set travels as one thing.

The raw dump (`data/raw/`) is **not** in this repository. `scripts/fetch_data.py`
downloads it from Kaggle with your own credentials.

The tweets were anonymised by the dataset's authors: user handles are numeric ids,
and this project masks flight numbers, URLs, phone numbers and booking references
again before any text reaches a model.
