# tokenomics-sdk

Emit OpenTelemetry GenAI spans with cost attribution to a [Tokenomics](../../README.md) server.

```python
from openai import OpenAI
from tokenomics_sdk import configure, track, wrap_openai

configure(endpoint="http://localhost:8000/v1/traces", project="checkout", environment="prod")
client = wrap_openai(OpenAI())

with track(feature="cart-summarizer", subject_id="cust_123", prompt_version="v4"):
    client.chat.completions.create(model="gpt-4o", messages=[...])
```

`track()` nests, so an inner block can add `prompt_version` without knowing what the
outer block set. Anthropic's exclusive token accounting is converted to the
OTel-inclusive form automatically — see `attributes.py`.
