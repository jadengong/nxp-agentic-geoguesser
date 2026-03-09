# nxp-agentic-geoguesser
NXP-CTRL Project 2

## Optional: External labels API

CLIP uses candidate labels to score images. By default a small built-in list is used. To use an external API instead, set:

```bash
LABELS_API_URL=https://your-api.com/labels
```

The URL must return JSON in one of these forms:

- A list of strings: `["a European city", "a beach", ...]`
- An object with a `labels` key: `{"labels": ["a European city", ...]}`

Labels are cached for 5 minutes. If the URL is not set or the request fails, the built-in default labels are used.
