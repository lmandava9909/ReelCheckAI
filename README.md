# ReelCheck AI

ReelCheck AI extracts the caption, spoken audio, and visible text from a public Instagram post, identifies externally verifiable claims, retrieves independent evidence, and produces an explainable ReelCheck Brief.

## Demonstrated reference use case

Reference Reel: `https://www.instagram.com/reel/Dbn0wRps876/`

The original workflow detected a meaning-changing ASR issue (`Ben Chinese AI equipment`) and corrected it to `ban Chinese AI equipment` using caption and metadata context while preserving the raw transcript and audit record. Historical validation target: transcript quality 0.85, two checked claims, overall assessment Mostly Supported, risk score 41/100. The fixture is a validation target, not hardcoded live output.

## Capability

- Public `/reel/`, `/p/`, and `/tv/` URL handling
- Reel, video, image, and carousel asset enumeration through Instaloader where public access permits
- Caption/manual fallback
- Whisper transcription for video assets
- OCR attempt for image assets when Tesseract is available
- Phrase-level transcript correction and Trust Audit
- Databricks Foundation Model claim extraction with Pydantic validation
- Tavily evidence search
- Authority and claim-relationship scoring
- Deterministic verdict guardrails
- User-centered Streamlit experience
- Vector-ready Delta semantic-chunk schema

## Limits

Instagram can block automated extraction and private posts are unsupported. OCR depends on Tesseract availability and otherwise falls back gracefully. Production Vector Search index creation remains a workspace-specific deployment step. Results are informational and are not financial, legal, medical, or safety advice.

## Databricks App resources

The supplied `app.yaml` expects:
- Lakebase resource key: `database`
- Add Tavily as an app secret/environment variable named `TAVILY_API_KEY`

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
