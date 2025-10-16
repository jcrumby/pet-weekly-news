"""Gemini prompt templates specific to the Pet Connect workflow."""

LISTING_METADATA_PROMPT = """
You are reviewing the latest articles from {site_name} for the Pet Connect Weekly Newsletter.
Below is the page content captured on {today}.

Task:
- Identify news articles published within the last 7 days.
- For each article provide a JSON object with:
  {{
    "title": "...",
    "url": "...",
    "date": "...",        # ISO 8601 if possible, otherwise original string
    "description": "..." # short summary if available, otherwise leave empty string
  }}
- Return ONLY a JSON array with these objects; no code fences, text, or commentary.
- Exclude navigation, evergreen, or promotional links.

Listing content:
{markdown}
"""

RELEVANCE_PROMPT = """
You are evaluating pet industry news for inclusion in the Pet Connect Weekly Newsletter.
You will rank articles based on their relevance to Pet Connect USA 2025, a high-level business & innovation summit for the pet care ecosystem (brands, investors, tech, supply chain, vet / health).

Input format: a JSON array where each object looks like
{
  "title": "...",
  "description": "...",
  "url": "...",
  "date": "...",
  "source_section": "...",
  "sponsors": [... optional list ...]
}

Requirements:
1. Preserve every original field exactly as provided.
2. Append both:
   - "relevant": true/false
   - "relevance_score": integer 1-5

Scoring guide:
- 5 = Critical developments for the pet sector: major M&A or investments, large-scale product launches,
      facility expansions, regulatory changes, or strategic moves by leading pet brands directly shaping
      retail, manufacturing, or health.
- 4 = Strong industry impact: key partnerships, emerging tech or wellness trends, market data with clear
      business implications, or notable innovations in pet nutrition, digital commerce, or supply chain.
- 3 = Somewhat relevant: niche or regional initiatives, early-stage innovations, or consumer behavior
      stories with partial links to pet business strategy.
- 2 = Weak relevance: lifestyle or owner-focused content with limited commercial connection, minor retail
      or care stories without wider industry impact.
- 1 = Not relevant to Pet Connect: entertainment or general human-interest pieces unrelated
      to the pet care, retail, or manufacturing ecosystem.

Sponsors (if present) are supportive context only - do not inflate scores purely for a mention.

Output ONLY valid JSON (no code fences), mirroring the input array order. Each object must
include the new "relevant" and "relevance_score" fields.
"""

SUMMARY_PROMPT_TEMPLATE = """
You are creating concise newsletter summaries for the Pet Connect Weekly Newsletter.

Input: JSON array where each item has keys "title", "url", and "text" (article markdown).

Write a two-sentence summary (maximum 60 words) that highlights why the story matters to
pet retail, nutrition, supply chain, or services.

Return only JSON formatted like:
[
  {{"url": "...", "summary": "..."}},
  ...
]

Do not include any additional commentary or code fences.

Article data:
{articles_json}
"""

COMPANY_EXTRACTION_PROMPT = """
You are a research analyst. Extract distinct company or brand names mentioned in the
following Pet Connect newsletter summary. Return a comma-separated list with no duplicates
and no commentary.

Summary:
{summary}
"""


# OLD RELEVANCE PROMPT (for reference)
# Scoring guide:
# - 5 = Critical developments for pet retail/manufacturing/distribution: major product launches,
#       facility investments, supply-chain strategy, M&A, regulatory updates, category leadership.
# - 4 = Strong industry impact: notable partnerships, emerging trends in nutrition or wellness,
#       technology adoption across pet commerce, notable market data with clear implications.
# - 3 = Somewhat relevant: niche interest stories, localized initiatives, or softer consumer angles.
# - 2 = Weak relevance: light lifestyle content, generic human-interest pieces, limited connection
#       to the business of pets.
# - 1 = Not relevant to Pet Connect stakeholders.
