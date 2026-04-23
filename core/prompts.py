"""
Prompt templates for the 202x Headlines LLM candidate filter.

Kept in a separate module from filter.py so the prompt text can be read,
tested, and revised without touching the scoring logic.
"""


def build_filter_prompt(headline: str, excerpt: str | None) -> str:
    """
    Build and return the full prompt string to send to the Ollama model.

    The prompt instructs the model to act as an editorial filter for
    202x Headlines — a publication that curates real news headlines so
    absurd they function as accidental satire, specifically critiquing
    late-stage capitalism by holding up a mirror to how the ideology
    shapes human behaviour.

    The model is asked to evaluate five criteria and return a structured
    JSON response with a numeric score, a brief written rationale, and
    a list of applicable flags.

    Parameters
    ----------
    headline : The article headline to evaluate.
    excerpt  : The article excerpt (lede or first paragraph), or None if
               not available.

    Returns
    -------
    str
        The complete prompt string ready to send to the model.
    """
    # Format the excerpt block — omit it cleanly if none was scraped.
    if excerpt:
        excerpt_block = f"Excerpt: {excerpt}"
    else:
        excerpt_block = "Excerpt: (none available)"

    return f"""You are an editorial filter for a publication called 202x Headlines.

202x Headlines curates real news headlines that are so absurd they function
as accidental satire. The publication's focus is late-stage capitalism: it
holds up a mirror to how this ideology makes us behave. Every headline must
be real reporting — not parody, not opinion, not lifestyle content.

Evaluate the article below on these five criteria:

1. ABSURD HEADLINE — Is the headline absurd on its face?
2. CAPITALISM-RELEVANT — On reflection, does it reveal something true about
   late-stage capitalism or the current historical moment?
3. DARK WIT — Does it have dark wit? Not just depressing or cruel — there
   must be some quality of the grotesque or the ironic.
4. REAL REPORTING — Is this factual news coverage, not parody, satire,
   opinion, or lifestyle content?
5. CLASSIC REGISTER — Does it fit one of the registers that work especially
   well for this publication?
     - Animal or natural world behaviour warped by human excess
     - Economic logic producing grotesque outcomes
     - Technology solving a problem it created

---

Article to evaluate:

Headline: {headline}
{excerpt_block}

---

Respond with valid JSON only. No preamble. No explanation outside the JSON.
No markdown code fences. Your entire response must be a single JSON object
with exactly these three keys:

  "score"     — a float from 0.0 to 1.0 representing overall fit
  "rationale" — one or two sentences explaining your judgement
  "flags"     — a list of zero or more strings from this exact set:
                "absurd-headline", "capitalism-relevant", "dark-wit",
                "real-reporting", "animal-warped", "economic-grotesque",
                "tech-irony"

Example of a valid response (do not copy this — evaluate the actual article):
{{"score": 0.85, "rationale": "Exemplifies economic logic producing a grotesque outcome with genuine dark wit.", "flags": ["absurd-headline", "capitalism-relevant", "dark-wit", "economic-grotesque"]}}

Your response:"""
