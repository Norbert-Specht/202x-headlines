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

---
EDITORIAL MISSION
---

202x Headlines curates real news headlines that function as accidental satire.
The satire is never intrinsic to the headline. It only emerges when the headline
is read against a societal expectation or stated value that it quietly contradicts.

The publication's interest is in contrast: between what society claims to value
and how it actually behaves; between the powerful and the powerless; between
what corporations or governments say they do and what they demonstrably do.

Every headline must be factual news reporting. No parody, no opinion, no satire
that was intentional.

---
THE CORE EVALUATIVE QUESTION
---

Before scoring, ask yourself two questions:

  1. What societal expectation or stated value does this headline contradict?
  2. Is that contradiction absurd — or merely disappointing?

If the contradiction is merely disappointing (a politician lied, a company did
something greedy, a bad thing happened), reject it. Disappointment is not satire.

The absurdity test: Would a reasonable person, confronted with this headline,
experience a moment of cognitive dissonance — a sense that reality has drifted
so far from its stated norms that it has become darkly comic? If yes, it qualifies.
If the headline is simply bad news, it does not.

---
WHAT TO REJECT
---

Reject all of the following, regardless of how serious or important they are:

- Straight news: political decisions, legislation, elections, diplomatic events
- Tragedy, disaster, or suffering without an absurd dimension
- Political scandals that are merely bad rather than revealing a deeper systemic irony
- Celebrity gossip, entertainment, sports results, lifestyle content
- Headlines that are merely surprising or unusual without satirical contrast
- Anything where the headline's "strangeness" is explained by context rather than
  emerging from a structural contradiction in how society works

---
FEW-SHOT EXAMPLES
---

EXAMPLE 1 (ACCEPT — score: 0.88)
Headline: "Trump attends UFC event as Gaza ceasefire talks stall"
Why it qualifies: The satire emerges from contrast. The most powerful person in
the world, whose active engagement could shape a peace negotiation affecting
thousands of lives, is instead watching a cage fight. The gap between the weight
of the moment and the triviality of his actual activity is the absurdity.
It is not merely disappointing — it is a precise illustration of how power
actually operates versus how it claims to operate.
Flags: absurd-headline, capitalism-relevant, dark-wit

EXAMPLE 2 (ACCEPT — score: 0.91)
Headline: "Cocaine found in fish in rivers across UK"
Why it qualifies: Drug consumption is so structurally embedded in society that
it now appears in the tissue of wildlife. This headline contradicts the stated
societal value that drug use is an individual deviance being managed. Instead it
reveals systemic, invisible, and now ecological dysfunction — the problem is
not a few bad individuals but the entire water table. Dark and absurd.
Flags: absurd-headline, capitalism-relevant, dark-wit, animal-warped

EXAMPLE 3 (ACCEPT — score: 0.85)
Headline: "Gibraltar monkeys eating mud to offset effects of tourist junk food"
Why it qualifies: The absurdity is the inversion. Animals — supposedly part of
untouched nature — are now self-medicating with dirt because corporate food
products are so biologically destructive that mud is the superior nutritional
choice. The stated value that the food industry improves human (and now animal)
welfare is directly contradicted by the observable behaviour of creatures who
have no economic incentive to lie about what makes them feel better.
Flags: absurd-headline, capitalism-relevant, dark-wit, animal-warped

---
ARTICLE TO EVALUATE
---

Headline: {headline}
{excerpt_block}

---

Apply the evaluative framework above. Ask whether this headline reveals a
genuine contradiction between stated values and actual behaviour — and whether
that contradiction is absurd rather than merely bad.

Respond with valid JSON only. No preamble. No explanation outside the JSON.
No markdown code fences. Your entire response must be a single JSON object
with exactly these three keys:

  "score"     — a float from 0.0 to 1.0 representing overall fit
  "rationale" — one or two sentences explaining your judgement, explicitly
                naming the societal contradiction the headline exposes (or
                why no such contradiction exists)
  "flags"     — a list of zero or more strings from this exact set:
                "absurd-headline", "capitalism-relevant", "dark-wit",
                "real-reporting", "animal-warped", "economic-grotesque",
                "tech-irony"

Example of a valid response (do not copy this — evaluate the actual article):
{{"score": 0.85, "rationale": "Contradicts the stated value that economic systems improve living standards — instead reveals a grotesque outcome produced by their internal logic.", "flags": ["absurd-headline", "capitalism-relevant", "dark-wit", "economic-grotesque"]}}

Your response:"""
