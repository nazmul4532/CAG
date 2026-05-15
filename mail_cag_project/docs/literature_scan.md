# Quick Literature Scan

Last checked: 2026-05-15

## What Looks Related

There is active related work around LLMs and phishing/scam detection:

- LLM phishing detectors and multi-agent phishing detectors.
- LLM-generated phishing content and defenses against it.
- LLM-based adversarial training for scam detection.
- Classic NLP adversarial attacks through TextAttack recipes such as
  TextFooler, PWWS, and DeepWordBug.

## What Our Narrow Question Adds

I did not find an exact match for this setup in a quick search:

```text
ALBERT phishing-email classifier
+ cyclic LLM-generated label-preserving rewrites
+ Model B phishing-only vs Model C both-label rewriting
+ held-out evaluation with TextFooler, PWWS, and DeepWordBug
```

That makes the current project framing useful, but we should phrase the paper
carefully: the broad area exists; our contribution is the controlled cyclic
comparison and the clean separation between LLM training rewrites and classic
held-out attack evaluation.

## Why Model B vs Model C Matters

Model B asks whether rewriting only phishing examples improves robustness where
we care most.

Model C asks whether rewriting both labels prevents a shortcut: ALBERT might
learn that "LLM-looking text" means phishing if only phishing emails are
rewritten. If Model C is better, balanced rewriting probably helped reduce that
shortcut. If Model B is better, phishing-focused augmentation may be enough and
cheaper.

## Sources To Cite Later

- Robust Scam Detection via LLM-based Adversarial Training
- MultiPhishGuard: An LLM-based Multi-Agent System for Phishing Email Detection
- Phishing Email Detection Using Large Language Models
- Paladin: Defending LLM-enabled Phishing Emails with a New Trigger-Tag Paradigm
- TextAttack documentation and attack recipe papers for TextFooler, PWWS, and
  DeepWordBug
