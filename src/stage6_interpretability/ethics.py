"""
Stage 6 ethical reflection notes module.

Overall this module is where I write the ethics section of my Stage 6, the basic idea
is to capture the seven ethical considerations the brief asks me to discuss in a single
text block that gets printed to console and saved to file. In my project I focused on
automation bias, privacy, feedback loops, the synthetic data caveat, fairness limits,
human in the loop and explainability accountability. What this module demonstrates is
the ethics reflection that goes straight into the report.
"""


def print_ethical_notes():
    """
    Print the seven ethical considerations and return them as a single text block.

    Overall this is the ethics reflection content, the basic idea is to keep the notes
    in one place so the report can quote them verbatim and the file output can be saved
    as evidence. What this also returns is the same text so run.py can write it to the
    outputs folder.
    """
    print("\n" + "=" * 70)
    print("ETHICAL REFLECTION NOTES")
    print("=" * 70)

    notes = """
1. AUTOMATION BIAS
   When my model flags an email as likely to escalate, there is a real risk that
   a human reviewer trusts the flag uncritically and skips the actual content
   review. The basic idea here is that automation should support the reviewer
   not replace them, so my deployment recommendation in Stage 7 is human in the
   loop with the model used as a triage aid not a final decision.

2. PRIVACY
   The dataset contains customer email body text and customer_id, so any
   deployed system needs to handle these as personal data under UK GDPR. In my
   project I focused on minimising what features get used, dropping customer_id
   from the training matrix and only keeping the text content for TF-IDF
   features. What this also means in production is that model training on
   real customer emails would need a Data Protection Impact Assessment.

3. FEEDBACK LOOPS
   If my model flags certain customer types or regions more often, those flags
   could shape future review priorities, which then reinforces the flagging
   pattern. Overall this is the classic feedback loop risk in operational ML,
   and the only way I can mitigate it is through ongoing fairness monitoring
   per Stage 7's monitoring plan.

4. SYNTHETIC DATA CAVEAT
   The dataset is a synthetic teaching dataset rather than real customer
   complaints, so any conclusions I draw here are about the modelling
   methodology, not about any real customer population. The basic idea is
   that the patterns I see (positive sentiment escalating more than negative,
   weak categorical signals) might not generalise to real data, so my
   deployment recommendations have to be conditional on a real-data validation
   study.

5. FAIRNESS LIMITS
   I checked equalised odds across region, customer_type and tenure_type and
   the gaps were within the 80% rule, but my fairness audit did not include
   sensitive attributes like age, ethnicity or income because they are not in
   the dataset. What this also means is that my fairness conclusions are
   limited to the attributes I can observe, and any production deployment
   would need a wider fairness audit.

6. HUMAN IN THE LOOP
   Given the FN cost is high (regulatory escalation) and the FP cost is low
   (a wasted review), my recommendation is to keep the model as a flagging
   tool with a human reviewer making the final call. This is really important
   for accountability because it keeps the responsibility for the decision
   with a person who can be held to account.

7. EXPLAINABILITY AND ACCOUNTABILITY
   I picked Logistic Regression as my primary model partly because the
   coefficients are directly interpretable, and I added LIME explanations on
   top so individual predictions can be defended. What this also matters for
   is the regulatory side, an energy provider needs to be able to explain why
   a particular email was flagged and the LIME output gives me that.
"""

    print(notes)
    return notes
