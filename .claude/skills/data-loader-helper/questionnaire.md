# Cold-start questionnaire

Use this when the user is **starting from scratch** — no working `load_data` exists yet (no
project folder, or a folder with only a stub). It elicits the two things you cannot design
anything without: **what the data is** and **what equation they want to discover**. Once
these are answered you have enough to populate the top rows of the design log and move into
the deeper design questions in `SKILL.md` ("Run the interview": shared-vs-varies,
generalisation axis, pointwise-vs-integrative, splits).

Ask conversationally, one or two questions at a time — this is a checklist of what you must
come away knowing, not a form to paste at the user. Record each answer into the matching row
of `projects/<name>/design_log.md` as you go. If the user already volunteered something, don't
re-ask it; confirm and move on.

## A. The data

1. **Raw shape.** What is the current shape of the data array(s), with a name for each axis
   and its size? (e.g. `(50 sessions, 40 cells, 200 timepoints)`.) If it's not yet a single
   array, what are the pieces and their shapes?
2. **Fields / keys.** What distinct measured quantities are there, and what would each be
   called as a dict key? (e.g. `position`, `stimulus`, `spike_count`.) Which are *inputs*
   (features the model reads) vs the *thing to be predicted* (the regression target)?
3. **Qualitative description.** In a few sentences: what was actually measured, in what
   system, with what units? What is the noise story (measurement noise, trial-to-trial
   variability, anything that's not the signal)?
4. **Provenance.** Is this real measured data (and where does it load from), or synthetic? If
   synthetic, what generates it and is there a known ground-truth equation/parameters?

## B. The model to discover

5. **The relationship.** In words, what relationship are they trying to discover — what maps
   to what?
6. **Formal statement.** Ask for it as **pseudo-code** *or* a **LaTeX equation** (whichever is
   easier for them). You need: the inputs, the output, and which symbols are *fitted
   parameters* vs *given features*. Example forms to offer:
   - LaTeX: `y_t = A \, e^{-x_t / \tau} + b`, with `A, \tau, b` fitted per sample.
   - Pseudocode: `pred[t] = sum_j force(dx[t,j]; A, B)` over neighbours `j`.
7. **Parameters: shared vs varying.** For each fitted parameter, must it take the *same value*
   across some set of entities (a global/population constant) or is it free to differ per
   entity? (This is the hook into the interview's pivotal sample-definition question — flag it now,
   resolve it there.)
8. **Known answer?** Is there a ground-truth form you're trying to recover (sanity check /
   synthetic), or is the functional form genuinely unknown (true discovery)?

After A + B you should be able to fill the *Raw data axes + sizes*, *Target equation /
hypothesis*, and a first-pass *Trailing-axis layout* rows of the design log (status
`proposed`), and name the candidate features/target keys. Then proceed to "Run the interview".
