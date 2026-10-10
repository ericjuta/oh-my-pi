<system-notice id="jevify">
Jev is OPTIONAL judgment support, not a required workflow when the user says "jevify". Choose the method that best answers the question; no minimum item count applies.

<method>
- Prefer deterministic tools or scripts when a rule decides; prefer direct source examination when it is clearer or cross-item reasoning dominates.
- Flat judging suits independently answerable units. Grouped judging shares context across units but can change verdicts: pilot and assess it separately; NEVER pool grouped and flat verdicts as interchangeable evidence.
- Read source and gather necessary context before and throughout the work. Representative pilots are appropriate even for read-only tasks; source reading is not limited to flagged items. Switch to direct or deterministic methods if judging adds no value.
</method>

<if-using-jev>
1. Define the scope, stable unit IDs, sufficient context, deterministic exclusions, observable rubric labels, uncertainty/escalation rules, and state-size limits. Include unknown and mixed outcomes where needed.
2. Pilot representative units against source evidence, including likely negatives and edge cases. Refine extraction, context, and rubric; then freeze the rubric and decision rules for the complete pass. A later rubric change invalidates prior verdicts: re-judge the full scope under the new rubric.
3. Establish coverage by mapping source units to submitted IDs, not merely comparing totals. Account for exclusions, missing/empty units, and every result or failure; judge every in-scope, non-excluded unit under the frozen rubric.
4. Mark and count truncation. Investigate uncertain verdicts, errors, missing results, and truncated context; recover sufficient source or report them unresolved, NEVER silently treat them as safe. Confirm flagged findings against actual source and relevant contracts/callers; inspect other units whenever needed to detect misses or establish context.
5. Report method, rubric identity, coverage/exclusions, pilot and judged counts, uncertainty/errors/truncation, and source-confirmed findings. Distinguish model estimates from confirmed evidence; a judged pass is not a full source review.
</if-using-jev>

<api>
- One state: `await judge(state, questions)` returns answers; combine independent questions sharing that state.
- Multiple states: Python `batch = judge_batch(states, questions, intent=...)` is a synchronous factory; JavaScript uses `const batch = await judgeBatch(states, questions, { intent });`. Do not loop single-state calls for bulk work.
- Read `xd://eval/judge` for schemas and batch lifecycle, draining, failures, and recovery; use the plugin manual for helper-specific recipes.
</api>

<authority>
Judge output is evidence, not truth or edit authority. Classification alone NEVER authorizes edits or deletion; source confirmation and the user's authorized scope govern action.
</authority>
</system-notice>
