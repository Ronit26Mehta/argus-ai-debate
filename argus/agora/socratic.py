"""
Socratic Engine — central orchestrator for an AGORA session.

Drives the five-phase deliberation pipeline, coordinating all
sub-engines (procedural rules, evidence docket, coalition detection,
floor time, quorum, filibuster, Senate Record) and delegating
LLM interactions through ARGUS's existing agent infrastructure.

Supports unbounded sessions: when no time limit is set, agents
complete all phases naturally without any artificial time constraint.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

from argus.agora.models import (
    AgoraResult,
    AgoraSessionConfig,
    ChallengeOutcome,
    ChallengeType,
    DocketEvidenceType,
    EvidencePolarity,
    RecordEntryType,
    SenateRecordEntry,
    SenateSpec,
    SenatorCategory,
    SenatorScorecard,
    SenatorSpec,
    SessionPhase,
    StoppingTrigger,
    _utcnow,
    _uid,
)
from argus.agora.procedures import (
    PhaseManager,
    FloorTimeEngine,
    FilibusterDetector,
    PointOfOrderHandler,
    QuorumEngine,
)
from argus.agora.docket import EvidenceDocket
from argus.agora.coalitions import CoalitionDetectionEngine
from argus.agora.minority import MinorityReportEngine
from argus.agora.record import SenateRecord
from argus.agora.results import AgoraResultBuilder

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# ── LLM Prompts ───────────────────────────────────────────────────────

_OPENING_SYSTEM = """\
You are {name}, a {category} senator in the AGORA Senate.
Your expertise: {expertise}
Your mandate: {mandate}
Your starting position on the proposition (0.0 = strongly oppose, 1.0 = strongly support): {prior}
Your temperament: {temperament}

Deliver a concise opening statement (2-3 sentences) on this proposition.
State your initial assessment and what evidence you intend to seek.
"""

_EVIDENCE_SYSTEM = """\
You are {name}, a {category} senator gathering evidence.
Your expertise: {expertise}
Your mandate: {mandate}
Your current position: {current_pos:.2f}

Given the proposition and session context so far, submit ONE piece of evidence.
Output as JSON:
{{
  "claim": "<your evidence claim, 1-2 sentences>",
  "polarity": "<supports|attacks|qualifies>",
  "source_reference": "<type of source: study, report, case, analysis, etc.>",
  "evidence_type": "<quantitative|qualitative|historical|theoretical|anecdotal|legal|experimental>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<why this evidence is relevant>"
}}
"""

_CHALLENGE_SYSTEM = """\
You are {name}, an {category} senator. You may challenge any evidence \
that you find to be weak, biased, or incorrectly inferred.

Review the following evidence item and decide if you want to challenge it.
If you challenge, output JSON:
{{
  "should_challenge": true,
  "challenge_type": "<claim|source|inference|confidence>",
  "argument": "<your challenge argument, 1-2 sentences>"
}}

If you accept it, output:
{{
  "should_challenge": false
}}
"""

_EA_RULING_SYSTEM = """\
You are the Epistemic Auditor presiding over a formal evidence challenge.

Challenge type: {challenge_type}
Challenge argument: {argument}
Evidence being challenged: {evidence_text}
Submitting senator's reply: {reply}

Rule on this challenge. Output JSON:
{{
  "outcome": "<sustained|overruled|noted>",
  "reasoning": "<1-2 sentence ruling rationale>"
}}
"""

_SYNTHESIS_SYSTEM = """\
You are {name}, a Synthesis Agent in the AGORA Senate.
Given the evidence submitted so far and the current posterior:

Summarise the key themes, identify unresolved tensions, and state
which evidence you consider most impactful. Output 2-3 sentences.
"""

_CLOSING_SYSTEM = """\
You are {name}, a {category} senator delivering your closing statement.
Your final position: {final_pos:.2f}

Summarise your position in 1-2 sentences. Has your position changed since
your opening statement? If so, explain what evidence moved you.
"""

_POSITION_UPDATE_SYSTEM = """\
You are {name}, a {category} senator updating your position.
Your current position: {current_pos:.2f}
New evidence just submitted: {evidence_text}
Evidence polarity: {polarity}
Evidence DEW score: {dew:.2f}

Given this new evidence, what is your updated position?
Output ONLY a JSON object:
{{"new_position": <float 0.0-1.0>, "reasoning": "<brief reason>"}}
"""


# ══════════════════════════════════════════════════════════════════════
# Live Senator — runtime wrapper
# ══════════════════════════════════════════════════════════════════════

class _LiveSenator:
    """Runtime wrapper binding a SenatorSpec to session state."""

    def __init__(self, spec: SenatorSpec):
        self.spec = spec
        self.current_position: float = spec.prior_position
        self.position_history: list[float] = [spec.prior_position]
        self.evidence_submitted: int = 0
        self.challenges_issued: int = 0
        self.challenges_received: int = 0
        self.challenges_sustained: int = 0
        self.challenges_overruled: int = 0
        self.points_of_order: int = 0
        self.active: bool = True

    def update_position(self, new_pos: float) -> None:
        self.current_position = max(0.0, min(1.0, new_pos))
        self.position_history.append(self.current_position)

    def to_scorecard(self) -> SenatorScorecard:
        return SenatorScorecard(
            senator_id=self.spec.id,
            senator_name=self.spec.name,
            category=self.spec.category,
            floor_time_used=self.evidence_submitted + self.challenges_issued + self.points_of_order,
            floor_time_budget=self.spec.floor_time_budget,
            evidence_submitted=self.evidence_submitted,
            challenges_issued=self.challenges_issued,
            challenges_received=self.challenges_received,
            challenges_sustained=self.challenges_sustained,
            challenges_overruled=self.challenges_overruled,
            points_of_order=self.points_of_order,
            position_trajectory=list(self.position_history),
        )


# ══════════════════════════════════════════════════════════════════════
# Socratic Engine
# ══════════════════════════════════════════════════════════════════════

class SocraticEngine:
    """Central orchestrator for an AGORA deliberation session.

    Coordinates the five procedural phases, enforces rules via
    sub-engines, and manages all LLM interactions through the
    ARGUS provider system.
    """

    # Rate-limit delay between LLM calls (seconds)
    _RATE_LIMIT_DELAY = 8.0

    def __init__(self, llm: "BaseLLM", config: AgoraSessionConfig | None = None):
        self.llm = llm
        self.config = config or AgoraSessionConfig()
        self.live_senators: dict[str, _LiveSenator] | None = None
        self.position_trajectories: dict[str, list[float]] | None = None
        self.docket: EvidenceDocket | None = None
        self.record: SenateRecord | None = None
        self.senate: SenateSpec | None = None

    def run_session(
        self,
        senate: SenateSpec,
        proposition: str,
        config: AgoraSessionConfig | None = None,
        round_callback: Optional[Callable] = None,
    ) -> AgoraResult:
        """Run a complete AGORA session.

        Args:
            senate: The generated SenateSpec.
            proposition: The proposition text.
            config: Session config (overrides constructor config).
            round_callback: Optional callback(phase, round_num, record_entry)
                invoked after each round for live UI updates.

        Returns:
            Complete AgoraResult with all 9 components.
        """
        config = config or self.config
        start_time = time.time()
        total_tokens = 0

        # ── Initialise sub-engines ────────────────────────────────────
        phase_mgr = PhaseManager(config)
        floor_time = FloorTimeEngine(senate)
        filibuster = FilibusterDetector()
        poo_handler = PointOfOrderHandler()
        quorum = QuorumEngine(senate, config.quorum_fraction)
        docket = EvidenceDocket()
        cde = CoalitionDetectionEngine(
            threshold=config.coalition_similarity_threshold,
            llm=self.llm,
        )
        record = SenateRecord(proposition=proposition)
        minority_engine = MinorityReportEngine(llm=self.llm)
        result_builder = AgoraResultBuilder(llm=self.llm)

        # ── Instantiate live senators ─────────────────────────────────
        live_senators: dict[str, _LiveSenator] = {}
        for spec in senate.senators:
            live_senators[spec.id] = _LiveSenator(spec)

        position_trajectories: dict[str, list[float]] = {
            sid: [ls.current_position] for sid, ls in live_senators.items()
        }
        self.live_senators = live_senators
        self.position_trajectories = position_trajectories
        self.docket = docket
        self.record = record
        self.senate = senate

        # Record session start
        record.add_entry(SenateRecordEntry(
            entry_type=RecordEntryType.SOCRATIC_ACTION,
            phase=SessionPhase.OPENING_STATEMENTS,
            content=(
                f"AGORA session opened. Proposition: \"{proposition}\" "
                f"Senate: {senate.senate_size} senators. "
                f"Session mode: {'UNBOUNDED' if config.is_unbounded else 'TIME-BOUNDED'}."
            ),
        ))

        stopping_trigger: StoppingTrigger | None = None

        # ══════════════════════════════════════════════════════════════
        # PHASE LOOP
        # ══════════════════════════════════════════════════════════════

        while not phase_mgr.is_session_complete:
            phase = phase_mgr.current_phase
            logger.info("═══ Phase %d: %s ═══", phase.phase_number, phase.value)

            # Check time boundary (for time-bounded sessions)
            if phase_mgr.check_time_boundary():
                stopping_trigger = StoppingTrigger.TIME_BOUNDARY
                record.add_entry(SenateRecordEntry(
                    entry_type=RecordEntryType.STOPPING_TRIGGER,
                    phase=phase,
                    round_num=phase_mgr.current_round,
                    content=f"Time boundary reached ({phase_mgr.elapsed_seconds:.0f}s). Session ending.",
                ))
                break

            # ── Phase-specific logic ──────────────────────────────────

            if phase == SessionPhase.OPENING_STATEMENTS:
                tokens = self._run_opening_phase(
                    live_senators, proposition, record, quorum, floor_time, phase_mgr,
                    round_callback,
                )
                total_tokens += tokens

            elif phase == SessionPhase.EVIDENCE_SUBMISSION:
                tokens, trigger = self._run_evidence_phase(
                    live_senators, proposition, record, docket, quorum,
                    floor_time, filibuster, cde, phase_mgr, config, position_trajectories,
                    round_callback,
                )
                total_tokens += tokens
                if trigger:
                    stopping_trigger = trigger
                    break

            elif phase == SessionPhase.CROSS_EXAMINATION:
                tokens = self._run_cross_exam_phase(
                    live_senators, proposition, record, docket,
                    floor_time, cde, phase_mgr, position_trajectories,
                    round_callback,
                )
                total_tokens += tokens

            elif phase == SessionPhase.DELIBERATIVE_SYNTHESIS:
                tokens = self._run_synthesis_phase(
                    live_senators, proposition, record, docket, cde, phase_mgr,
                    position_trajectories, round_callback,
                )
                total_tokens += tokens

            elif phase == SessionPhase.CLOSING_AND_VERDICT:
                tokens = self._run_closing_phase(
                    live_senators, proposition, record, phase_mgr,
                    round_callback,
                )
                total_tokens += tokens

            # ── Advance phase ─────────────────────────────────────────
            old_phase = phase_mgr.current_phase
            new_phase = phase_mgr.advance_phase()
            if new_phase:
                record.add_entry(phase_mgr.make_transition_record(new_phase))
                if round_callback:
                    round_callback(new_phase, phase_mgr.current_round, None)

        # ── Build final result ────────────────────────────────────────
        duration = time.time() - start_time
        scorecards = [ls.to_scorecard() for ls in live_senators.values()]
        final_trajectories = {
            sid: ls.position_history for sid, ls in live_senators.items()
        }

        result = result_builder.build(
            proposition=proposition,
            senate=senate,
            docket=docket,
            cde=cde,
            quorum=quorum,
            record=record,
            minority_engine=minority_engine,
            position_trajectories=final_trajectories,
            scorecards=scorecards,
            num_rounds=phase_mgr.total_rounds_elapsed,
            duration_seconds=duration,
            total_tokens_used=total_tokens,
            stopping_trigger=stopping_trigger,
        )

        logger.info(
            "AGORA session complete: %s (%.0%%) in %.0fs, %d evidence items",
            result.majority_opinion.verdict_label.value,
            result.majority_opinion.posterior_probability,
            duration, result.num_evidence,
        )
        return result

    # ══════════════════════════════════════════════════════════════════
    # Phase: Opening Statements
    # ══════════════════════════════════════════════════════════════════

    def _run_opening_phase(
        self,
        live_senators: dict[str, _LiveSenator],
        proposition: str,
        record: SenateRecord,
        quorum: QuorumEngine,
        floor_time: FloorTimeEngine,
        phase_mgr: PhaseManager,
        round_callback: Optional[Callable],
    ) -> int:
        """Each senator delivers an opening statement."""
        tokens = 0
        for idx, (sid, ls) in enumerate(live_senators.items()):
            if idx > 0:
                time.sleep(self._RATE_LIMIT_DELAY)

            prompt = f"Proposition: {proposition}"
            system = _OPENING_SYSTEM.format(
                name=ls.spec.name,
                category=ls.spec.category.display_name,
                expertise=ls.spec.domain_expertise,
                mandate=ls.spec.evidence_gathering_mandate,
                prior=ls.spec.prior_position,
                temperament=ls.spec.deliberative_temperament,
            )
            try:
                response = self.llm.generate(
                    prompt=prompt, system_prompt=system,
                    temperature=0.5, max_tokens=256,
                )
                statement = response.content.strip()
                tokens += response.usage.total_tokens if response.usage else 300
            except Exception as exc:
                logger.warning("Opening statement failed for %s: %s", ls.spec.name, exc)
                statement = f"I am {ls.spec.name}, {ls.spec.category.display_name}. I bring expertise in {ls.spec.domain_expertise}."

            quorum.record_participation(sid)
            floor_time.consume(sid, "statement", phase_mgr.current_round)
            record.add_entry(SenateRecordEntry(
                entry_type=RecordEntryType.SENATOR_STATEMENT,
                phase=SessionPhase.OPENING_STATEMENTS,
                round_num=0,
                senator_id=sid,
                senator_name=ls.spec.name,
                content=statement,
            ))

        phase_mgr.advance_round()
        if round_callback:
            round_callback(SessionPhase.OPENING_STATEMENTS, 0, None)
        return tokens

    # ══════════════════════════════════════════════════════════════════
    # Phase: Evidence Submission
    # ══════════════════════════════════════════════════════════════════

    def _run_evidence_phase(
        self,
        live_senators: dict[str, _LiveSenator],
        proposition: str,
        record: SenateRecord,
        docket: EvidenceDocket,
        quorum: QuorumEngine,
        floor_time: FloorTimeEngine,
        filibuster: FilibusterDetector,
        cde: CoalitionDetectionEngine,
        phase_mgr: PhaseManager,
        config: AgoraSessionConfig,
        trajectories: dict[str, list[float]],
        round_callback: Optional[Callable],
    ) -> tuple[int, StoppingTrigger | None]:
        """Multi-round evidence gathering."""
        tokens = 0

        while True:
            phase_exhausted = phase_mgr.advance_round()
            round_num = phase_mgr.current_round
            logger.info("  Evidence round %d/%d", round_num, phase_mgr.max_rounds_in_phase)

            for idx, (sid, ls) in enumerate(live_senators.items()):
                if not ls.active or not floor_time.can_speak(sid):
                    continue
                if idx > 0:
                    time.sleep(self._RATE_LIMIT_DELAY)

                # Check time boundary
                if phase_mgr.check_time_boundary():
                    return tokens, StoppingTrigger.TIME_BOUNDARY

                # Get evidence via LLM
                evidence_data = self._get_evidence_from_senator(
                    ls, proposition, docket, round_num,
                )
                if evidence_data is None:
                    continue

                claim = evidence_data.get("claim", "")
                if not claim:
                    continue

                # Filibuster check
                if filibuster.check(sid, claim):
                    record.add_entry(filibuster.make_filibuster_record(
                        sid, ls.spec.name, SessionPhase.EVIDENCE_SUBMISSION, round_num,
                    ))
                    logger.info("Filibuster blocked: %s (round %d)", ls.spec.name, round_num)
                    continue

                filibuster.register(sid, claim)

                # Parse polarity
                polarity_str = evidence_data.get("polarity", "supports").lower()
                polarity_map = {
                    "supports": EvidencePolarity.SUPPORTS,
                    "attacks": EvidencePolarity.ATTACKS,
                    "qualifies": EvidencePolarity.QUALIFIES,
                }
                polarity = polarity_map.get(polarity_str, EvidencePolarity.SUPPORTS)

                # Parse evidence type
                etype_str = evidence_data.get("evidence_type", "qualitative").lower()
                try:
                    etype = DocketEvidenceType(etype_str)
                except ValueError:
                    etype = DocketEvidenceType.QUALITATIVE

                confidence = float(evidence_data.get("confidence", 0.5))

                # Submit to docket
                item = docket.submit_evidence(
                    senator_id=sid,
                    senator_name=ls.spec.name,
                    senator_category=ls.spec.category,
                    claim_text=claim,
                    polarity=polarity,
                    source_reference=evidence_data.get("source_reference", ""),
                    source_type=evidence_data.get("source_reference", "general"),
                    confidence_score=confidence,
                    evidence_type=etype,
                    round_num=round_num,
                )
                ls.evidence_submitted += 1
                quorum.record_participation(sid)
                floor_time.consume(sid, "evidence_submit", round_num)
                tokens += 500  # estimate

                # Record
                record.add_entry(docket.make_submission_record(
                    item, SessionPhase.EVIDENCE_SUBMISSION, round_num,
                ))

                # Update positions of all senators based on new evidence
                self._update_positions_from_evidence(
                    live_senators, item, proposition, cde, trajectories,
                )

            # Coalition detection after each round
            cde.detect_coalitions()
            for coalition in cde.current_coalitions:
                record.add_entry(cde.make_coalition_record(
                    coalition, SessionPhase.EVIDENCE_SUBMISSION, round_num,
                ))

            # Quorum check
            record.add_entry(quorum.make_quorum_record(
                SessionPhase.EVIDENCE_SUBMISSION, round_num,
            ))

            if round_callback:
                round_callback(SessionPhase.EVIDENCE_SUBMISSION, round_num, None)

            # Check stopping triggers
            if StoppingTrigger.QUORUM_FAILURE in config.active_triggers:
                if quorum.check_quorum_failure():
                    record.add_entry(SenateRecordEntry(
                        entry_type=RecordEntryType.STOPPING_TRIGGER,
                        phase=SessionPhase.EVIDENCE_SUBMISSION,
                        round_num=round_num,
                        content="Quorum failure — insufficient participation. Session ending.",
                    ))
                    return tokens, StoppingTrigger.QUORUM_FAILURE

            if StoppingTrigger.FULL_EVIDENCE in config.active_triggers:
                all_met = all(
                    ls.evidence_submitted >= config.min_evidence_per_senator
                    for ls in live_senators.values()
                    if ls.active
                )
                if all_met:
                    record.add_entry(SenateRecordEntry(
                        entry_type=RecordEntryType.STOPPING_TRIGGER,
                        phase=SessionPhase.EVIDENCE_SUBMISSION,
                        round_num=round_num,
                        content="All senators met minimum evidence requirement. Moving to cross-examination.",
                    ))
                    break

            if StoppingTrigger.CONVERGENCE in config.active_triggers:
                positions = [ls.current_position for ls in live_senators.values()]
                if len(positions) >= 2:
                    import math
                    mean = sum(positions) / len(positions)
                    std_dev = math.sqrt(sum((p - mean) ** 2 for p in positions) / len(positions))
                    if std_dev < config.convergence_threshold:
                        record.add_entry(SenateRecordEntry(
                            entry_type=RecordEntryType.STOPPING_TRIGGER,
                            phase=SessionPhase.EVIDENCE_SUBMISSION,
                            round_num=round_num,
                            content=f"Position convergence detected (σ={std_dev:.3f}). Moving forward.",
                        ))
                        break

            if phase_exhausted:
                break

        return tokens, None

    def _get_evidence_from_senator(
        self,
        ls: _LiveSenator,
        proposition: str,
        docket: EvidenceDocket,
        round_num: int,
    ) -> dict | None:
        """Get one evidence item from a senator via LLM."""
        # Recent docket context
        recent_items = docket.all_items[-5:]
        context = "\n".join(
            f"- [{i.evidence_id_display}] {i.claim_text[:100]} ({i.polarity.value})"
            for i in recent_items
        ) if recent_items else "No evidence submitted yet."

        prompt = (
            f"Proposition: {proposition}\n\n"
            f"Current round: {round_num}\n"
            f"Recent evidence in the docket:\n{context}\n"
        )
        system = _EVIDENCE_SYSTEM.format(
            name=ls.spec.name,
            category=ls.spec.category.display_name,
            expertise=ls.spec.domain_expertise,
            mandate=ls.spec.evidence_gathering_mandate,
            current_pos=ls.current_position,
        )

        try:
            response = self.llm.generate(
                prompt=prompt, system_prompt=system,
                temperature=0.5, max_tokens=800,
            )
            text = response.content.strip()
            if "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                return json.loads(text[start:end])
        except Exception as exc:
            logger.warning("Evidence LLM failed for %s: %s", ls.spec.name, exc)
        return None

    def _update_positions_from_evidence(
        self,
        live_senators: dict[str, _LiveSenator],
        new_item: Any,
        proposition: str,
        cde: CoalitionDetectionEngine,
        trajectories: dict[str, list[float]],
    ) -> None:
        """Update all senators' positions in response to new evidence."""
        for sid, ls in live_senators.items():
            if sid == new_item.senator_id:
                continue  # Don't update the submitter

            # Simplified position update based on evidence polarity and DEW
            shift = 0.0
            if new_item.polarity == EvidencePolarity.SUPPORTS:
                shift = new_item.dew_score * 0.05
            elif new_item.polarity == EvidencePolarity.ATTACKS:
                shift = -new_item.dew_score * 0.05
            else:  # QUALIFIES
                shift = new_item.dew_score * 0.02 * (0.5 - ls.current_position)

            new_pos = ls.current_position + shift
            ls.update_position(new_pos)

            # Feed into coalition detection
            signal = 1.0 if new_pos > 0.5 else (-1.0 if new_pos < 0.5 else 0.0)
            cde.update_position(sid, ls.spec.name, signal)

            # Track trajectory
            trajectories.setdefault(sid, []).append(ls.current_position)

    # ══════════════════════════════════════════════════════════════════
    # Phase: Cross-Examination
    # ══════════════════════════════════════════════════════════════════

    def _run_cross_exam_phase(
        self,
        live_senators: dict[str, _LiveSenator],
        proposition: str,
        record: SenateRecord,
        docket: EvidenceDocket,
        floor_time: FloorTimeEngine,
        cde: CoalitionDetectionEngine,
        phase_mgr: PhaseManager,
        trajectories: dict[str, list[float]],
        round_callback: Optional[Callable],
    ) -> int:
        """Challenge rounds — AC, EA, and DA senators challenge evidence."""
        tokens = 0
        challenge_categories = {
            SenatorCategory.ADVERSARIAL_CHALLENGER,
            SenatorCategory.EPISTEMIC_AUDITOR,
            SenatorCategory.DEVILS_ADVOCATE,
        }

        while True:
            phase_exhausted = phase_mgr.advance_round()
            round_num = phase_mgr.current_round

            # Get challengeable evidence
            all_evidence = docket.all_items
            if not all_evidence:
                break

            for sid, ls in live_senators.items():
                if ls.spec.category not in challenge_categories:
                    continue
                if not floor_time.can_speak(sid):
                    continue

                time.sleep(self._RATE_LIMIT_DELAY)

                # Pick evidence to potentially challenge (most recent, not own)
                for target_item in reversed(all_evidence):
                    if target_item.senator_id == sid:
                        continue
                    if target_item.is_challenged:
                        continue

                    challenge_data = self._ask_senator_to_challenge(
                        ls, target_item, proposition,
                    )
                    if challenge_data and challenge_data.get("should_challenge"):
                        # Issue challenge
                        ch_type_str = challenge_data.get("challenge_type", "claim")
                        try:
                            ch_type = ChallengeType(ch_type_str)
                        except ValueError:
                            ch_type = ChallengeType.CLAIM

                        challenge = docket.issue_challenge(
                            challenger_id=sid,
                            challenger_name=ls.spec.name,
                            target_evidence_id=target_item.id,
                            challenge_type=ch_type,
                            argument=challenge_data.get("argument", ""),
                        )
                        ls.challenges_issued += 1
                        target_senator = live_senators.get(target_item.senator_id)
                        if target_senator:
                            target_senator.challenges_received += 1

                        floor_time.consume(sid, "challenge", round_num)
                        record.add_entry(docket.challenge_handler.make_challenge_record(
                            challenge, SessionPhase.CROSS_EXAMINATION,
                        ))
                        tokens += 500

                        # EA ruling (auto-adjudicate)
                        ea_ruling = self._get_ea_ruling(
                            challenge, target_item, live_senators,
                        )
                        docket.resolve_challenge(
                            challenge, ea_ruling["outcome"], ea_ruling["reasoning"],
                        )
                        record.add_entry(docket.challenge_handler.make_ruling_record(
                            challenge, SessionPhase.CROSS_EXAMINATION,
                        ))
                        tokens += 500

                        if target_senator:
                            if ea_ruling["outcome"] == ChallengeOutcome.SUSTAINED:
                                target_senator.challenges_sustained += 1
                            elif ea_ruling["outcome"] == ChallengeOutcome.OVERRULED:
                                target_senator.challenges_overruled += 1

                        break  # one challenge per senator per round

            if round_callback:
                round_callback(SessionPhase.CROSS_EXAMINATION, round_num, None)

            if phase_exhausted:
                break

        return tokens

    def _ask_senator_to_challenge(
        self,
        ls: _LiveSenator,
        target: Any,
        proposition: str,
    ) -> dict | None:
        """Ask a senator if they want to challenge an evidence item."""
        system = _CHALLENGE_SYSTEM.format(
            name=ls.spec.name,
            category=ls.spec.category.display_name,
        )
        prompt = (
            f"Proposition: {proposition}\n\n"
            f"Evidence to review [{target.evidence_id_display}]:\n"
            f"Claim: {target.claim_text}\n"
            f"Polarity: {target.polarity.value}\n"
            f"Submitted by: {target.senator_name} ({target.senator_category.display_name})\n"
            f"Confidence: {target.confidence_score:.2f}\n"
            f"DEW: {target.dew_score:.2f}\n"
        )
        try:
            response = self.llm.generate(
                prompt=prompt, system_prompt=system,
                temperature=0.3, max_tokens=512,
            )
            text = response.content.strip()
            if "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                return json.loads(text[start:end])
        except Exception as exc:
            logger.warning("Challenge LLM failed: %s", exc)
        return None

    def _get_ea_ruling(
        self,
        challenge: Any,
        target: Any,
        live_senators: dict[str, _LiveSenator],
    ) -> dict:
        """Get EA ruling on a challenge via LLM."""
        system = _EA_RULING_SYSTEM.format(
            challenge_type=challenge.challenge_type.value,
            argument=challenge.challenge_argument,
            evidence_text=target.claim_text,
            reply="No reply submitted.",
        )
        try:
            response = self.llm.generate(
                prompt="Please rule on this challenge.",
                system_prompt=system,
                temperature=0.3, max_tokens=512,
            )
            text = response.content.strip()
            if "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                data = json.loads(text[start:end])
                outcome_str = data.get("outcome", "noted")
                outcome_map = {
                    "sustained": ChallengeOutcome.SUSTAINED,
                    "overruled": ChallengeOutcome.OVERRULED,
                    "noted": ChallengeOutcome.NOTED,
                }
                return {
                    "outcome": outcome_map.get(outcome_str, ChallengeOutcome.NOTED),
                    "reasoning": data.get("reasoning", ""),
                }
        except Exception as exc:
            logger.warning("EA ruling LLM failed: %s", exc)
        return {"outcome": ChallengeOutcome.NOTED, "reasoning": "Unable to adjudicate — noted."}

    # ══════════════════════════════════════════════════════════════════
    # Phase: Deliberative Synthesis
    # ══════════════════════════════════════════════════════════════════

    def _run_synthesis_phase(
        self,
        live_senators: dict[str, _LiveSenator],
        proposition: str,
        record: SenateRecord,
        docket: EvidenceDocket,
        cde: CoalitionDetectionEngine,
        phase_mgr: PhaseManager,
        trajectories: dict[str, list[float]],
        round_callback: Optional[Callable],
    ) -> int:
        """Synthesis agents summarise themes; all senators update positions."""
        tokens = 0
        synthesis_categories = {
            SenatorCategory.SYNTHESIS_AGENT,
            SenatorCategory.CROSS_DOMAIN_INTEGRATOR,
        }

        while True:
            phase_exhausted = phase_mgr.advance_round()
            round_num = phase_mgr.current_round

            # Synthesis agents produce summaries
            stats = docket.summary_stats()
            top_evidence = docket.get_top_weighted(5)
            evidence_summary = "\n".join(
                f"- [{e.evidence_id_display}] {e.claim_text[:100]} (DEW: {e.dew_score:.2f})"
                for e in top_evidence
            )

            for sid, ls in live_senators.items():
                if ls.spec.category not in synthesis_categories:
                    continue
                if not floor_time.can_speak(sid) if hasattr(self, '_floor_time') else False:
                    continue

                time.sleep(self._RATE_LIMIT_DELAY)

                system = _SYNTHESIS_SYSTEM.format(
                    name=ls.spec.name,
                )
                prompt = (
                    f"Proposition: {proposition}\n\n"
                    f"Evidence stats: {stats['total']} items — "
                    f"{stats['supports']} support, {stats['attacks']} attack\n"
                    f"Top evidence:\n{evidence_summary}\n"
                    f"Coalitions: {len(cde.current_coalitions)} detected\n"
                )
                try:
                    response = self.llm.generate(
                        prompt=prompt, system_prompt=system,
                        temperature=0.4, max_tokens=1024,
                    )
                    synthesis_text = response.content.strip()
                    tokens += response.usage.total_tokens if response.usage else 500
                except Exception as exc:
                    logger.warning("Synthesis LLM failed for %s: %s", ls.spec.name, exc)
                    synthesis_text = "Synthesis: Evidence is mixed with both supporting and challenging items."

                record.add_entry(SenateRecordEntry(
                    entry_type=RecordEntryType.SENATOR_STATEMENT,
                    phase=SessionPhase.DELIBERATIVE_SYNTHESIS,
                    round_num=round_num,
                    senator_id=sid,
                    senator_name=ls.spec.name,
                    content=synthesis_text,
                ))

            # Update all positions after synthesis
            for sid, ls in live_senators.items():
                # Gentle position update towards evidence consensus
                supports = docket.get_by_polarity(EvidencePolarity.SUPPORTS)
                attacks = docket.get_by_polarity(EvidencePolarity.ATTACKS)
                sup_weight = sum(e.dew_score for e in supports)
                atk_weight = sum(e.dew_score for e in attacks)
                total_weight = sup_weight + atk_weight
                if total_weight > 0:
                    evidence_signal = sup_weight / total_weight
                    # Move 10% towards evidence consensus
                    new_pos = ls.current_position * 0.9 + evidence_signal * 0.1
                    ls.update_position(new_pos)
                    trajectories.setdefault(sid, []).append(ls.current_position)

            if round_callback:
                round_callback(SessionPhase.DELIBERATIVE_SYNTHESIS, round_num, None)

            if phase_exhausted:
                break

        return tokens

    # ══════════════════════════════════════════════════════════════════
    # Phase: Closing
    # ══════════════════════════════════════════════════════════════════

    def _run_closing_phase(
        self,
        live_senators: dict[str, _LiveSenator],
        proposition: str,
        record: SenateRecord,
        phase_mgr: PhaseManager,
        round_callback: Optional[Callable],
    ) -> int:
        """Each senator delivers a closing statement."""
        tokens = 0
        phase_mgr.advance_round()

        for idx, (sid, ls) in enumerate(live_senators.items()):
            if idx > 0:
                time.sleep(self._RATE_LIMIT_DELAY)

            system = _CLOSING_SYSTEM.format(
                name=ls.spec.name,
                category=ls.spec.category.display_name,
                final_pos=ls.current_position,
            )
            try:
                response = self.llm.generate(
                    prompt=f"Proposition: {proposition}",
                    system_prompt=system,
                    temperature=0.4, max_tokens=512,
                )
                closing = response.content.strip()
                tokens += response.usage.total_tokens if response.usage else 300
            except Exception as exc:
                logger.warning("Closing statement failed for %s: %s", ls.spec.name, exc)
                closing = (
                    f"After careful deliberation, my position stands at "
                    f"{ls.current_position:.0%}."
                )

            record.add_entry(SenateRecordEntry(
                entry_type=RecordEntryType.SENATOR_STATEMENT,
                phase=SessionPhase.CLOSING_AND_VERDICT,
                round_num=phase_mgr.current_round,
                senator_id=sid,
                senator_name=ls.spec.name,
                content=closing,
            ))

            # Record final position
            record.add_entry(SenateRecordEntry(
                entry_type=RecordEntryType.FINAL_POSITIONS,
                phase=SessionPhase.CLOSING_AND_VERDICT,
                round_num=phase_mgr.current_round,
                senator_id=sid,
                senator_name=ls.spec.name,
                content=(
                    f"Final position: {ls.current_position:.2%} "
                    f"(started at: {ls.spec.prior_position:.2%}, "
                    f"delta: {ls.current_position - ls.spec.prior_position:+.2%})"
                ),
                metadata={
                    "final_position": ls.current_position,
                    "prior_position": ls.spec.prior_position,
                    "delta": ls.current_position - ls.spec.prior_position,
                },
            ))

        if round_callback:
            round_callback(SessionPhase.CLOSING_AND_VERDICT, phase_mgr.current_round, None)

        return tokens
