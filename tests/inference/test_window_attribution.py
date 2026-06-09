"""
Unit tests for Window Attribution and DFS Enumeration logic.
Validates canonical deduplication, primary selection tiebreakers, and
the proper formation of WindowAttribution objects.
"""

from decimal import Decimal

import pytest

from poker_vision.inference.opponent_action_inferencer import (
    InferredAction,
    OpponentActionInferencer,
    _canonical_key,
    _deduplicate_sequences,
)


class TestCanonicalDeduplication:

    def test_identical_sequences_dedupe_to_one(self):
        seq1 = [InferredAction("v1", "call", Decimal("10"))]
        seq2 = [InferredAction("v1", "call", Decimal("10"))]

        unique, collisions = _deduplicate_sequences([seq1, seq2])

        assert len(unique) == 1
        assert collisions == 1
        assert unique[0] == seq1

    def test_different_action_kinds_per_player_preserved(self):
        seq1 = [InferredAction("v1", "check", Decimal("0"))]
        seq2 = [InferredAction("v1", "fold", Decimal("0"))]

        unique, collisions = _deduplicate_sequences([seq1, seq2])

        assert len(unique) == 2
        assert collisions == 0

    def test_different_amounts_per_player_preserved(self):
        seq1 = [InferredAction("v1", "bet", Decimal("10"))]
        seq2 = [InferredAction("v1", "bet", Decimal("20"))]

        unique, collisions = _deduplicate_sequences([seq1, seq2])

        assert len(unique) == 2
        assert collisions == 0

    def test_decimal_normalization_handled(self):
        """CRITICAL: Decimal('2') and Decimal('2.00') must be considered identical."""
        seq1 = [InferredAction("v1", "call", Decimal("2"))]
        seq2 = [InferredAction("v1", "call", Decimal("2.00"))]

        # O _canonical_key deve gerar a mesma chave graças ao .normalize()
        assert _canonical_key(seq1) == _canonical_key(seq2)

        unique, collisions = _deduplicate_sequences([seq1, seq2])
        assert len(unique) == 1
        assert collisions == 1

    def test_dedup_is_stable_first_occurrence_wins(self):
        # A primeira sequência deve ser a que fica na lista final
        seq1 = [InferredAction("v1", "call", Decimal("5"))]
        seq2 = [InferredAction("v1", "call", Decimal("5"))]

        unique, _ = _deduplicate_sequences([seq1, seq2])
        assert unique[0] is seq1


class TestPrimarySelection:

    @pytest.fixture
    def inferencer(self):
        return OpponentActionInferencer()

    def test_highest_score_wins(self, inferencer):
        seq_weak = [InferredAction("v1", "fold", Decimal("0"))]
        seq_strong = [InferredAction("v1", "call", Decimal("10"))]

        scored = [(seq_weak, 0.4), (seq_strong, 0.9)]

        primary, alternatives, p_score, r_score = inferencer._rank_and_select_primary(scored)

        assert primary == seq_strong
        assert p_score == 0.9
        assert r_score == 0.4
        assert len(alternatives) == 1
        assert alternatives[0] == seq_weak

    def test_lex_tiebreaker_on_equal_scores(self, inferencer):
        """When scores are perfectly equal, lexicographic string sorting breaks the tie deterministically."""
        seq_a = [InferredAction("v2", "check", Decimal("0"))]
        seq_b = [InferredAction("v1", "check", Decimal("0"))]

        scored = [(seq_a, 0.8), (seq_b, 0.8)]  # Scores are equal

        primary, alternatives, p_score, r_score = inferencer._rank_and_select_primary(scored)

        # 'v1' is lexicographically smaller than 'v2', so seq_b must be primary
        assert primary == seq_b
        assert p_score == 0.8
        assert r_score == 0.8
        assert alternatives[0] == seq_a

    def test_determinism_repeated_calls_same_output(self, inferencer):
        seq_a = [InferredAction("v2", "bet", Decimal("5"))]
        seq_b = [InferredAction("v1", "bet", Decimal("5"))]
        scored = [(seq_a, 0.5), (seq_b, 0.5)]

        # Múltiplas chamadas devem garantir rigorosamente a mesma ordem (v1 vence)
        for _ in range(5):
            primary, alts, _, _ = inferencer._rank_and_select_primary(scored)
            assert primary == seq_b
            assert alts[0] == seq_a


class TestConfidenceLogicPreserved:

    @pytest.fixture
    def inferencer(self):
        return OpponentActionInferencer()

    def test_existing_confidence_logic_preserved(self, inferencer):
        """
        Validates that splitting the method didn't break the reconciliation dampening logic.
        """
        # Configuração de cenário onde ocorreu reconciliação (unconstrained > constrained)
        primary_score = 0.9
        runner_up_score = 0.5
        pre_constraint_count = 4
        post_constraint_count = 2  # Metade foi podada

        # Chama a nova função isolada
        confidence, was_reconciled = inferencer._compute_primary_confidence(
            primary_score=primary_score,
            runner_up_score=runner_up_score,
            cfg=inferencer.cfg,
            pre_constraint_count=pre_constraint_count,
            post_constraint_count=post_constraint_count,
        )

        assert was_reconciled is True
        # O valor de confiança deve ter sofrido a penalização (dampening) de reconciliação
        # margin = (0.9 - 0.5) + 1.5 * (1 - 2/4) = 0.4 + 0.75 = 1.15
        # base = min(0.90, 0.55 + 0.35 * 1.15) = min(0.90, 0.9525) = 0.90
        # pos-reconciliacao = 0.90 - 0.15 = 0.75
        assert round(confidence, 2) == 0.75
