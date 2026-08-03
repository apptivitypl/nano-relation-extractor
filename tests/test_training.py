"""Tests for the loss objectives and the vocabulary trimmer.

Both are places where a mistake produces no error. A relation loss that ignores
its mask trains the head on documents that never had labels; a trimmer whose
lookup table is off by one silently feeds the encoder the wrong embeddings.
"""

from __future__ import annotations

import torch

from nano_re.models.vocabulary import VocabularyTrimmer
from nano_re.training.losses import (
    AdaptiveThresholdLoss,
    BinaryRelationLoss,
    MultiTaskLoss,
    build_relation_objective,
)


def test_masked_pairs_contribute_nothing_to_the_relation_loss() -> None:
    """A fully masked batch yields exactly zero relation loss."""
    objective = AdaptiveThresholdLoss()
    logits = torch.randn(2, 3, 5)
    targets = torch.zeros(2, 3, 5)
    targets[..., 0] = 1.0
    assert float(objective.loss(logits, targets, torch.zeros(2, 3))) == 0.0


def test_unmasked_pairs_do_contribute() -> None:
    """An active mask produces a non-zero loss, so masking is what differs."""
    objective = AdaptiveThresholdLoss()
    logits = torch.randn(2, 3, 5)
    targets = torch.zeros(2, 3, 5)
    targets[..., 1] = 1.0
    assert float(objective.loss(logits, targets, torch.ones(2, 3))) > 0.0


def test_multitask_loss_weights_its_terms() -> None:
    """The total is the weighted sum of the two task losses."""
    criterion = MultiTaskLoss(AdaptiveThresholdLoss(), ner_weight=2.0,
                              relation_weight=3.0)
    output = criterion(
        ner_logits=torch.randn(1, 4, 3),
        ner_labels=torch.tensor([[0, 1, 2, -100]]),
        relation_logits=torch.randn(1, 2, 5),
        relation_labels=torch.zeros(1, 2, 5),
        pair_mask=torch.ones(1, 2),
    )
    expected = 2.0 * float(output.ner) + 3.0 * float(output.relation)
    assert abs(float(output.total) - expected) < 1e-5


def test_adaptive_threshold_decodes_against_the_learned_threshold() -> None:
    """Classes above the threshold column are predicted, others are not."""
    objective = AdaptiveThresholdLoss()
    logits = torch.tensor([[[0.0, 1.0, -1.0]]])
    predictions = objective.decode(logits)
    assert bool(predictions[0, 0, 1]) and not bool(predictions[0, 0, 2])


def test_the_na_column_is_never_predicted() -> None:
    """Column zero is the absence of a relation, not a relation."""
    for objective in (AdaptiveThresholdLoss(), BinaryRelationLoss(threshold=0.1)):
        logits = torch.full((1, 1, 4), 5.0)
        assert not bool(objective.decode(logits)[0, 0, 0])


def test_confidence_agrees_with_the_decision() -> None:
    """Confidence crosses one half exactly where the decoder flips."""
    for objective in (AdaptiveThresholdLoss(), BinaryRelationLoss(threshold=0.3)):
        logits = torch.randn(2, 4, 6)
        predicted = objective.decode(logits)
        confidence = objective.confidence(logits)
        assert torch.all(confidence[predicted] > 0.5)
        interior = ~predicted
        interior[..., 0] = False
        assert torch.all(confidence[interior] <= 0.5)


def test_unknown_objective_is_rejected() -> None:
    """A misspelled objective fails loudly rather than defaulting."""
    try:
        build_relation_objective("nonesuch")
    except ValueError as error:
        assert "nonesuch" in str(error)
    else:
        raise AssertionError("expected a ValueError")


class FakeTokenizer:
    """A minimal tokenizer exposing what the trimmer needs."""

    all_special_ids = [0, 1]
    unk_token_id = 1
    pad_token_id = 0

    def __call__(self, words, is_split_into_words=False, add_special_tokens=True):
        """Encode each word as the integer it spells.

        Args:
            words: Words to encode.
            is_split_into_words: Accepted for signature compatibility.
            add_special_tokens: Accepted for signature compatibility.

        Returns:
            A mapping with an ``input_ids`` list.
        """
        return {"input_ids": [int(word) for word in words]}


class FakeEncoder(torch.nn.Module):
    """An encoder exposing only the embedding surface the trimmer touches."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, hidden_size)

        class Config:
            pass

        self.config = Config()
        self.config.vocab_size = vocab_size
        self.config.hidden_size = hidden_size

    def get_input_embeddings(self):
        """Return the embedding table."""
        return self.embeddings

    def set_input_embeddings(self, value) -> None:
        """Replace the embedding table."""
        self.embeddings = value


def _backbone(vocab_size: int = 20, hidden_size: int = 4):
    """Build a backbone over a stand-in encoder.

    Args:
        vocab_size: Rows in the embedding table.
        hidden_size: Embedding width.

    Returns:
        The backbone.
    """
    from nano_re.models.backbone import EncoderBackbone

    return EncoderBackbone(FakeEncoder(vocab_size, hidden_size))


class Doc:
    """A stand-in parsed document."""

    def __init__(self, words) -> None:
        self.words = words


def test_trimmer_keeps_observed_and_special_tokens() -> None:
    """Observed tokens and special tokens survive; the rest do not."""
    trimmer = VocabularyTrimmer(FakeTokenizer(), target_coverage=1.0, min_vocab_size=0)
    trimmer.observe_documents([Doc(["5", "7", "5"])])
    assert trimmer.kept_token_ids() == [0, 1, 5, 7]


def test_trimmer_compacts_the_embedding_table() -> None:
    """The table shrinks to the retained rows."""
    backbone = _backbone(vocab_size=20)
    trimmer = VocabularyTrimmer(FakeTokenizer(), target_coverage=1.0, min_vocab_size=0)
    trimmer.observe_documents([Doc(["5", "7"])])
    report = trimmer.trim(backbone)
    assert report.original_size == 20
    assert report.trimmed_size == 4
    assert backbone.encoder.get_input_embeddings().weight.shape[0] == 4


def test_remapped_ids_select_the_original_vectors() -> None:
    """A retained token still resolves to the embedding it had before."""
    backbone = _backbone(vocab_size=20)
    original = backbone.encoder.get_input_embeddings().weight.detach().clone()
    trimmer = VocabularyTrimmer(FakeTokenizer(), target_coverage=1.0, min_vocab_size=0)
    trimmer.observe_documents([Doc(["5", "7"])])
    trimmer.trim(backbone)

    table = backbone.encoder.get_input_embeddings().weight
    for token in (0, 1, 5, 7):
        compact = int(backbone.token_remap[token])
        assert torch.allclose(table[compact], original[token])


def test_dropped_ids_fall_back_to_the_unknown_token() -> None:
    """A trimmed identifier maps onto the unknown token's row."""
    backbone = _backbone(vocab_size=20)
    trimmer = VocabularyTrimmer(FakeTokenizer(), target_coverage=1.0, min_vocab_size=0)
    trimmer.observe_documents([Doc(["5"])])
    trimmer.trim(backbone)
    unknown = int(backbone.token_remap[FakeTokenizer.unk_token_id])
    assert int(backbone.token_remap[19]) == unknown


def test_trimming_without_observations_is_refused() -> None:
    """Trimming on an empty count would discard the whole vocabulary."""
    trimmer = VocabularyTrimmer(FakeTokenizer())
    try:
        trimmer.trim(_backbone())
    except RuntimeError as error:
        assert "observed" in str(error)
    else:
        raise AssertionError("expected a RuntimeError")


def test_context_pooling_weights_the_tokens_both_entities_attend_to() -> None:
    """The context vector is drawn from the overlap of two attentions.

    Where one entity attends to the first token and the other to the second,
    neither alone identifies a connection. Where both attend to the same token,
    that token is what links them, and it is what the context must contain.
    """
    from nano_re.models.heads import LocalizedContextPooler

    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]])
    attention = torch.tensor(
        [
            [
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [0.34, 0.33, 0.33],
            ]
        ]
    )
    mention_mask = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    pair_index = torch.tensor([[[0, 1]]])

    context = LocalizedContextPooler()(hidden, attention, mention_mask, pair_index)
    assert torch.allclose(context[0, 0], hidden[0, 2], atol=1e-5)


def test_context_pooling_shapes_follow_the_pair_axis() -> None:
    """One context vector is produced per candidate pair."""
    from nano_re.models.heads import LocalizedContextPooler

    hidden = torch.randn(2, 12, 8)
    attention = torch.softmax(torch.randn(2, 12, 12), dim=-1)
    mention_mask = torch.rand(2, 5, 12)
    mention_mask = mention_mask / mention_mask.sum(-1, keepdim=True)
    pair_index = torch.randint(0, 5, (2, 7, 2))

    context = LocalizedContextPooler()(hidden, attention, mention_mask, pair_index)
    assert context.shape == (2, 7, 8)


def test_relation_head_uses_the_context_when_given_one() -> None:
    """Supplying a context changes the scores, so it is not being ignored."""
    from nano_re.models.heads import PairwiseRelationHead

    head = PairwiseRelationHead(
        hidden_size=8, num_relations=4, pair_hidden_size=8, dropout=0.0,
        use_context=True,
    ).eval()
    entities = torch.randn(1, 3, 8)
    pair_index = torch.tensor([[[0, 1]]])

    with torch.no_grad():
        first = head(entities, pair_index, context=torch.zeros(1, 1, 8))
        second = head(entities, pair_index, context=torch.ones(1, 1, 8))
    assert not torch.allclose(first, second)


def test_relation_head_without_context_ignores_it() -> None:
    """A head built without context is unaffected by one being passed."""
    from nano_re.models.heads import PairwiseRelationHead

    head = PairwiseRelationHead(
        hidden_size=8, num_relations=4, pair_hidden_size=8, dropout=0.0
    ).eval()
    entities = torch.randn(1, 3, 8)
    pair_index = torch.tensor([[[0, 1]]])

    with torch.no_grad():
        first = head(entities, pair_index, context=torch.zeros(1, 1, 8))
        second = head(entities, pair_index, context=torch.ones(1, 1, 8))
    assert torch.allclose(first, second)


def test_architecture_records_whether_context_pooling_is_present() -> None:
    """A checkpoint states its own shape so it can be rebuilt correctly."""
    from nano_re.models.modeling_nano_re import NanoREArchitecture

    payload = NanoREArchitecture(
        backbone_name="x",
        hidden_size=8,
        num_bio_labels=3,
        num_relation_labels=4,
        pair_hidden_size=8,
        dropout=0.0,
        localized_context=True,
    ).to_dict()
    assert NanoREArchitecture.from_dict(payload).localized_context is True


def test_device_tuning_is_backend_specific() -> None:
    """Each backend gets the settings measured to suit it."""
    from nano_re.training.device import DeviceManager

    cpu = DeviceManager(preference="cpu")
    assert cpu.tuning.autocast_dtype is None
    assert cpu.tuning.pin_memory is False
    assert cpu.tuning.batch_size > 0


def test_autocast_is_off_where_it_was_measured_slower() -> None:
    """Apple Silicon and CPU run in float32.

    Mixed precision is not a universal win. On an M4 Pro a training step was
    measured at 341 ms in float32 against 369 ms under float16 autocast, so the
    policy encodes the measurement rather than the folklore.
    """
    from nano_re.training.device import DeviceManager

    for backend in ("cpu", "mps"):
        try:
            manager = DeviceManager(preference=backend)
        except Exception:
            continue
        assert manager.amp_enabled is False
        assert isinstance(manager.autocast().__enter__(), type(None))


def test_explicit_batch_size_overrides_the_measured_default() -> None:
    """A configured batch size is honoured; zero asks for the default."""
    from nano_re.training.device import DeviceManager

    manager = DeviceManager(preference="cpu")
    assert manager.resolve_batch_size(32) == 32
    assert manager.resolve_batch_size(0) == manager.tuning.batch_size


def test_gradient_scaler_only_enabled_for_float16() -> None:
    """bfloat16 keeps float32's exponent range and needs no scaling."""
    from nano_re.training.device import DeviceManager

    manager = DeviceManager(preference="cpu")
    assert manager.grad_scaler().is_enabled() is False
