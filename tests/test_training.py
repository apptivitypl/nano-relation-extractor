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


def test_cuda_batch_size_scales_with_card_memory() -> None:
    """A small card gets a small batch, since memory binds before throughput."""
    from nano_re.training.device import DeviceManager

    thresholds = [(8e9, 8), (12e9, 12), (16e9, 16), (24e9, 24), (80e9, 32)]
    for total, expected in thresholds:
        class Properties:
            total_memory = total

        original = torch.cuda.get_device_properties
        torch.cuda.get_device_properties = lambda index: Properties()
        try:
            assert DeviceManager._cuda_batch_size() == expected
        finally:
            torch.cuda.get_device_properties = original


def test_cuda_batch_size_falls_back_when_the_card_cannot_be_queried() -> None:
    """An unreadable device yields a conservative batch rather than an error."""
    from nano_re.training.device import DeviceManager

    original = torch.cuda.get_device_properties

    def explode(index):
        raise RuntimeError("no device")

    torch.cuda.get_device_properties = explode
    try:
        assert DeviceManager._cuda_batch_size() == 8
    finally:
        torch.cuda.get_device_properties = original


def test_accumulation_reaches_a_constant_effective_batch() -> None:
    """A small batch is compensated so the optimiser step is device independent."""
    from nano_re.training.device import DeviceManager

    manager = DeviceManager(preference="cpu")
    target = manager.tuning.effective_batch_size
    for batch in (4, 8, 16, 32):
        accumulation = manager.resolve_accumulation(0, batch)
        assert abs(batch * accumulation - target) <= batch // 2
    assert manager.resolve_accumulation(7, 8) == 7


def test_batch_probing_halves_until_a_step_fits() -> None:
    """Probing shrinks the batch rather than failing an hour into a run."""
    from nano_re.training.device import DeviceManager

    attempted: list[int] = []

    def probe(size: int) -> None:
        attempted.append(size)
        if size > 4:
            raise torch.OutOfMemoryError("CUDA out of memory")

    assert DeviceManager(preference="cpu").fit_batch_size(probe, 16) == 4
    assert attempted == [16, 8, 4]


def test_batch_probing_reraises_unrelated_failures() -> None:
    """A bug in the model is not silently treated as a memory limit."""
    from nano_re.training.device import DeviceManager

    def probe(size: int) -> None:
        raise RuntimeError("shape mismatch")

    try:
        DeviceManager(preference="cpu").fit_batch_size(probe, 8)
    except RuntimeError as error:
        assert "shape mismatch" in str(error)
    else:
        raise AssertionError("expected the original error")


def test_batch_probing_gives_up_with_actionable_advice() -> None:
    """When nothing fits, the message names what to change."""
    from nano_re.training.device import DeviceManager

    def probe(size: int) -> None:
        raise torch.OutOfMemoryError("CUDA out of memory")

    try:
        DeviceManager(preference="cpu").fit_batch_size(probe, 8)
    except RuntimeError as error:
        assert "NANO_RE_MAX_SEQUENCE_LENGTH" in str(error)
    else:
        raise AssertionError("expected a RuntimeError")


def test_relation_tail_is_pruned_by_coverage() -> None:
    """Rare relations are dropped without anyone choosing a threshold."""
    from nano_re.schema import RelationInventory

    inventory = RelationInventory()
    for index in range(5):
        for _ in range(1000):
            inventory.add(f"P{index}")
    for index in range(5, 500):
        inventory.add(f"P{index}")

    assert len(inventory) == 500
    assert inventory.to_schema(coverage=1.0).num_relation_labels - 1 == 500
    kept = inventory.to_schema(coverage=0.99).num_relation_labels - 1
    assert 5 <= kept < 500


def test_several_gpus_are_not_used_at_once() -> None:
    """Only the first card is used, deliberately.

    DataParallel replicates a module with its parameters detached from their
    registration, so a replica's ``parameters()`` is empty and any model reading
    ``self.device`` during forward fails with a bare StopIteration. The encoder
    here does exactly that, which was observed on two T4s.
    """
    from nano_re.training.device import DeviceManager

    manager = DeviceManager(preference="cpu")
    sentinel = object()
    assert manager.parallelise(sentinel) is sentinel


def test_bfloat16_needs_hardware_support_not_merely_availability() -> None:
    """Turing reports bfloat16 support and emulates it, so capability decides.

    ``torch.cuda.is_bf16_supported`` answers yes on a T4, which then runs
    bfloat16 slower than float32. Ampere and later implement it in hardware.
    """
    from nano_re.training.device import DeviceManager

    original = torch.cuda.get_device_capability
    try:
        torch.cuda.get_device_capability = lambda index=0: (7, 5)
        assert DeviceManager._bfloat16_is_native() is False
        torch.cuda.get_device_capability = lambda index=0: (8, 6)
        assert DeviceManager._bfloat16_is_native() is True
    finally:
        torch.cuda.get_device_capability = original






def test_an_unsupported_card_is_not_selected() -> None:
    """A visible but unusable GPU is skipped with an explanation.

    Kaggle's PyTorch build dropped kernels for Pascal, so a P100 is visible,
    reports 16 GB, and then fails at the first allocation with a message that
    names neither the card nor the cause. Detecting it up front turns hours of
    confusion into one sentence.
    """
    from nano_re.training.device import DeviceManager

    original = (
        torch.cuda.is_available,
        torch.cuda.get_device_capability,
        torch.cuda.get_device_name,
        torch.cuda.get_arch_list,
    )
    torch.cuda.is_available = lambda: True
    torch.cuda.get_device_capability = lambda index=0: (6, 0)
    torch.cuda.get_device_name = lambda index=0: "Tesla P100-PCIE-16GB"
    torch.cuda.get_arch_list = lambda: ["sm_70", "sm_75", "sm_80"]
    try:
        backend, warning = DeviceManager._detect_backend()
        assert backend != "cuda"
        assert "sm_60" in warning and "P100" in warning
        assert "T4" in warning
    finally:
        (
            torch.cuda.is_available,
            torch.cuda.get_device_capability,
            torch.cuda.get_device_name,
            torch.cuda.get_arch_list,
        ) = original


def test_a_supported_card_is_selected_without_complaint() -> None:
    """A card the build has kernels for is used silently."""
    from nano_re.training.device import DeviceManager

    original = (
        torch.cuda.is_available,
        torch.cuda.get_device_capability,
        torch.cuda.get_device_name,
        torch.cuda.get_arch_list,
    )
    torch.cuda.is_available = lambda: True
    torch.cuda.get_device_capability = lambda index=0: (7, 5)
    torch.cuda.get_device_name = lambda index=0: "Tesla T4"
    torch.cuda.get_arch_list = lambda: ["sm_70", "sm_75", "sm_80"]
    try:
        backend, warning = DeviceManager._detect_backend()
        assert backend == "cuda"
        assert warning == ""
    finally:
        (
            torch.cuda.is_available,
            torch.cuda.get_device_capability,
            torch.cuda.get_device_name,
            torch.cuda.get_arch_list,
        ) = original


def test_an_unreadable_architecture_list_does_not_block_cuda() -> None:
    """When the build reports no architectures, CUDA is not refused.

    Some builds return an empty list. That means the check cannot be made, not
    that nothing is supported, and refusing on it would disable working GPUs.
    """
    from nano_re.training.device import DeviceManager

    original = (torch.cuda.is_available, torch.cuda.get_arch_list)
    torch.cuda.is_available = lambda: True
    torch.cuda.get_arch_list = lambda: []
    try:
        usable, reason = DeviceManager._cuda_is_usable()
        assert usable and reason == ""
    finally:
        torch.cuda.is_available, torch.cuda.get_arch_list = original
