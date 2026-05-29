"""Fixed 75-question evaluation set over the LLM reasoning and evaluation corpus.

Questions span factual lookups, method comparisons, and methodological "how"
questions, mirroring the three query types the router classifies. Each question
is answerable from the corpus, which is what makes the RAGAS faithfulness
metric meaningful: a hallucinated answer is detectable against a known source.

The target metrics (faithfulness and context precision) do not require gold
reference answers, so ``reference`` is left empty; supply references here to
enable reference-based RAGAS metrics.
"""

# Each record: {"question": str, "category": str, "reference": str}.
EVALUATION_QUESTIONS: list[dict] = [
    # --- Chain-of-thought prompting ---
    {"question": "What is chain-of-thought prompting and how does it differ from standard prompting?", "category": "factual", "reference": ""},
    {"question": "How does chain-of-thought prompting affect performance on arithmetic reasoning tasks?", "category": "methodological", "reference": ""},
    {"question": "What is zero-shot chain-of-thought prompting and what trigger phrase enables it?", "category": "factual", "reference": ""},
    {"question": "How does self-consistency improve over greedy chain-of-thought decoding?", "category": "comparative", "reference": ""},
    {"question": "What sampling strategy does self-consistency use to aggregate reasoning paths?", "category": "methodological", "reference": ""},
    {"question": "Does chain-of-thought prompting help small language models, and if not, why?", "category": "factual", "reference": ""},
    {"question": "What is the relationship between model scale and the emergence of chain-of-thought reasoning?", "category": "comparative", "reference": ""},
    {"question": "How does least-to-most prompting decompose complex problems?", "category": "methodological", "reference": ""},
    {"question": "What is the difference between chain-of-thought and tree-of-thoughts prompting?", "category": "comparative", "reference": ""},
    {"question": "How does program-aided language modeling differ from natural-language chain-of-thought?", "category": "comparative", "reference": ""},
    {"question": "What evidence suggests chain-of-thought rationales may not faithfully reflect model computation?", "category": "factual", "reference": ""},
    {"question": "How does few-shot exemplar selection affect chain-of-thought performance?", "category": "methodological", "reference": ""},
    # --- Reasoning benchmarks ---
    {"question": "What reasoning capabilities does the GSM8K benchmark measure?", "category": "factual", "reference": ""},
    {"question": "How are tasks in BIG-Bench selected and categorized?", "category": "methodological", "reference": ""},
    {"question": "What is BIG-Bench Hard and why was it constructed?", "category": "factual", "reference": ""},
    {"question": "What domains does the MMLU benchmark cover?", "category": "factual", "reference": ""},
    {"question": "How does the HellaSwag benchmark test commonsense reasoning?", "category": "methodological", "reference": ""},
    {"question": "What distinguishes the ARC benchmark from simpler question-answering datasets?", "category": "comparative", "reference": ""},
    {"question": "What is the purpose of the DROP benchmark for reading comprehension?", "category": "factual", "reference": ""},
    {"question": "How does the StrategyQA dataset test implicit multi-step reasoning?", "category": "methodological", "reference": ""},
    {"question": "What contamination concerns affect evaluation on public benchmarks?", "category": "factual", "reference": ""},
    {"question": "How is answer accuracy computed on GSM8K given free-form model outputs?", "category": "methodological", "reference": ""},
    {"question": "What limitations does multiple-choice evaluation introduce when assessing reasoning?", "category": "factual", "reference": ""},
    {"question": "How does the MATH dataset differ from GSM8K in difficulty and structure?", "category": "comparative", "reference": ""},
    {"question": "What is the role of held-out test splits in preventing benchmark overfitting?", "category": "methodological", "reference": ""},
    {"question": "How do dynamic or adversarial benchmarks address the saturation of static ones?", "category": "comparative", "reference": ""},
    # --- Evaluation methodology ---
    {"question": "What are the trade-offs between automatic metrics and human evaluation for generated text?", "category": "comparative", "reference": ""},
    {"question": "How does using an LLM as a judge introduce bias into evaluation?", "category": "factual", "reference": ""},
    {"question": "What is position bias in pairwise LLM-as-judge evaluation?", "category": "factual", "reference": ""},
    {"question": "How can self-enhancement bias affect a model grading its own outputs?", "category": "methodological", "reference": ""},
    {"question": "What is the difference between reference-based and reference-free evaluation?", "category": "comparative", "reference": ""},
    {"question": "How does BLEU correlate with human judgments of quality on reasoning tasks?", "category": "factual", "reference": ""},
    {"question": "Why is exact-match accuracy insufficient for evaluating open-ended answers?", "category": "factual", "reference": ""},
    {"question": "What is the purpose of measuring inter-annotator agreement in human evaluation?", "category": "methodological", "reference": ""},
    {"question": "How do pass@k metrics evaluate code-generation models?", "category": "methodological", "reference": ""},
    {"question": "What is the difference between intrinsic and extrinsic evaluation?", "category": "comparative", "reference": ""},
    {"question": "How does prompt sensitivity complicate fair model comparison?", "category": "methodological", "reference": ""},
    {"question": "What role does sampling temperature play in the variance of evaluation results?", "category": "factual", "reference": ""},
    {"question": "How can evaluation be made robust to the ordering of answer options?", "category": "methodological", "reference": ""},
    {"question": "What is the value of reporting confidence intervals over benchmark scores?", "category": "factual", "reference": ""},
    # --- Calibration and uncertainty ---
    {"question": "What does it mean for a language model to be well-calibrated?", "category": "factual", "reference": ""},
    {"question": "How is expected calibration error computed?", "category": "methodological", "reference": ""},
    {"question": "How does reinforcement learning from human feedback affect model calibration?", "category": "factual", "reference": ""},
    {"question": "What is the difference between aleatoric and epistemic uncertainty in model predictions?", "category": "comparative", "reference": ""},
    {"question": "How can verbalized confidence be elicited from a language model?", "category": "methodological", "reference": ""},
    {"question": "What is selective prediction and how does it use confidence thresholds?", "category": "factual", "reference": ""},
    {"question": "How does temperature scaling improve calibration?", "category": "methodological", "reference": ""},
    {"question": "What is the relationship between model size and calibration?", "category": "comparative", "reference": ""},
    {"question": "How can token-level probabilities be used to estimate answer confidence?", "category": "methodological", "reference": ""},
    {"question": "Why might a model be accurate but poorly calibrated?", "category": "factual", "reference": ""},
    {"question": "What methods detect when a model is likely to hallucinate?", "category": "methodological", "reference": ""},
    {"question": "How does self-consistency relate to confidence estimation?", "category": "comparative", "reference": ""},
    # --- Reasoning failure modes ---
    {"question": "What are common failure modes of LLMs on multi-step arithmetic?", "category": "factual", "reference": ""},
    {"question": "How does the order of premises affect logical reasoning performance?", "category": "methodological", "reference": ""},
    {"question": "What is the reversal curse in language model reasoning?", "category": "factual", "reference": ""},
    {"question": "How do distractor sentences degrade reasoning accuracy?", "category": "methodological", "reference": ""},
    {"question": "What is sycophancy in language model responses?", "category": "factual", "reference": ""},
    {"question": "How can chain-of-thought produce a correct answer with incorrect reasoning?", "category": "factual", "reference": ""},
    {"question": "What is unfaithful reasoning and how is it measured?", "category": "methodological", "reference": ""},
    {"question": "How do models fail on compositional generalization tasks?", "category": "factual", "reference": ""},
    {"question": "What types of errors are revealed by perturbing benchmark inputs?", "category": "methodological", "reference": ""},
    {"question": "How does prompt injection compromise reasoning reliability?", "category": "factual", "reference": ""},
    {"question": "Why do models struggle with counting and length-generalization tasks?", "category": "factual", "reference": ""},
    {"question": "How does spurious correlation in training data manifest as reasoning failure?", "category": "methodological", "reference": ""},
    # --- Techniques and methods ---
    {"question": "How does retrieval-augmented generation reduce hallucination?", "category": "methodological", "reference": ""},
    {"question": "What is the role of a reranker in a retrieval pipeline?", "category": "factual", "reference": ""},
    {"question": "How does reciprocal rank fusion combine multiple retrieval results?", "category": "methodological", "reference": ""},
    {"question": "What is the difference between dense and sparse retrieval?", "category": "comparative", "reference": ""},
    {"question": "How does the ReAct framework interleave reasoning and acting?", "category": "methodological", "reference": ""},
    {"question": "What is Reflexion and how does it use self-generated feedback?", "category": "factual", "reference": ""},
    {"question": "How does self-critique or verification improve answer accuracy?", "category": "methodological", "reference": ""},
    {"question": "What is the difference between a cross-encoder and a bi-encoder in retrieval?", "category": "comparative", "reference": ""},
    {"question": "How does instruction tuning affect zero-shot reasoning performance?", "category": "factual", "reference": ""},
    {"question": "What is the effect of the decoding strategy on reasoning quality?", "category": "methodological", "reference": ""},
    {"question": "How can intermediate reasoning steps be evaluated independently of the final answer?", "category": "methodological", "reference": ""},
]


def get_questions(limit: int | None = None) -> list[dict]:
    """Return the evaluation questions, optionally truncated.

    Parameters
    ----------
    limit : int or None, optional
        If given, return only the first ``limit`` questions. The
        implementation sequencing recommends a 20-question smoke run before the
        full 75-question suite.

    Returns
    -------
    list of dict
        Question records.
    """
    if limit is None:
        return list(EVALUATION_QUESTIONS)
    return EVALUATION_QUESTIONS[:limit]
