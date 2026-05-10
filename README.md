\## Results: Ablation Study



| Step | Change | Execution Accuracy |

|---|---|---|

| Phi-2 2.7B baseline | Unaugmented schema | 40% |

| Llama 3.1 8B | Unaugmented schema | 70.89% |

| + Column order fix | Normalise result tuple order | 72% |

| + Augmented inference | Sample rows, 142 examples | 75% |

| + Token count fix | All 301 wrong/error examples | 77.27% |

| + Augmented fine-tuning | 6,607 training examples | 81% |

| + Agentic loop + CoT | Chain-of-thought + retry on all failures | 84% |

