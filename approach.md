
**Input:**
- Natural language query or job description
- Optional parameters such as `top_k`

**Output:**
- Ranked list of recommended assessments
- Metadata including test type, duration, category, and synthetic flag

The API layer is intentionally lightweight, delegating logic to the core recommendation engine.

---

## 8. Evaluation Methodology

### 8.1 Metric Selection

The system is evaluated using **Recall@K** on the provided training dataset. Recall@K measures whether relevant assessments appear within the top-K recommendations, making it suitable for retrieval and recommendation tasks.

### 8.2 Observations

- Semantic retrieval provides a reasonable baseline
- Reranking with duration and domain heuristics consistently improves recall
- Absolute metric values are influenced by catalog size and label sparsity

The evaluation primarily validates **pipeline correctness and behavior**, rather than maximizing absolute scores.

---

## 9. Limitations & Future Work

### Current Limitations
- Limited catalog size due to crawling restrictions
- Heuristic-based reranking rather than learned ranking
- No user feedback loop

### Future Improvements
- Full catalog ingestion with authenticated access
- Learning-to-rank or cross-encoder reranking models
- User interaction and feedback incorporation
- Online evaluation and A/B testing

---

## 10. Conclusion

This project demonstrates a **robust, production-oriented semantic recommendation system** that balances modern NLP techniques with practical engineering constraints. The solution is modular, interpretable, and resilient to real-world data availability issues.

Key design priorities include:
- Correctness over over-engineering
- Transparency over opaque heuristics
- Extensibility for future enhancements

---

### Submission Artifacts

- GitHub repository containing full source code
- `submission_predictions.csv`
- Reproducible environment via `requirements.txt`
