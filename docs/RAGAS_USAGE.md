# RAGAS Evaluation Usage Guide

This guide explains how to use RAGAS (Retrieval Augmented Generation Assessment) evaluation with your multimodal research RAG system.

## Quick Reference

**Common Issues Fixed:**
- ✅ **ImportError `ragas.configs`**: Fixed in code - use latest version
- ⚠️ **OpenAI API Key for `answer_relevancy`**: This metric requires embeddings (OpenAI API). Other metrics work with DeepSeek only.

**Recommended Commands:**
```bash
# Without OpenAI API (use metrics that don't need embeddings)
uv run ark evaluate --metrics ragas --ragas-metrics faithfulness -n 5

# With OpenAI API (can use all metrics including answer_relevancy)
export OPENAI_API_KEY=your-key-here
uv run ark evaluate --metrics all -n 5
```

## Table of Contents

- [Overview](#overview)
- [What is RAGAS?](#what-is-ragas)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Commands](#cli-commands)
- [RAGAS Metrics Explained](#ragas-metrics-explained)
- [Examples](#examples)
- [Interpreting Results](#interpreting-results)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## Overview

RAGAS provides **LLM-judged evaluation** of your RAG system, measuring:
- **Faithfulness**: Are responses factually consistent with retrieved context?
- **Answer Relevancy**: Do responses address the user's question?
- **Context Precision**: Are retrieved contexts relevant?
- **Context Recall**: Was all relevant information retrieved?

This complements your existing simple metrics (Precision@K, Recall@K, MRR) by using LLMs to evaluate **quality** rather than just **retrieval**.

---

## What is RAGAS?

**RAGAS** = Retrieval Augmented Generation Assessment

- Uses **LLM-as-a-judge** to evaluate RAG systems
- Measures **generation quality** (not just retrieval)
- Provides **human-aligned metrics** that correlate with user satisfaction
- Developed by Exploding Gradients (https://github.com/explodinggradients/ragas)

### Why RAGAS?

| Metric Type | What It Measures | Example |
|-------------|------------------|---------|
| **Simple Metrics** | Retrieval quality | "Did we find the right document?" |
| **RAGAS Metrics** | Generation quality | "Is the response factually accurate?" |

**For research systems like yours**, RAGAS is critical because:
- You care about **grounding** (Faithfulness)
- You care about **answer quality** (Answer Relevancy)
- You need to catch **hallucinations** (Faithfulness < 0.8)

---

## Installation

RAGAS is already installed in your project.

```bash
# Check RAGAS version
uv run python -c "import ragas; print(f'RAGAS {ragas.__version__}')"

# Update RAGAS to latest
uv sync --upgrade
```

### Dependencies

RAGAS requires:
- ✅ `ragas>=0.2.0`
- ✅ `langchain-openai`
- ✅ **DeepSeek API key** (for most metrics) - Already configured in `.env`
- ⚠️ **OpenAI API key** (ONLY for `answer_relevancy` metric)

**Important:** The `answer_relevancy` metric requires embeddings, which defaults to using OpenAI's API. If you don't have an OpenAI API key, you can still use the other three metrics (`faithfulness`, `context_precision`, `context_recall`) with just your DeepSeek API key.

---

## Quick Start

### 1. Run Simple Evaluation (Backward Compatible)

```bash
# Runs only simple metrics (Precision@K, Recall@K, MRR)
uv run ark evaluate -n 20
```

### 2. Run RAGAS Evaluation

```bash
# Runs only RAGAS metrics (Faithfulness, Answer Relevancy)
uv run ark evaluate --metrics ragas -n 50
```

---

## RAGAS Metrics Explained

### 1. Faithfulness ⭐⭐⭐⭐⭐

**What it measures:** Factual consistency of the response with retrieved context (Grounding).

**How it works:**
1. Extracts factual claims/findings from the response
2. Verifies each finding against retrieved contexts using LLM
3. Calculates: `(Supported Findings) / (Total Findings)`

**Score interpretation:**
- **0.8-1.0** (Excellent): Responses are well-grounded in context
- **0.6-0.8** (Good): Minor hallucinations
- **<0.4** (Poor): Major factual errors

**Why it matters for research:**
- Directly measures your system's reliability
- Catches hallucinations before they impact research findings
- Critical for academic and professional use

---

## FAQ

### Q: Can I use RAGAS with my own dataset?

**A:** Yes! The system is designed to work with any question-answer-context dataset. If using a CSV:
- `question` → research question
- `ground_truth` → expected answer/finding
- `contexts` → retrieved documents

---

**Last Updated:** 2026-02-23 (Updated for ARK focus)
