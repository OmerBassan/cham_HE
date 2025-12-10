# 🦎 Chameleon: LLM Robustness Benchmark Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

**Evaluate LLM robustness under lexical distortions using semantic paraphrasing.**

Chameleon tests how well language models handle semantically equivalent but lexically varied questions. It applies controlled distortions (μ=0.0 to μ=0.9) while preserving meaning and correct answers, then measures performance degradation.

## ⚡ Requirements

| Component | Provider | Purpose |
|-----------|----------|---------|
| **Distortion Engine** | [Mistral AI](https://console.mistral.ai/) | Generates semantic paraphrases |
| **Target Model** | [OpenAI](https://platform.openai.com/) | Model being evaluated (GPT-4o, GPT-5.1, etc.) |

> **Note:** You need API keys from both providers. Get your Mistral key at [console.mistral.ai](https://console.mistral.ai/) and OpenAI key at [platform.openai.com](https://platform.openai.com/api-keys).

## ✨ Key Features

- 🔬 **Semantic Distortion Engine**: Uses Mistral to generate meaning-preserving paraphrases at 10 intensity levels
- 📊 **Statistical Analysis**: McNemar's tests, confidence intervals, significance testing
- 📈 **Rich Visualizations**: Heatmaps, accuracy plots, degradation analysis
- 🚀 **Batch API Support**: OpenAI & Mistral batch APIs for efficient large-scale evaluation
- 📝 **Executive Reports**: Auto-generated markdown reports with charts and insights

## 📦 Installation

### Option 1: pip install (Recommended)

```bash
# Clone the repository
git clone https://github.com/stevesolun/Chameleon.git
cd Chameleon

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Docker

```bash
# Build the Docker image
docker build -t chameleon .

# Run interactive CLI
docker run -it --rm \
  -v $(pwd)/Projects:/app/Projects \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e MISTRAL_API_KEY=$MISTRAL_API_KEY \
  chameleon python cli.py --help
```

## 🔑 API Keys Setup

You need API keys from:
- **Mistral AI**: [console.mistral.ai](https://console.mistral.ai/) - for distortion generation
- **OpenAI**: [platform.openai.com](https://platform.openai.com/api-keys) - for target model evaluation

Set them as environment variables or the CLI will prompt you:

```bash
export MISTRAL_API_KEY="your-mistral-key"
export OPENAI_API_KEY="your-openai-key"
```

## 🚀 Quick Start

### 1. Create a Project

```bash
python cli.py init
```

Follow the interactive prompts to configure:
- Project name
- Target model (e.g., gpt-5.1, gpt-4o)
- Distortion settings (μ values, distortions per question)
- API keys

### 2. Upload Your Data

When prompted, provide CSV files with questions. See [Data Formats](#-data-formats) below for details.

### 3. Generate Distortions

```bash
python cli.py distort --project MyProject
```

Uses Mistral to create semantic paraphrases at each μ level.

### 4. Evaluate Target Model

```bash
python cli.py evaluate --project MyProject
```

Sends distorted questions to your target model via OpenAI Batch API.

### 5. Run Analysis

```bash
python cli.py analyze --project MyProject
```

Generates statistical analysis, visualizations, and executive report.

## 📋 CLI Commands

```bash
# Project Management
python cli.py init                    # Create new project (interactive)
python cli.py list                    # List all projects
python cli.py status -p PROJECT       # Show project status
python cli.py edit -p PROJECT         # Edit project configuration
python cli.py delete -p PROJECT       # Delete project (double confirmation)

# Distortion & Evaluation
python cli.py distort -p PROJECT      # Generate distortions (requires Mistral)
python cli.py evaluate -p PROJECT     # Evaluate target model (requires OpenAI)

# Analysis
python cli.py analyze -p PROJECT      # Run full analysis

# Help
python cli.py help                    # Show all commands
```

## 📁 Project Structure

```
Chameleon/
├── chameleon/                 # Main package
│   ├── core/                  # Config, project management
│   ├── distortion/            # Mistral-based distortion engine
│   ├── evaluation/            # OpenAI batch evaluation
│   └── analysis/              # Statistics and visualizations
├── Projects/                  # Your evaluation projects
│   └── MyProject/
│       ├── original_data/     # Input CSV files
│       ├── distorted_data/    # Generated distortions
│       ├── results/           # Evaluation results & analysis
│       └── config.yaml        # Project settings
├── cli.py                     # CLI entry point
├── requirements.txt           # Dependencies
└── Dockerfile                 # Docker support
```

## 📊 Understanding μ (Miu) Distortion Levels

| μ Level | Distortion Type | Description |
|---------|-----------------|-------------|
| 0.0 | None (baseline) | Original question unchanged |
| 0.1-0.2 | Minimal | 1-3 word synonyms |
| 0.3-0.4 | Moderate | Phrase restructuring |
| 0.5-0.6 | Mixed | Lexical + structural changes |
| 0.7-0.8 | Heavy | Major paraphrasing |
| 0.9 | Full | Complete reconstruction |

## 📈 Analysis Output

After running `python cli.py analyze --project YourProject`, all outputs are saved to `Projects/YourProject/results/analysis/` (~23 files):

### 📊 Core Metrics (Data + Charts)

| File | Description | Key Insight |
|------|-------------|-------------|
| `01_accuracy_by_miu.csv` | Accuracy data by μ level | Raw numbers for each distortion level |
| `01_accuracy_by_miu.png` | 📈 **Line chart**: accuracy vs distortion | Visualize degradation curve |
| `02_accuracy_by_subject_miu.csv` | Per-subject accuracy data | Which subjects are most vulnerable? |
| `02_subject_ranking.png` | 📊 **Bar chart**: subject performance | Rank subjects by baseline accuracy |
| `02_subject_miu_heatmap.png` | 🔥 **Heatmap**: absolute accuracy (Subject × μ) | See accuracy patterns |
| `02_degradation_heatmap.png` | 🔥 **Heatmap**: % degradation from baseline | Identify vulnerable subjects |
| `03_chameleon_robustness_index.csv` | CRI scores (global + per-subject) | Single metric for model ranking |
| `04_elasticity.csv` | Degradation slope data | Quantify fragility numerically |
| `04_elasticity.png` | 📈 **Scatter + regression**: degradation rate | Visualize slope |
| `05_model_comparison.csv` | Head-to-head comparison table | Compare all metrics |
| `05_model_comparison.png` | 📊 **Scatter plot**: CRI vs accuracy | Compare models visually |

### 🔬 Error Analysis

| File | Description | Key Insight |
|------|-------------|-------------|
| `06_error_taxonomy.json` | Classification: blank, wrong_choice, invalid_format, multiple_options | Where do failures come from? |
| `07_confusion_clusters.json` | TF-IDF + KMeans clustering of failures | Which linguistic patterns cause errors? |

### 📉 Statistical Analysis

| File | Description | Key Insight |
|------|-------------|-------------|
| `08_bootstrap_intervals.csv` | 95% confidence intervals (500 samples) | Are differences statistically significant? |
| `11_mcnemar_distortion.csv` | McNemar's test: μ=0 vs each μ>0 | Paired significance testing |
| `11_mcnemar_distortion.png` | 📊 **Bar chart**: baseline vs distorted (* = p<0.05) | Visualize significant differences |
| `12_mcnemar_subject.csv` | Per-subject McNemar tests | Subject-specific significance |
| `12_mcnemar_subject.png` | 📊 **Bar chart**: per-subject significance | Which subjects show real degradation? |

### 🎯 Advanced Analysis

| File | Description | Key Insight |
|------|-------------|-------------|
| `09_delta_accuracy_heatmap.csv` | Subject × μ degradation matrix (data) | Raw delta values |
| `09_delta_accuracy_heatmap.png` | 🔥 **Heatmap**: change from baseline | Visual: Red = high degradation |
| `10_question_difficulty_tiers.json` | Easy/Medium/Hard/Chameleon Breakers | Find pattern-matching evidence |
| `13_key_insights.png` | 📊 **4-panel summary**: curve + bars + pie + stats | Quick visual overview |
| `EXECUTIVE_REPORT.md` | 📄 **START HERE** - Full findings report | Comprehensive interpretation |

---

## 🔑 Key Metrics Explained

### Chameleon Robustness Index (CRI)
Weighted accuracy that emphasizes high-distortion performance:
```
CRI = Σ(accuracy(μ) × w(μ)) where w(μ) = exp(2.0 × μ) / Σ exp(2.0 × μ)
```
- **CRI > 0.7**: Highly robust
- **CRI 0.5-0.7**: Moderately robust  
- **CRI < 0.5**: Fragile

### Elasticity Slope
Linear regression of accuracy vs μ:
- **Slope ≈ 0**: Robust (stable across distortions)
- **Slope < -0.05**: Fragile (>5% accuracy loss per 0.1 μ)

### Question Difficulty Tiers

| Tier | Definition | Interpretation |
|------|------------|----------------|
| 🟢 **Easy** | ≥80% at μ=0, ≥70% at μ=0.9 | True understanding |
| 🟡 **Medium** | Good at low μ, struggles at high | Partial understanding |
| 🔴 **Hard** | <50% even at μ=0 | Knowledge gap |
| 💀 **Chameleon Breaker** | ≥70% at μ=0, <30% at μ=0.9 | **Surface pattern matching** |

> **Chameleon Breakers** are the most important finding - they reveal questions where the model appears to understand at baseline but fails catastrophically under paraphrasing, indicating reliance on lexical patterns rather than semantic comprehension.

## 🐳 Docker Usage

### Option 1: Docker Compose (Recommended)

```bash
# Set your API keys in .env or export them
export MISTRAL_API_KEY="your-mistral-key"
export OPENAI_API_KEY="your-openai-key"

# Build and run
docker-compose build
docker-compose run chameleon python cli.py init
docker-compose run chameleon python cli.py distort -p MyProject
docker-compose run chameleon python cli.py evaluate -p MyProject
docker-compose run chameleon python cli.py analyze -p MyProject

# Or run analysis only (no API keys needed)
PROJECT=MyProject docker-compose run analyze
```

### Option 2: Docker Direct

```bash
# Build
docker build -t chameleon .

# Run with mounted projects and API keys
docker run -it --rm \
  -v $(pwd)/Projects:/app/Projects \
  -e MISTRAL_API_KEY=$MISTRAL_API_KEY \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  chameleon python cli.py init
```

## 📋 Data Formats

Chameleon works with **closed-answer multiple-choice questions** in CSV format.

### Input Format (Original Data)

Your source data should have these columns:

| Column | Required | Description |
|--------|----------|-------------|
| `subject` | Optional | Category/topic (e.g., "Biology", "History") |
| `question` | **Yes** | The question text |
| `answer_options` | **Yes** | JSON object with options: `{"A": "...", "B": "...", "C": "...", "D": "..."}` |
| `correct_answer` | **Yes** | Correct answer letter(s): `"A"` or `"A, D"` for multiple |
| `question_id` | Optional | Unique identifier (auto-generated if missing) |

**Example:**
```csv
subject,question,answer_options,correct_answer,question_id
Biology,"What is the powerhouse of the cell?","{""A"": ""Nucleus"", ""B"": ""Mitochondria"", ""C"": ""Ribosome"", ""D"": ""Golgi""}",B,BIO_001
```

### Output Format (Results)

The full results CSV includes all processing columns:

| Column | Description |
|--------|-------------|
| `subject` | Category/topic |
| `question_id` | Unique question identifier |
| `question_text` | Original question |
| `options_json` | Answer options as JSON |
| `distorted_question` | Paraphrased version (or original if μ=0) |
| `distortion_id` | Unique ID: `{question_id}_d{N}_m{miu}` |
| `miu` | Distortion level (0.0 - 0.9) |
| `answer` | Correct answer(s) |
| `target_model_name` | Model evaluated (e.g., "gpt-5.1") |
| `target_model_answer` | Model's response |
| `is_correct` | Whether model answered correctly |

## 💡 Tips

### Using Local Models for Distortion

By default, Chameleon uses **Mistral API** for distortion generation (recommended). However, you can configure local models during project setup.

> ⚠️ **Hardware Requirements for Local Models**
> 
> Running local LLMs requires significant computational resources:
> - **GPU**: NVIDIA GPU with 8GB+ VRAM recommended (16GB+ for larger models)
> - **RAM**: 16GB+ system memory
> - **Storage**: 10-50GB for model weights
> - **Time**: Local inference is significantly slower than API calls
> 
> If you don't have a powerful workstation, stick with the API option. It's faster and more reliable for large datasets.

### Multiple Correct Answers

Chameleon supports questions with multiple correct answers. Use comma-separated letters:
- Single answer: `"B"`
- Multiple answers: `"A, D"` (order doesn't matter, case-insensitive)

The evaluation uses smart comparison: `"A, D"` equals `"D, A"` equals `"a,d"`.

## 📄 Citation

If you use Chameleon in your research, please cite:

```bibtex
@software{chameleon2025,
  title={Chameleon: LLM Robustness Testing Framework},
  author={Steve Solun},
  year={2025},
  url={https://github.com/stevesolun/Chameleon}
}
```

**Foundational Work:**

```bibtex
@article{cohen2025forget,
  title={Forget What You Know about LLMs Evaluations - LLMs are Like a Chameleon},
  author={Cohen-Inger, Nurit and Elisha, Yehonatan and Shapira, Bracha and Rokach, Lior and Cohen, Seffi},
  journal={arXiv preprint arXiv:2502.07445},
  year={2025},
  url={https://arxiv.org/abs/2502.07445}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built with ❤️ for the AI research community**
