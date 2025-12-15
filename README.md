# 🦎 Chameleon: LLM Robustness Benchmark Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

**Evaluate the Robustness of LLM Code Generation under Semantic Distortions.**

Chameleon challenges Large Language Models by testing their consistency. It takes standard coding problems (like HumanEval), semantically distorts the problem description while preserving the logic, and measures if the model can still generate functionally correct code.

If a model can solve the original problem but fails when the same problem is rephrased (distorted), it indicates **brittleness** and reliance on surface-level pattern matching rather than true semantic understanding.

## ⚡ Core Concept

1.  **Original Task**: A standard coding problem (e.g., "Write a function to sort a list").
2.  **Distortion Engine**: Chameleon uses a high-quality LLM (e.g., Mistral Large) to rewrite the problem description at varying levels of "Semantic Noise" ($\mu$).
    -   $\mu=0.0$: Original text.
    -   $\mu=0.5$: Moderate paraphrasing.
    -   $\mu=1.0$: Complete stylistic rewrite (e.g., "Imagine you are a wizard sorting potions...").
3.  **Code Generation**: The Target Model (the model being evaluated) generates code for these distorted prompts.
4.  **Functional Evaluation**: The generated code is executed against the **original unit tests**. Since the logic is unchanged, the code *should* still pass.
5.  **Analysis**: We measure the "Degradation Curve"—how fast performance drops as distortion increases.

## ✨ Key Features

-   **🧬 Semantic Distortion Engine**: Tunable distortion using state-of-the-art LLMs (Mistral, OpenAI) or local models (Ollama, HuggingFace).
-   **🛠 Functional Correctness**: Evaluates code by actually running it (Sandboxed execution of generated Python code against test cases).
-   **📊 Robustness Metrics**:
    -   **Pass@k Stability**: How pass rates change across $\mu$.
    -   **CRI (Chameleon Robustness Index)**: A weighted score of model stability.
    -   **Elasticity**: The slope of performance degradation.
-   **🐳 Dockerized Workflow**: complete isolation for safe code execution.
-   **📈 Rich Reporting**: Generates heatmaps, comparison charts, and executive summaries.

## 📦 Installation

### Option 1: pip install (Recommended)

```bash
# Clone the repository
git clone https://github.com/stevesolun/Chameleon.git
cd Chameleon

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install human-eval  # Required for code evaluation
```

### Option 2: Docker (Best for Evaluation)
Since Chameleon executes generated code, Docker is highly recommended for isolation.

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

## 🚀 Quick Start

### 1. Initialize a Project

```bash
python cli.py init
```
Follow the prompts to set up your project name, target model, and distortion settings.

### 2. Prepare Data (HumanEval Format)
Chameleon expects a JSONL file with coding problems.
Format:
```json
{"task_id": "Task/1", "prompt": "def add(a, b):", "test": "assert add(1,2)==3", "entry_point": "add"}
```
Place your `.jsonl` file in `Projects/<YourProject>/original_data/`.

### 3. Generate Distortions

```bash
python cli.py distort --project MyProject
```
This uses the configured Distortion Engine to create paraphrased versions of your prompts.

### 4. Evaluate Target Model

```bash
python cli.py evaluate --project MyProject
```
This runs the full pipeline:
1.  Generates code using the Target Model (e.g., GPT-4o, Llama-3).
2.  Executes the code against the unit tests.
3.  Calculates Pass@1 for every distortion level.

### 5. Run Analysis

```bash
python cli.py analyze --project MyProject
```
Generates charts and reports in `Projects/MyProject/results/analysis/`.

## 📁 Project Structure

```
Chameleon/
├── chameleon/                 # Core Framework
│   ├── distortion/            # Paraphrasing logic (Mistral/Local)
│   ├── evaluation/            # Code execution & Pass@k calculation
│   └── analysis/              # Data Science & Visualization
├── Projects/                  # user workspaces
│   └── MyProject/
│       ├── original_data/     # Input JSONL source
│       ├── distorted_data/    # Paraphrased prompts
│       ├── results/           # Execution logs & Analysis output
│       └── config.yaml        # Project configuration
├── cli.py                     # Main Entry Point
└── toy_workflow_demo.py       # Simple end-to-end test script
```

## 📊 Understanding The Metrics

### $\mu$ (Miu) - Distortion Level
-   **0.0 (Baseline)**: The original, clean problem description.
-   **0.3 (Low)**: Minor synonym swaps and sentence restructuring.
-   **0.6 (Medium)**: Significant rewording, changing context/setting.
-   **0.9 (High)**: Complete narrative overhaul, abstract analogies, complex sentence structures.

### CRI (Chameleon Robustness Index)
A score from 0 to 1 indicating how well the model maintains its capability under pressure.
-   **> 0.8**: Extremely Robust.
-   **< 0.5**: Fragile (Model is "overfitting" to standard prompt formats).

## 💡 Configuration

You can configure the framework in `config.yaml` or via the CLI wizard.

### Supported Distortion Engines
-   **API**: Mistral AI (Recommended), OpenAI, Anthropic.
-   **Local**: HuggingFace (Transformers), Ollama (for offline use).

### Supported Target Models
-   Any model compatible with OpenAI's API format.
-   Local models via vLLM or Ollama.

## 📄 Citation

If you use Chameleon in your research, please cite:

```bibtex
@software{chameleon2025,
  title={Chameleon: LLM Code Robustness Benchmark},
  author={Steve Solun},
  year={2025},
  url={https://github.com/stevesolun/Chameleon}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
