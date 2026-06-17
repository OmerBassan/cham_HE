# Model Accuracy Degradation Analysis

I have created a professional visualization script to analyze the performance drop between original and distorted inputs for your models.

## 📊 Results Summary

Based on the data provided (at $\mu = 0.5$):

| Model | Original Accuracy | Distorted Accuracy | **Accuracy Drop (%)** |
| :--- | :--- | :--- | :--- |
| **Gemini 1.5 Flash** | 72.00% | 66.67% | **-7.4%** |
| **Pixtral Latest** | 72.00% | 60.56% | **-15.9%** |

> [!NOTE]
> **Gemini 1.5 Flash** demonstrated significantly higher robustness, with nearly half the accuracy degradation compared to **Pixtral Latest** under the same distortion conditions.

## 🛠️ How to Use the Script

The script [visualize_results.py](file:///c:/Users/Omer%20Bassan/Desktop/%D7%94%D7%A0%D7%93%D7%A1%D7%AA%20%D7%A0%D7%AA%D7%95%D7%A0%D7%99%D7%9D/%D7%A9%D7%A0%D7%94%20%D7%93%20%D7%A1%D7%9E%20%D7%90/%D7%A4%D7%A8%D7%95%D7%99%D7%A7%D7%98%20%D7%9E%D7%A1%D7%9B%D7%9D/cham_HE/visualize_results.py) uses `matplotlib` and `seaborn` to generate high-resolution charts.

1. **Run the script**:
   ```powershell
   python visualize_results.py
   ```
2. **View the output**:
   The generated chart is saved locally as: `benchmarks/accuracy_downgrade_report.png`

## 🎨 Visualization Preview

Below is a premium mockup of how these results look in a research dashboard:

![Benchmark Downgrade Dashboard](C:\Users\Omer Bassan\.gemini\antigravity\brain\272a2094-a005-4cb8-9c52-35537dbe8676\benchmark_downgrade_dashboard_1777479568867.png)

---

### Key Observations
*   **Uniform Baseline**: Both models started with an identical VQA accuracy of 72% on the original tasks.
*   **Distortion Impact**: The distortion significantly impacted Pixtral, leading to a much steeper decline in performance.
*   **Exact Match**: Although the primary focus was accuracy, the Exact Match (EM) scores followed a similar trend, confirming the reliability of the accuracy metric.
