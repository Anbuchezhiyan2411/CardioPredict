# CardioPredict: System, Software & Environment Requirements

This document provides the hardware, software, environment, and library specifications for developing, executing, and deploying the **CardioPredict** project.

---

## 1. Hardware Requirements

### Minimum Hardware Specification (Development & Local Inference)
* **Processor (CPU)**: Dual-core x86_64 or ARM64 processor (Intel Core i3 4th Gen / AMD Ryzen 3 / Apple Silicon M1 or equivalent).
* **Random Access Memory (RAM)**: Minimum **4 GB RAM** (8 GB recommended for seamless multi-model training and visualization generation).
* **Storage**: At least **1 GB of available disk storage** (covers dataset, Python virtual environment, serialised models, generated plots, and documentation).
* **Display**: Standard 1080p ($1920 \times 1080$) monitor or laptop display for optimal browser visualization of the analytics dashboard.
* **Network**: Active internet connection required only during the initial installation of Python packages via `pip`. Once installed, CardioPredict runs 100% offline.

### Recommended Hardware Specification (Production / Server Hosting)
* **Processor (CPU)**: Quad-core modern processor ($2.0\text{ GHz}+$ clock speed).
* **Memory (RAM)**: **8 GB to 16 GB DDR4/DDR5 RAM**.
* **Storage**: Solid State Drive (NVMe or SATA SSD) with at least 5 GB free disk space.

---

## 2. Software Requirements

### Supported Operating Systems
* **Microsoft Windows**: Windows 10 (64-bit), Windows 11 (64-bit).
* **Linux Distributions**: Ubuntu 20.04 LTS / 22.04 LTS / 24.04 LTS, Debian 11/12, Fedora 38+, Arch Linux.
* **Apple macOS**: macOS Monterey (12.x), macOS Ventura (13.x), macOS Sonoma (14.x) on Intel or Apple Silicon (M1/M2/M3).

### Python Version Support
* **Target Version**: **Python 3.10.x, 3.11.x, 3.12.x, or 3.13.x** (64-bit architecture).
* *Note: Tested and verified on Python 3.13.6 (64-bit).*
* Python package installer: `pip` version 23.0 or higher.

### Recommended Integrated Development Environments (IDEs)
* **Visual Studio Code (VS Code)** (Highly Recommended):
  * Recommended Extensions: Python (Microsoft), Pylance, Jupyter, HTML CSS Support.
* **PyCharm**: Community or Professional Edition.
* **JupyterLab / Jupyter Notebook**: For interactive exploration of `notebooks/CardioPredict_EDA.ipynb`.

---

## 3. Python Package Dependencies & Libraries

The following dependencies are declared in `requirements.txt`:

| Package | Minimum Version | Verified Version | Purpose & Clinical Role |
|---|:---:|:---:|---|
| **Python** | `3.10` | `3.13.6` | Core programming language runtime. |
| **pandas** | `>= 2.0.0` | `3.1.1` | Ingestion, data cleaning, tabular reshaping, and descriptive profiling. |
| **numpy** | `>= 1.25.0` | `2.5.4` | Vectorized numerical array computations and mathematical operations. |
| **scipy** | `>= 1.11.0` | `1.18.1` | Inferential statistics: Two-sample $t$-tests, Chi-square tests, and ANOVA. |
| **scikit-learn** | `>= 1.3.0` | `1.9.1` | Model training, preprocessing (`StandardScaler`), metrics, and benchmarking. |
| **Flask** | `>= 3.0.0` | `3.1.3` | Lightweight WSGI web framework for serving predictions and views. |
| **joblib** | `>= 1.3.0` | `1.6.0` | High-efficiency disk serialization and deserialization of NumPy-based estimators. |
| **matplotlib** | `>= 3.8.0` | `3.11.2` | Core 2D plotting engine for academic figure and curve generation. |
| **seaborn** | `>= 0.13.0` | `0.13.2` | High-level statistical visualization (heatmaps, violin plots, and distributions). |

---

## 4. Frontend & Browser Requirements

### Supported Web Browsers
* Google Chrome (Version 90+)
* Mozilla Firefox (Version 88+)
* Microsoft Edge (Chromium-based, Version 90+)
* Apple Safari (Version 14+)
* Brave, Opera, and other modern Chromium-based browsers.

### Frontend Technologies Used (Included via CDN & Local Assets)
* **HTML5**: Semantic web structure.
* **CSS3**: Healthcare theme styling, custom shadows, and transitions.
* **JavaScript (ES6+)**: Dynamic form interactions, validation, and demo profile loading.
* **Bootstrap 5.3.3**: Responsive grid system, buttons, and alert components.
* **Bootstrap Icons 1.11.3**: Clinical iconography (heart pulse, hospital, activity badges).

---

## 5. Verification Commands

To verify your system meets all requirements:

```powershell
# 1. Verify Python Version (must be >= 3.10)
python --version

# 2. Verify Pip Version
pip --version

# 3. Verify Installed Packages match requirements.txt
pip list
```
