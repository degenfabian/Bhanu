# Bhanu - Non-invasive Parkinson's Disease Detection Using Transformer-based Analysis of Photoplethysmography Signals

## Abstract
This independent research project investigates the potential of photoplethysmography (PPG) signals as non-invasive biomarkers for Parkinson's Disease (PD) detection. 
By leveraging the MIMIC-III waveform and clinical databases and adapting and finetuning the HeartGPT architecture (https://github.com/harryjdavies/HeartGPT), 
I demonstrate the feasibility of using transformer-based deep learning models for analyzing physiological time series data in neurological disease detection. 

## Research Objectives
1. Evaluate the efficacy of PPG signals as biomarkers for Parkinson's Disease
2. Develop and validate a transformer-based deep learning approach for medical time series classification by adapting and fine-tuning HeartGPT
3. Examine the effectiveness of selective fine-tuning by training the final five transformer blocks while keeping earlier layers frozen, testing whether HeartGPT's learned signal representations transfer to PD detection
4. Implement a systematic approach for processing MIMIC-III waveform data, including patient matching, data loading and signal preprocessing pipelines suitable for deep learning applications
## Methodology

### Data Collection and Preprocessing
The study utilizes the MIMIC-III (Medical Information Mart for Intensive Care III) database. MIMIC-III comprises deidentified health-related data associated with over forty thousand patients 
who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012.
Of particular interest to this study are the high-resolution physiological waveforms, specifically the PPG signals, recorded during patient stays. 
My preprocessing pipeline includes the following:

1. Patient Cohort Selection:
   - Identification of PD patients using ICD-9 codes
   - Demographic matching according to patient age and gender with control subjects

2. Signal Preprocessing:
   - Extraction of 4-second PPG segments from waveform data
   - Bandpass filtering (0.7 Hz - 10 Hz)
   - Removal of segments that contain missing or NaN values

3. Data Transformation:
   - Signal normalization (zero mean, unit variance)
   - Min-max scaling to [0,1] range
   - Tokenization into discrete values (0-100 -> 101 total tokens)
   - Train/validation/test split with stratification

```bibtex
@article{johnson2016mimic,
    title={MIMIC-III, a freely accessible critical care database},
    author={Johnson, Alistair EW and Pollard, Tom J and Shen, Lu and 
            Li-Wei, H Lehman and Feng, Mengling and Ghassemi, Mohammad and 
            Moody, Benjamin and Szolovits, Peter and Celi, Leo Anthony and 
            Mark, Roger G},
    journal={Scientific data},
    volume={3},
    number={1},
    pages={1--9},
    year={2016},
    publisher={Nature Publishing Group}
}

@article{moody2020mimic,
    title={MIMIC-III Waveform Database (version 1.0)},
    author={Moody, Benjamin and Moody, George and Villarroel, Mauricio and 
            Clifford, Gari D and Silva, Ikaro},
    journal={PhysioNet},
    year={2020},
    doi={10.13026/c2607m}
}
```

### Model Architecture
My approach builds upon the HeartGPT model, with some modifications for PD detection:

- Input Layer: Processes tokenized PPG sequences
- Transformer Backbone: 8 layers with 8 attention heads
- Custom Classification Head: For classifying PD from PPG signals
- Embedding Dimension: 64
- Sequence Length: 500 tokens (4-second PPG window sampled at 125 Hz)

The model employs a fine-tuning strategy where:
- Initial layers remain frozen, preserving learned physiological features
- Final five transformer blocks are fine-tuned
- New classification head is trained from scratch

```bibtex
@misc{davies2024interpretablepretrainedtransformersheart,
      title={Interpretable Pre-Trained Transformers for Heart Time-Series Data}, 
      author={Harry J. Davies and James Monsen and Danilo P. Mandic},
      year={2024},
      eprint={2407.20775},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.20775}, 
}
```

## Hardware recommendations
- RAM: 64 GB
- Storage: 128 GB SSD
- GPU: NVIDIA A100 or similar
- High-speed internet connection

## Reproduction of results

### Prerequisites
- Valid PhysioNet credentials
- MIMIC-III data use agreement
- Completed CITI training
- Python 3.10+
- For dependencies see requirements.txt

### Installation
1. Clone the repository:
```
git clone https://github.com/degenfabian/Bhanu.git
cd Bhanu
```

2. Install dependencies:
```
pip install -r requirements.txt

```

3. (Optional) Configure hyperparameters in Config class in train_and_eval.py
4. Download data:
```
python download_data.py

```
5. Preprocess data (takes ~1-2 hours):
```
python preprocess_data.py

```
6. Create model_weights directory inside root project folder
```
mkdir model_weights

```

7. Download model weights from https://github.com/harryjdavies/HeartGPT (PPGPT_500k_iters.pth) and put them in model_weights directory
8. Train and evaluate model:
```
python train_and_eval.py

```



### Project Structure
```
Bhanu/
├── data/
│   ├── waveform_data/
│   │   ├── PD/
│   │   └── non_PD/
│   ├── train_dataset.pt
│   ├── val_dataset.pt
│   └── test_dataset.pt
├── preprocess_data.py
├── model.py
├── train_and_eval.py
├── metrics.py
└── model_weights/
    └── PPGPT_500k_iters.pth
```

## Training and Evaluation
This model was trained in Google Colab using an A100 GPU.
The split for the dataset was 70% for training, 10% for validation and the remaining 20% for testing.
Around 5147.8 hours of PPG data is from patients with Parkinson's disease, amounting to 18.57 GB of data.
For patients without Parkinson's disease there are around 4583.1 hours of PPG data, amounting to 16.53 GB of data.
The total dataset size is therefore 35.1 GB.

## Results and Discussion
*Note: The model is currently undergoing training. This section will be updated with final results.*

### Performance Metrics
The model will be evaluated using the following metrics:
- Accuracy
- Sensitivity
- Specificity
- F1 Score

## Limitations and Biases

### Dataset-Specific Biases
1. Selection Bias
   - MIMIC-III data comes exclusively from ICU/hospital settings, meaning all subjects (both PD and control) were ill enough to require hospitalization
   - PD patients in the dataset may represent more severe or complicated cases than the general PD population
   - Control subjects are not healthy individuals but other hospitalized patients, potentially confounding the analysis

2. Demographic Biases
   - MIMIC-III data comes from a single medical center (Beth Israel Deaconess Medical Center)
   - Geographic limitation to one region may not represent global population variations
   - Potential socioeconomic biases based on hospital location and accessibility

### Methodological Biases

1. Signal Processing Biases
   - 4-second PPG segment selection may miss longer-term patterns
   - Bandpass filtering could eliminate potentially relevant signal components
   - Tokenization process may introduce quantization artifacts

2. Model Architecture Biases
   - Transfer learning from HeartGPT may carry over biases from cardiac domain
   - Frozen initial layers may retain inappropriate feature representations
  
### Future work
- External validation on independent datasets
- Prospective validation studies in clinical settings
- Comparison with traditional PD diagnostic methods
- Assessment of model performance across different PD stages

## Contact
Maintainer: [Fabian Degen] - [fabidegen@gmail.com]

For bugs and feature requests, please open an issue in this GitHub repository.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Research Disclaimer**: This work is intended for research purposes only. The methods and findings presented here should not be used for clinical diagnosis without proper validation and regulatory approval.

**Acknowledgments**: I thank the PhysioNet team for providing access to the MIMIC-III database and the original HeartGPT authors for their spectacular work.
