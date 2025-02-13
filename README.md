# Voice Classification Project

This project demonstrates a **complete pipeline** for classifying audio data by **speaker** (Jeevan vs. not Jeevan) and **language** (English vs. not English). It includes **audio preprocessing** (removing silence, splitting audio, etc.) and a **machine learning pipeline** that uses MFCC features and Support Vector Machines (SVM) for classification.

---

## Project Structure

```
VOICE-CLASSIFICATION/
├── data/
│   ├── audio_files/         # Processed audio files (or final audio segments)
│   ├── raw_audio/           # Original, unprocessed audio files
│   └── metadata.csv         # CSV listing each file & its labels (speaker_label, language_label)
├── audio_classification.ipynb   # Jupyter notebook for classification tasks
├── audio_preprocessing.py       # Python script for audio data cleaning & splitting
├── LICENSE                      # Project license
└── README.md                    # This file
```

### Key Files

1. **`audio_preprocessing.py`**  
   - Removes silence at the beginning/end and optionally from the middle of audio if it exceeds a threshold.  
   - Normalizes audio signals.  
   - Splits long files into fixed-length segments (default: 7 seconds).  
   - Saves processed segments in `data/audio_files`.

2. **`audio_classification.ipynb`**  
   - Loads **metadata.csv** which labels each audio file with `speaker_label` (`Jeevan` or `Not_Jeevan`) and `language_label` (`English` or `Not_English`).  
   - Extracts MFCC features using **librosa**.  
   - Builds two separate **binary classification** models (one for speaker, one for language) using **SVM**.  
   - Evaluates performance with **k-Fold Cross-Validation** and reports **accuracy, precision, recall, F1-score**, and confusion matrices.

3. **`metadata.csv`**  
   - A CSV mapping each `filename` to `speaker_label` and `language_label`.  
   - Example:
     ```csv
     filename,speaker_label,language_label
     file_001.wav,Jeevan,English
     file_002.wav,Jeevan,Not_English
     file_003.wav,Not_Jeevan,English
     ...
     ```

---

## Getting Started

1. **Install Dependencies**  
   - Python 3.x  
   - [librosa](https://pypi.org/project/librosa/)  
   - [SoundFile](https://pypi.org/project/soundfile/)  
   - [scikit-learn](https://pypi.org/project/scikit-learn/)  
   - [NumPy](https://pypi.org/project/numpy/)  
   - [pandas](https://pypi.org/project/pandas/)

   ```bash
   pip install librosa soundfile scikit-learn numpy pandas
   ```

2. **Prepare Your Audio Data**  
   - Place your **raw** `.wav` files in `data/raw_audio`.  
   - Update the CSV file (`metadata.csv`) with the correct filenames and labels.

3. **Preprocess Audio**  
   - Run `audio_preprocessing.py` to remove silence and split audio into 7-second segments.  
   - The resulting processed segments will be saved in `data/audio_files`.

   ```bash
   python audio_preprocessing.py
   ```

4. **Run Classification**  
   - Open **`audio_classification.ipynb`** in Jupyter Notebook or JupyterLab.  
   - Adjust paths if necessary (`metadata_csv_path` and `audio_directory`).  
   - **Run all cells** to extract MFCC features, train & evaluate the SVM models for **speaker** and **language** classification.  
   - Observe the **classification reports** and **confusion matrices**.

---

## How It Works

1. **Audio Preprocessing**  
   - **Silence Removal**: Uses `librosa.effects.trim()` to remove leading/trailing silence.  
   - **Long Silence Removal**: Optionally merges sections if a silence gap is longer than 2 seconds in the middle.  
   - **Normalization**: Sets audio to zero mean and unit variance.  
   - **Splitting**: Splits each file into 7-second segments.

2. **Feature Extraction**  
   - Uses **MFCC** features from `librosa.feature.mfcc()`.  
   - Optionally, you can add other features like delta MFCCs, spectral centroids, etc.

3. **Classification**  
   - **Binary Classification** for speaker (Jeevan vs. Not_Jeevan) and language (English vs. Not_English).  
   - Uses **SVM** (Support Vector Machine) with linear kernel.  
   - **Stratified k-Fold** cross-validation to preserve class distribution in each fold.

4. **Evaluation**  
   - Prints **precision, recall, F1-score**, and **confusion matrix** for each task.  
   - Helps you see how well the model is performing on each class.

---

## Future Improvements

- **Hyperparameter Tuning**: Use `GridSearchCV` or `RandomizedSearchCV` for optimal SVM parameters.  
- **Neural Networks**: Experiment with CNNs or RNNs if you have sufficient data.  
- **Data Augmentation**: Add pitch shifting, time stretching, or noise injection to increase robustness.  
- **Advanced Audio Features**: Incorporate delta MFCCs, spectral roll-off, etc.

---

## License

This project is provided under the [MIT License](LICENSE). Feel free to modify and distribute as needed.

## Contact

For any questions or suggestions, please [open an issue](https://github.com/lifee77) or contact **Jeevan Bhatta** at [jeevanbhattacs@gmail.com](mailto:jeevanbhattacs@gmail.com).

---

**Happy Audio Classifying!**  
Feel free to contribute or suggest new features.