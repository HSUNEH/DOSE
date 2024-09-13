# DOSE : Drum One-Shot Extraction
Drum one-shot samples are crucial for music production, particularly in sound design and electronic music. This paper introduces Drum One-Shot Extraction from music mixtures, a task designed to extract drum one-shots directly from reference mixtures. To facilitate this, we propose the Random Mixture One-shot Dataset (RMOD), comprising large-scale, randomly arranged music mixtures paired with corresponding drum one-shot samples. Our proposed model, Drum One- Shot Extractor (DOSE), leverages neural audio codec language models for end-to-end extraction, bypassing traditional source separation steps. Additionally, we introduce a novel onset loss function, which emphasizes accurate prediction of the initial transient of drum one-shots, crucial for capturing timbral characteristics. We compare this approach against a source separation-based extraction method as a baseline. The results, evaluated using Fre ́chet Audio Distance (FAD) and Mel-Spectrogram Similarity (MSS), demonstrate that DOSE, enhanced with onset loss, outperforms the baseline, providing more accurate and higher-quality drum one-shots from music mixtures. 
## How to Run

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/HSUNEH/DrumSlayer.git
   cd drumslayer
2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
3. **Evaluation**
    ```bash
    python evaluate.py --model_path /path/to/trained/model --test_data /path/to/test/data

You can Check Demo Here
https://hsuneh.notion.site/DrumSlayer-110005fe1b9443f58668c999c81c5745?pvs=4