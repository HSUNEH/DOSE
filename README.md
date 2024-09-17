# DOSE : Drum One-Shot Extraction
![Figure 1](./figures/1_task.png)
Drum one-shot samples are crucial for music production, particularly in sound design and electronic music. This paper introduces Drum One-Shot Extraction from music mixtures, a task designed to extract drum one-shots directly from reference mixtures. To facilitate this, we propose the Random Mixture One-shot Dataset (RMOD), comprising large-scale, randomly arranged music mixtures paired with corresponding drum one-shot samples. Our proposed model, Drum One- Shot Extractor (DOSE), leverages neural audio codec language models for end-to-end extraction, bypassing traditional source separation steps. Additionally, we introduce a novel onset loss function, which emphasizes accurate prediction of the initial transient of drum one-shots, crucial for capturing timbral characteristics. We compare this approach against a source separation-based extraction method as a baseline. The results, evaluated using Fre ́chet Audio Distance (FAD) and Mel-Spectrogram Similarity (MSS), demonstrate that DOSE, enhanced with onset loss, outperforms the baseline, providing more accurate and higher-quality drum one-shots from music mixtures. 
![Figure 2](./figures/2_method.png)
The input audio mixture is encoded into a sequence of discrete tokens using a frozen DAC encoder, which are then fed into a decoder-only Transformer. The Transformer is trained to autoregressively predict the groundtruth drum one-shot tokens by minimizing two losses: onset loss and full-length loss. Finally, the predicted token sequence is decoded into drum one-shot audio using the DAC decoder.

<!-- ## How to Run -->
<!-- 
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/HSUNEH/DOSE.git
   cd DOSE
2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
3. **Evaluation**
    ```bash
    python evaluate.py --model_path /path/to/trained/model --test_data /path/to/test/data -->

## Check Demo 
https://hsuneh.notion.site/DrumSlayer-110005fe1b9443f58668c999c81c5745?pvs=4

## RMOD (Random Mixture One-shot Dataset)
![Figure 3](./figures/3_dataset.png)
Dataset generation process. First, kick, snare, and hi-hat loops are synthesized from one-shot drum audio samples using randomly generated MIDI notes. Next, optional bass, piano, guitar, and vocal loops are selected. The drum loops and other musical loops are then processed through independent mixing chains, which apply gain, EQ, compression, panning, limiting, delay, and reverb effects. Finally, all tracks are combined and passed through a mastering chain consisting of EQ and limiter effects.

You can download RMOD from Kaggle.(test, 10000 files)

https://www.kaggle.com/datasets/sunehflower/random-mixture-one-shot-dataset-rmod


(Full Dataset Coming Soon)
