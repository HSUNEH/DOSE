# DrumSlayer

DrumSlayer is a Transformer-based model designed to generate high-quality drum one-shots (kick, snare, hi-hat) from complex music inputs. It leverages a Decoder-only Transformer architecture combined with the Descript Audio Codec (DAC) for precise, low-artifact drum sound generation. The model is trained on the Randomly Synthesized Drum Dataset (RSDD) and can be used for various tasks such as drum analysis and creative remixing in music production.

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