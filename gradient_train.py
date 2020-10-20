import logging
logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
import os
from aitextgen import aitextgen

steps = int(os.getenv('steps', 500))
ai = aitextgen(tf_gpt2="124M", to_gpu=True)

file_name = os.getenv('filename', "shakespeare.txt")

ai.train(file_name,
         output_dir='/artifacts',
         line_by_line=False,
         from_cache=False,
         num_steps=steps,
         generate_every=100,
         save_every=100,
         save_gdrive=False,
         learning_rate=1e-4,
         batch_size=1,      
         )
         
ai = aitextgen(model="trained_model/pytorch_model.bin", config="trained_model/config.json", to_gpu=True)

ai.generate(n=5,
            batch_size=5,
            prompt="ROMEO:",
            max_length=256,
            temperature=1.0,
            top_p=0.9)