from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModelWithLMHead,
    GPT2Config,
)
from transformers.convert_gpt2_original_tf_checkpoint_to_pytorch import (
    convert_gpt2_checkpoint_to_pytorch,
)
from transformers.modeling_utils import Conv1D
from torch.nn import Linear, Embedding
import torch
import os
import logging
import sys
from tqdm.auto import trange
from datetime import datetime
from random import randint
from .TokenDataset import TokenDataset
import pytorch_lightning as pl
from .utils import (
    download_gpt2,
    encode_text,
    set_seed,
    reset_seed,
)
from .train import ATGTransformer, ATGProgressBar
from .colab import create_gdrive_folder
from typing import Union, Optional, List
from pkg_resources import resource_filename
import shutil

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    pass

logger = logging.getLogger("aitextgen")
logger.setLevel(logging.INFO)

STATIC_PATH = resource_filename(__name__, "static")


class aitextgen:
    """
    Class that serves as the main aitextgen object for training and generation.

    :param model: Either the file path of a PyTorch GPT-2 model, or a string
    representing the Huggingface model to download.
    :param config: Either a file path of a config.json representing the model,
    or a GPT2Config with the model architecture.
    :param vocab_file: Path to a vocab file (generated by train_tokenizer())
    :param merges_file: Path to a merges file (generated by train_tokenizer())
    :param cache_dir: folder path which downloaded models will be stored and loaded
    :param tf_gpt2: model indicator of OpenAI-distributed version of GPT-2.
    This will convert the model to PyTorch if not present.
    :param to_gpu: Whether to load the model into the GPU after loading
    (good for generation)
    :param to_fp16: Whether to convert the model to FP16 before loading
    to GPU (for supported GPUs only)
    :param verbose: Whether to enable logging from base Huggingface packages
    :param torchscript: Whether the input model is a TorchScript traced model
    :param ts_to_trace: Whether to prep the input model to be exported to TorchScript
    :param bos_token: String to override the beginning-of-string token
    :param eos_token: String to override the end-of-string token
    :param unk_token: String to override the unknown token
    """

    torchscript = False

    # default values for GPT2Tokenizer
    tokenizer = None
    vocab_file = os.path.join(STATIC_PATH, "gpt2_vocab.json")
    merges_file = os.path.join(STATIC_PATH, "gpt2_merges.txt")
    bos_token = "<|endoftext|>"
    eos_token = "<|endoftext|>"
    unk_token = "<|endoftext|>"
    pad_token = "<|endoftext|>"

    def __init__(
        self,
        model: str = None,
        config: Union[str, GPT2Config] = None,
        vocab_file: str = None,
        merges_file: str = None,
        cache_dir: str = "aitextgen",
        tf_gpt2: str = None,
        to_gpu: bool = False,
        to_fp16: bool = False,
        verbose: bool = False,
        torchscript: bool = False,
        ts_to_trace: bool = False,
        bos_token: str = None,
        eos_token: str = None,
        unk_token: str = None,
        **kwargs,
    ) -> None:

        if not verbose:
            for module in [
                "transformers.file_utils",
                "transformers.configuration_utils",
                "transformers.tokenization_utils",
                "filelock",
                "transformers.modeling_gpt2",
            ]:
                logging.getLogger(module).setLevel(logging.WARN)
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

        if torchscript:
            assert model
            logger.info(f"Loading traced GPT-2 model from provided {model}.")
            if config is None:
                config = GPT2Config()
            self.torchscript = True
            self.model = GPT2LMHeadModel(config)

            # Transpose the traced model attributes to a GPT2LMHeadModel class
            # so it can inherit its functions
            pt_model = torch.jit.load(model)
            self.model.transformer = pt_model.transformer
            self.model.lm_head = pt_model.lm_head

        elif tf_gpt2:
            # Download + convert the TF weights if a PyTorch model has not been created
            if not os.path.isfile(
                os.path.join(cache_dir, f"pytorch_model_{tf_gpt2}.bin")
            ):
                assert tf_gpt2 in [
                    "124M",
                    "355M",
                    "774M",
                    "1558M",
                ], "Invalid TensorFlow GPT-2 model size."

                logger.info(
                    f"Downloading the {tf_gpt2} GPT-2 TensorFlow weights/config "
                    + "from Google's servers"
                )

                download_gpt2(cache_dir, tf_gpt2)

                logger.info(
                    f"Converting the {tf_gpt2} GPT-2 TensorFlow weights to PyTorch."
                )

                config_path = os.path.join(cache_dir, tf_gpt2, "hparams.json")

                convert_gpt2_checkpoint_to_pytorch(
                    os.path.join(cache_dir, tf_gpt2), config_path, cache_dir,
                )

                os.rename(
                    os.path.join(cache_dir, "pytorch_model.bin"),
                    os.path.join(cache_dir, f"pytorch_model_{tf_gpt2}.bin"),
                )

                os.rename(
                    os.path.join(cache_dir, "config.json"),
                    os.path.join(cache_dir, f"config_{tf_gpt2}.json"),
                )

            logger.info(f"Loading {tf_gpt2} GPT-2 model from /{cache_dir}.")
            model = os.path.join(cache_dir, f"pytorch_model_{tf_gpt2}.bin")
            config = os.path.join(cache_dir, f"config_{tf_gpt2}.json")

            self.model = GPT2LMHeadModel.from_pretrained(model, config=config)

        elif model and os.path.exists(model):
            # A pytorch_model.bin (+ optional config/config.json) is provided
            logger.info(f"Loading GPT-2 model from provided {model}.")
            if config is None:
                config = GPT2Config()
            if ts_to_trace:
                config.torchscript = True
            self.model = GPT2LMHeadModel.from_pretrained(model, config=config)
        elif config:
            if ts_to_trace:
                config.torchscript = True
            # Manually construct a GPT-2 model from scratch
            logger.info("Constructing GPT-2 model from provided config.")
            self.model = AutoModelWithLMHead.from_config(config=config)
        else:
            # Download and cache model from Huggingface
            if os.path.isdir(cache_dir) and len(os.listdir(cache_dir)) > 0:
                logger.info(f"Loading {model or 'gpt2'} model from /{cache_dir}.")
            else:
                logger.info(f"Downloading {model or 'gpt2'} model to /{cache_dir}.")
            self.model = GPT2LMHeadModel.from_pretrained(
                model or "gpt2", cache_dir=cache_dir, torchscript=ts_to_trace
            )
            if model and "gpt2" not in model:
                logger.info(f"Using the tokenizer for {model}.")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    model, cache_dir=cache_dir,
                )

        if self.tokenizer is None:
            # Update tokenizer settings (if not set already)
            args = locals()
            custom_tokenizer = False
            for attr in [
                "vocab_file",
                "merges_file",
                "bos_token",
                "eos_token",
                "unk_token",
            ]:
                if args[attr] is not None:
                    custom_tokenizer = True
                    setattr(self, attr, args[attr])

            if custom_tokenizer:
                logger.info("Using a custom tokenizer.")
            else:
                logger.info("Using the default GPT-2 Tokenizer.")

            self.tokenizer = GPT2Tokenizer(
                vocab_file=self.vocab_file,
                merges_file=self.merges_file,
                bos_token=self.bos_token,
                eos_token=self.eos_token,
                unk_token=self.unk_token,
                pad_token=self.pad_token,
            )

        if to_gpu:
            if to_fp16:
                self.to_fp16()
            self.to_gpu()

    def generate(
        self,
        n: int = 1,
        prompt: str = None,
        max_length: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        return_as_list: bool = False,
        seed: int = None,
        **kwargs,
    ) -> Optional[str]:
        """
        Generates texts using the stored Transformers model.
        Currently generates text using the model's generate() function.

        :param n: Numbers of texts to generate.
        :param prompt: Text to force the generated text to start with
        :param max_length: Maximum length for the generated text
        :param temperature: Determines the "creativity" of the generated text.
        The value range is different for each type of Transformer.
        :param do_sample: Samples the text, which is what we want. If False,
        the generated text will be the optimal prediction at each time,
        and therefore deterministic.
        :param return_as_list: Boolean which determine if text should be returned
        as a list. If False, the generated texts will be print to console.
        :param seed: A numeric seed which sets all randomness, allowing the
        generate text to be reproducible if rerunning with same parameters
        and model.
        """

        if prompt:
            prompt_text = prompt
            prompt = encode_text(prompt, self.tokenizer, self.get_device())

        if seed:
            set_seed(seed)

        # prevent an error from using a length greater than the model
        max_length = min(self.model.config.n_positions, max_length)

        outputs = self.model.generate(
            input_ids=prompt,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,
            num_return_sequences=n,
            **kwargs,
        )

        # Reset seed if used
        if seed:
            reset_seed()

        if n > 1:
            gen_texts = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
        else:
            gen_texts = [self.tokenizer.decode(outputs[0], skip_special_tokens=True)]

        if not return_as_list:
            if prompt is not None:
                # Bold the prompt if printing to console
                gen_texts = [
                    text.replace(prompt_text, f"\033[1m{prompt_text}\033[0m", 1)
                    for text in gen_texts
                ]

            print(*gen_texts, sep="\n" + "=" * 10 + "\n")
        else:
            return gen_texts

    def generate_one(self, **kwargs) -> None:
        """
        Generates a single text, and returns it as a string. Useful for
        returning a generated text within an API.

        See generate() for more parameters.
        """

        return self.generate(n=1, return_as_list=True, **kwargs)[0]

    def generate_samples(
        self, n: int = 3, temperatures: List[float] = [0.7, 1.0, 1.2], **kwargs
    ) -> None:
        """
        Prints multiple samples to console at specified temperatures.
        """

        for temperature in temperatures:
            print("#" * 20 + f"\nTemperature: {temperature}\n" + "#" * 20)
            self.generate(n=n, temperature=temperature, return_as_list=False, **kwargs)

    def generate_to_file(
        self,
        n: int = 20,
        batch_size: int = 1,
        destination_path: str = None,
        sample_delim: str = "=" * 20 + "\n",
        seed: int = None,
        cleanup: bool = True,
        **kwargs,
    ) -> None:
        """
        Generates a bulk amount of texts to a file, into a format
        good for manually inspecting and curating the texts.

        :param n: Number of texts to generate
        :param batch_size: Number of texts to generate simultaneously, taking
        advantage of CPU/GPU parallelization.
        :param destination_path: File name of the file. If None, a timestampped
        file name is automatically used.
        :param sample_delim: The text used to delimit each generated text.
        :param seed: Seed used for the generation. The last part of a file name
        will be the seed used to reproduce a generation.
        :param cleanup: Whether to polish the text before returning

        See generate() for more parameters.
        """

        assert n % batch_size == 0, f"n must be divisible by batch_size ({batch_size})."

        self.model = self.model.eval()

        if destination_path is None:
            # Create a time-based file name to prevent overwriting.
            # Use a 8-digit number as the seed, which is the last
            # numeric part of the file name.
            if seed is None:
                seed = randint(10 ** 7, 10 ** 8 - 1)

            destination_path = f"ATG_{datetime.utcnow():%Y%m%d_%H%M%S}_{seed}.txt"

        if seed:
            set_seed(seed)

        logger.info(f"Generating {n:,} texts to {destination_path}")

        pbar = trange(n)
        f = open(destination_path, "w", encoding="utf-8")

        for _ in range(n // batch_size):
            gen_texts = self.generate(n=batch_size, return_as_list=True, **kwargs)

            # Remove empty texts and strip out extra newlines/extra spaces
            if cleanup:
                texts_to_clean = gen_texts
                gen_texts = []
                for text in texts_to_clean:
                    clean_text = text.strip().strip("\n")
                    if clean_text and len(clean_text) >= 2:
                        gen_texts.append(clean_text)

            for gen_text in gen_texts:
                f.write("{}\n{}".format(gen_text, sample_delim))
            pbar.update(batch_size)

        pbar.close()
        f.close()

        if seed:
            reset_seed()

    def train(
        self,
        train_data: Union[str, TokenDataset],
        output_dir: str = "trained_model",
        fp16: bool = False,
        fp16_opt_level: str = "O1",
        n_gpu: int = -1,
        n_tpu_cores: int = 0,
        max_grad_norm: float = 0.5,
        gradient_accumulation_steps: int = 1,
        seed: int = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        num_steps: int = 5000,
        save_every: int = 1000,
        generate_every: int = 1000,
        n_generate: int = 1,
        loggers: List = None,
        batch_size: int = 1,
        num_workers: int = None,
        benchmark: bool = True,
        avg_loss_smoothing: float = 0.01,
        save_gdrive: bool = False,
        run_id: str = f"ATG_{datetime.utcnow():%Y%m%d_%H%M%S}",
        progress_bar_refresh_rate: int = 10,
        **kwargs,
    ) -> None:
        """
        Trains/finetunes the model on the provided file/dataset using pytorch-lightning.

        :param train_data: Either a TokenDataset containing the samples to be trained, or
        a string containing the text to be trained (shortcut instead of dataset)
        :param output_dir: A string indicating where to store the resulting
        model file folder.
        :param fp16: Boolean whether to use fp16, assuming using a compatible GPU/TPU.
        :param fp16_opt_level: Option level for FP16/APEX training.
        :param n_gpu: Number of GPU to use (-1 implies all available GPUs)
        :param n_tpu_cores: Number of TPU cores to use (should be a multiple of 8)
        :param max_grad_norm: Maximum gradient normalization
        :param gradient_accumulation_steps: Number of gradient acc steps
        :param seed: Interger representing the training seed.
        :param learning_rate: Training learnign rate for the default AdamW optimizer.
        :param weight_decay: Weight decay for the default AdamW optimizer.
        :param warmup_steps: Warmrup steps for the default AdamW optimizer.
        :param num_steps: Number of samples through the dataset.
        :param save_every: Number of steps for each time to save the model to disk
        :param generate_every: Number of steps for each time to generate sample text
        :param n_generate: Number of texts to generate when generate_every occurs.
        :param loggers: pytorch-lightning logger(s) to log results.
        :param batch_size: Number of input samples per batch
        :param num_workers: Number of DataLoader workers
        :param benchmark: If using GPU, whether to use cudnn.benchmarkl
        :param avg_loss_smoothing: Smoothing factor for Avg loss in progress bar
        :param save_gdrive: If using Colab, whether to save the notebook
        to Google Drive at each save_every
        :param run_id: Run identifier; used for save_gdrive
        :param progress_bar_refresh_rate: How often to update
        the progress bar while training.
        """

        assert not self.torchscript, "You cannot train a traced TorchScript model."

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if save_gdrive:
            assert (
                "google.colab" in sys.modules
            ), "You must be in Colaboratory to copy to your Google Drive"
            create_gdrive_folder(run_id)

        self.model = self.model.train()
        is_gpu_used = torch.cuda.is_available() and n_gpu != 0

        if isinstance(train_data, str):
            train_data = TokenDataset(
                vocab_file=self.vocab_file,
                merges_file=self.merges_file,
                bos_token=self.bos_token,
                eos_token=self.eos_token,
                unk_token=self.unk_token,
                file_path=train_data,
                block_size=self.model.config.n_positions,
                **kwargs,
            )

        if num_workers is None:
            # Use all CPU cores as workers if not training on CPU
            # Can overload 2x w/o diminishing returns
            if is_gpu_used or n_tpu_cores > 0:
                num_workers = os.cpu_count() * 2
            # If training on the CPU, use half the CPUs
            else:
                num_workers = int(os.cpu_count() / 2)

        hparams = dict(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            adam_epsilon=adam_epsilon,
            warmup_steps=warmup_steps,
            batch_size=batch_size,
            num_steps=num_steps,
            pin_memory=True if is_gpu_used else False,
            num_workers=num_workers,
            save_every=save_every,
            generate_every=generate_every,
            tpu=n_tpu_cores > 0,
        )

        # Wrap the model in a pytorch-lightning module
        train_model = ATGTransformer(self.model, train_data, hparams, self.tokenizer)

        # Begin training
        if seed:
            set_seed(seed)

        if os.path.exists(output_dir) and "pytorch_model.bin" in os.listdir(output_dir):
            logger.warning(
                f"pytorch_model.bin already exists in /{output_dir} and will be overwritten!"
            )

        # if try to use a GPU but no CUDA, use CPU
        if not is_gpu_used:
            n_gpu = 0

        train_params = dict(
            accumulate_grad_batches=gradient_accumulation_steps,
            gpus=n_gpu,
            max_steps=num_steps,
            #show_progress_bar=True,
            gradient_clip_val=max_grad_norm if not fp16 else 0,
            checkpoint_callback=False,
            logger=loggers if loggers else False,
            weights_summary=None,
            callbacks=[
                ATGProgressBar(
                    save_every,
                    generate_every,
                    output_dir,
                    n_generate,
                    is_gpu_used,
                    avg_loss_smoothing,
                    run_id,
                    save_gdrive,
                    progress_bar_refresh_rate,
                )
            ],
        )

        if fp16:
            train_params["precision"] = 16 if fp16 else 32
            train_params["amp_level"] = fp16_opt_level

        if n_tpu_cores > 0:
            train_params["num_tpu_cores"] = n_tpu_cores
            train_params["gpus"] = 0
            n_gpu = 0

        # benchmark gives a boost for GPUs if input size is constant,
        # which will always be the case with aitextgen training
        if n_gpu != 0 and benchmark:
            train_params["benchmark"] = True

        if n_gpu > 1:
            train_params["distributed_backend"] = "ddp"

        trainer = pl.Trainer(**train_params)
        trainer.fit(train_model)

        logger.info(f"Saving trained model pytorch_model.bin to /{output_dir}")

        if n_tpu_cores > 0:
            xm.rendezvous("save_model")
        self.model.save_pretrained(output_dir)

        if save_gdrive:
            for pt_file in ["pytorch_model.bin", "config.json"]:
                shutil.copyfile(
                    os.path.join(output_dir, pt_file),
                    os.path.join("/content/drive/My Drive/", run_id, pt_file),
                )

        if seed:
            reset_seed()

    def cross_train(
        self,
        inputs: List[TokenDataset],
        learning_rate: Union[float, List[float]] = 1e-4,
        num_steps: Union[int, List[int]] = 4000,
        run_id: str = f"ATG_{datetime.utcnow():%Y%m%d_%H%M%S}",
        **kwargs,
    ) -> None:
        """Trains a model across multiple input datasets, with automatic
        decay after each run."""

        datasets = [
            TokenDataset(
                vocab_file=self.vocab_file,
                merges_file=self.merges_file,
                bos_token=self.bos_token,
                eos_token=self.eos_token,
                unk_token=self.unk_token,
                file_path=x,
                **kwargs,
            )
            if isinstance(x, str)
            else x
            for x in inputs
        ]

        if not isinstance(learning_rate, list):
            learning_rate = [learning_rate / (2 ** x) for x in range(len(datasets))]

        if not isinstance(num_steps, list):
            num_steps = [int(num_steps / (2 ** x)) for x in range(len(datasets))]

        assert len(datasets) == len(learning_rate) == len(num_steps), (
            "The provided learning_rates or num_steps"
            + " is not equal to the number of inputs."
        )

        for i, dataset in enumerate(datasets):
            logger.info(f"Now training on {dataset} for {num_steps[i]:,} steps.")
            self.train(
                dataset,
                learning_rate=learning_rate[i],
                num_steps=num_steps[i],
                run_id=run_id,
                **kwargs,
            )

    def save(self, target_folder: str = os.getcwd()):
        """Saves the model into the specified directory."""
        self.model.save_pretrained(target_folder)

    def save_for_upload(self, target_folder: str = "my-model"):
        """
        Saves the model + tokenizerinto the specified directory.

        This generates the 6 files needed to upload the model to
        Huggingface's S3 bucket.
        """
        self.model.save_pretrained(target_folder)
        self.tokenizer.save_pretrained(target_folder)

    def quantize(self):
        """
        Quantizes the model, which gives it a generation performance boost.
        Should only be used to generate on a supported CPU.

        Currently only the lm_head layer is quantized:
        https://github.com/pytorch/pytorch/issues/34074
        """
        assert self.get_device() == "cpu", "quantize() can only be used on a CPU."

        self.model = torch.quantization.quantize_dynamic(
            self.model, {Linear, Embedding, Conv1D}, inplace=True
        )

    def export(self, quantize: bool = True, batch_size: int = 1) -> None:
        """
        Exports the model to TorchScript, with optional quantization
        """

        temp_model = self.model

        if quantize:
            temp_model = torch.quantization.quantize_dynamic(
                temp_model, {Linear, Embedding, Conv1D}, inplace=True
            )

        example = torch.zeros(
            (batch_size, self.model.config.n_positions), dtype=torch.long
        )
        temp_model = torch.jit.trace(temp_model.eval(), example)
        temp_model.save("model.pt")

    def to_gpu(self, index: int = 0) -> None:
        """Moves the model to the specified GPU."""

        assert torch.cuda.is_available(), "CUDA is not installed."

        self.model.to(torch.device("cuda", index))

    def to_cpu(self, index: int = 0) -> None:
        """Moves the model to the specified CPU."""

        self.model.to(torch.device("cpu", index))

    def to_tpu(self) -> None:
        """Moves the model to the TPU."""

        self.model.to(xm.xla_device())

    def to_fp16(self) -> None:
        """
        Converts the model to a FP16 representation.
        Should only be used to generate on a supported GPU.
        """

        self.model = self.model.half()

    def get_device(self) -> str:
        """Getter for the current device where the model is located."""
        return self.model.device.type