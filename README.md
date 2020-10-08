# Automated Distributed Training For Data Scientists

*This repo is for [Moon](https://www.linkedin.com/in/moonyuema/)'s project in Insight Fellow Program*

The GPT model uses two code bases: Torch-based code from [Andrej Karpathy](https://github.com/karpathy/minGPT) and Tensorflow-based code from [nshepperd](https://github.com/nshepperd/gpt-2).

The multinode training is implemented using built-in method `DistributedDataParalle()` for Torch and `Horovod` for Tensoflow.

You can use this repo in several ways:
1. Fine tune GPT2 on [paperspace](https://www.paperspace.com/home?utm_expid=.XZhCPCNrQCuE1jH9t8bIgg.1&utm_referrer=https%3A%2F%2Fwww.google.com%2F) platfrom with distribtued training.
2. Learn about how to debug when setting up [Kubeflow](https://www.kubeflow.org/) on GCP.
3. Learn about some caveats when running [Kubernetes](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus) with GPU nodes on GCP.
4. Learn about the concepts and tools in distributed training by reading a curated list.
5. Watch the [demo](https://drive.google.com/file/d/1PIz6eruE7x0P6MbpC0sz8bMfs1QpXoxW/view?usp=sharing) of the final deliverable for this project.


