## SAGE

Code repository of the paper [Identifying and Mitigating Machine Learning Biases for the Gravitational Wave Detection Problem](https://arxiv.org/abs/2501.13846).

**Abstract**

Matched-filtering is a long-standing technique for the optimal detection of known signals in stationary Gaussian noise. However, it has known departures from optimality when operating on unknown signals in real noise and suffers from computational inefficiencies in its pursuit to near-optimality. A compelling alternative that has emerged in recent years to address this problem is deep learning. Although it has shown significant promise when applied to the search for gravitational-waves in detector noise, we demonstrate the existence of a multitude of learning biases that hinder generalisation and detection performance. Our work identifies the sources of a set of 11 interconnected biases present in the supervised learning of the gravitational-wave detection problem, and contributes mitigation tactics and training strategies to concurrently address them. We introduce, Sage, a machine-learning based binary black hole search pipeline. We evaluate our pipeline on the injection study presented in the Machine Learning Gravitational-Wave Search Challenge and show that Sage detects ~11.2% more signals than the benchmark PyCBC analysis at a false alarm rate of one per month in O3a noise. Moreover, we also show that it can detect ~48.29% more signals than the previous best performing machine-learning pipeline on the same dataset. We empirically prove that our pipeline has the capability to effectively handle out-of-distribution noise power spectral densities and reject non-Gaussian transient noise artefacts. By studying machine-learning biases and conducting empirical investigations to understand the reasons for performance improvement/degradation, we aim to address the need for interpretability of machine-learning methods for gravitational-wave detection.

### TODO
1. Add documentation
2. Notes to reproduce results in the paper
3. Code cleanup

### Cite
If you found this work useful in your research, please consider citing:

```
@misc{nagarajan2025identifyingmitigatingmachinelearning,
      title={Identifying and Mitigating Machine Learning Biases for the Gravitational-wave Detection Problem}, 
      author={Narenraju Nagarajan and Christopher Messenger},
      year={2025},
      eprint={2501.13846},
      archivePrefix={arXiv},
      primaryClass={gr-qc},
      url={https://arxiv.org/abs/2501.13846}, 
}
```
Will be updated after publication.

### Acknowledgements

We appreciate the useful comments from Thomas Dent and Nikolaos Stergioulas on our paper. NN wishes to acknowledge and appreciate the support of Joseph Bayley, Michael Williams and Christian Chapman-Bird. We would also like to extend our sincere gratitude to the PHAS-ML group members from the University of Glasgow, for their fruitful weekly meetings. NN is supported by the College Scholarship offered by the School of Physics and Astronomy (2021-2025), University of Glasgow. CM is supported by STFC grant ST/Y004256/1. This material is based upon work supported by NSF’s LIGO Laboratory, a major facility fully funded by the National Science Foundation.
