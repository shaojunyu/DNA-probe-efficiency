<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/shaojunyu/DNA-probe-efficiency">
    <img src="images/DNA.png" alt="Logo" width="100" height="100">
  </a>

<h3 align="center">DNA Probe Targeting Efficiency</h3>

  <p style="text-align: center">
    A deep learning tool for predicting DNA probe on-target efficiency in targeting sequencing
    <br />
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#train">Train</a></li>
        <li><a href="#predict">Predict</a></li>
        <li><a href="#data format">Data format</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## About The Project

[//]: # ([![Product Name Screen Shot][product-screenshot]]&#40;https://example.com&#41;)
A deep learning tool to build models for predicting
On-target efficiency of DNA probes based on their sequence. We provide some sample data and pre-trained models for testing and evaluation. Use this tool with your dataset to train customized models or just predict new datasets with pre-trained models. It can also be easily modified and applied to other sequence regression problems.



<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->

## Getting Started
This tool is developed in Python, so you need to set up a proper Python environment to run the code.

### Prerequisites

The recommended way to run this code is to create a conda environment and install all the dependencies inside that environment.


```sh
$ conda create -n probe python=3.7
$ conda activate probe
```

Install the dependencies:
```sh
(probe) $ conda install pandas scipy numpy tqdm scikit-learn -c conda-forge
```
> if you want   


Install Pytorch
```sh
(probe) $ conda install pytorch -c pytorch
```

### Installation

* Download the latest release of this tool from the [release page](https://github.com/shaojunyu/DNA-probe-efficiency/releases), unzip it then you can use the tool.
```sh
(probe) $ python DNA_Probe.py -h
```


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->

## Usage

Here are some simple and useful examples of how to use this tools. For more options, please refer to the supported arguments. The main program contains two subcommands: `train` and `predict`. In the Train mode, you can train new models with the new dataset, and in the Predict mode, you can predict the efficiency of pre-trained models.

### - Train
  1. Train a new model with the input data and save the model to output. The model will only use the sequence features.
      ```sh
      python3 DNA_Probe.py train \  
      -input data/pig_probe_effiency_150bp_train.tsv.gz \  
      -output models/pig_150bp_model.h5
      ```

  2. Train a new model that include the strtture information and set the learning rate as 2e-5. The model will use sequence features as well as the corresponding strucuture information of the sequence.
      ```
      python DNA_Probe.py train \  
      -input data/human_probe_effiency_120bp_with_struc_train.tsv.gz \ 
      -use_struct \  
      -output models/human_120bp_struct.h5 \  
      -lr 2e-5
      ```

  3. Use GPU to accelerate the training process.
      ```
      python DNA_Probe.py train \  
      -input data/human_probe_effiency_120bp_with_struc_train.tsv.gz \ 
      -use_struct \  
      -output models/human_120bp_struct.h5 \  
      -lr 2e-5 \  
      -gpu 0
      ```

  ***Supportted arguments:***
  * ***input : str, required***  
    The file path of the input data.
  * ***output : str, required***   
    The file path of the output model.
  * ***gpu : int, optional***   
    The GPU device ID that used to accelerate the process. Leave it empty to use CPU if GPU is not avaliable. Default: None.
  * ***kmer : int, optional***    
    The kmer leagth of DNA seq. The defult value is 1, which is the onehot encoding of DNA. Any value larger than 1 will encode the DNA sequnce based on the kmer first. Please note that if you set the `use_struct` option Ture, this option will have not effect.
  * ***onehot : bool, optional***  
    If [default: True], use onehot encodding for DNA seq and structure seq. If None, will leave only if position is 0. Please note that this argument will overide the setting in `kmer`.
  * ***use_struct : : bool, optional***  
    If true, encorapte the structure information in the model and only onehot encoding is available. Default: False.
  * ***embed_dim : int, optional***  
    Set the embeding dimention [default: 32] for input sequences.
  * ***epochs : int, optional***  
    Set the epoches [default: 60] for model training. De
  * ***batch_size: int, optional***  
    Set the batch size [default: 64] for model training.
  * ***lr : float, optional***  
    Set the learning rate [defalt: 1e-4] for model traning.   

  
</br>  

### - Predict
  1. Predict efficiency on new data and save the result to a file.
      ```
      python DNA_Probe.py predict \  
      -input data/human_probe_effiency_120bp_with_struc_test.tsv.gz \  
      -model models/human_120bp_struct.h5_bk \  
      -output prediction.txt
      ```
  2. Use GPU to accelarate the prediction.
      ```
      python DNA_Probe.py predict \  
      -input data/human_probe_effiency_120bp_with_struc_test.tsv.gz \  
      -model models/human_120bp_struct.h5_bk \  
      -output prediction.txt
      -gpu 0
      ```

### - Data Format

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->

## Roadmap

- [x] Predict on-target efficiency based on probe seq
- [ ] Figure out the seq features that lead to high efficiency
- [ ] Design highly efficient probes 
    - [ ] Sequence modification (adaptors, primers)
    - [ ] Verification by experiments



<!-- LICENSE -->

## License

Distributed under the BSD License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->

## Contact

Shaojun Yu - shaojun.yu@emory.edu  
Zhuqing Zheng - zzq1207@126.com   
Project Link: [https://github.com/shaojunyu/DNA-probe-efficiency](https://github.com/shaojunyu/DNA-probe-efficiency)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->

## References

* [https://www.idtdna.com/pages/technology/next-generation-sequencing/dna-sequencing/targeted-sequencing](https://www.idtdna.com/pages/technology/next-generation-sequencing/dna-sequencing/targeted-sequencing)
* [https://github.com/genetic-medicine/PaddleHelix_RNA_UPP](https://github.com/genetic-medicine/PaddleHelix_RNA_UPP)
* Ma, X. et al. (2019) ‘Analysis of error profiles in deep next-generation sequencing data’, Genome Biology, 20(1), p. 50. doi:10.1186/s13059-019-1659-6.
* Kim, H.K. et al. (2018) ‘Deep learning improves prediction of CRISPR–Cpf1 guide RNA activity’, Nature Biotechnology, 36(3), pp. 239–241. doi:10.1038/nbt.4061.
* Huang, L. et al. (2019) ‘LinearFold: linear-time approximate RNA folding by 5’-to-3’ dynamic programming and beam search’, Bioinformatics, 35(14), pp. i295–i304. doi:10.1093/bioinformatics/btz375.
* Sato, K., Akiyama, M. and Sakakibara, Y. (2021) ‘RNA secondary structure prediction using deep learning with thermodynamic integration’, Nature Communications, 12(1), p. 941. doi:10.1038/s41467-021-21194-4.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge

[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge

[forks-url]: https://github.com/github_username/repo_name/network/members

[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge

[stars-url]: https://github.com/github_username/repo_name/stargazers

[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge

[issues-url]: https://github.com/github_username/repo_name/issues

[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge

[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

[linkedin-url]: https://linkedin.com/in/linkedin_username

[product-screenshot]: images/screenshot.png