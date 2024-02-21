# ModelEditingForDebias

Our code and dataset will be released soon.

## Data construction
- Base
    ```
    python data_construction/prepare_datasets_steroset.py \
        --model /path/to/llama \
        --counterfact_path /path/to/counterfact \
        --steroset_path /path/to/steroset \
        --output_path data_construction/outputs/steroset_llama
    ```

- Pos tag

- Causal
  ```
    python data_construction/causal_trace/experiments/new_causal_trace.py \
        --model_name /path/to/llama2-7b \
        --fact_dir data_construction/outputs/steroset_llama \
        --name edit \
        --output_dir data_construction/outputs/causal_llama
  ```
  

## Edit
- Before editing, both the MEND and SERAC methods require pre-training. The training script is located in `scripts/pre-train`. 
- For single editing, you can refer to `scripts/single`, and for sequential editing, refer to `scripts/sequential`. 
- Please ensure to modify the corresponding hyperparameter files, root path, and result path accordingly.

## Harness Eval
We evaluate the edited models' general capabilities using tasks such as CrowsPairs, Winogrande, OpenBookQA, and TruthfulQA from the LM-Harness framework. 
- The corresponding evaluation scripts can be found in `scripts/eval-harness`. 
- For final result visualization, utilize `DebiasEdit/bias_scripts/aggregate_harness.py`, specifying the model directory, seed list, and algorithm list.

## Aggregate results
During the editing process, we saved the relevant results for each test data point, requiring aggregation. The specific script can be found at `scripts/process_results`. 
- For **single editing**, we utilize `agg_single.sh`, where the **result file** needs to be specified. 
- For **sequential editing** with seeds, we employ `agg_seq.sh`, which requires specifying the seed, the **results dir**, and the algorithm. 
- To obtain the result figures for the paper, we use `agg_all.sh`, where the address for saving the results, the list of algorithms, and the list of seeds need to be specified.

## Others
- For experiments concerning the generalization across different biases, refer to `scripts/bias_type`, and employ `DebiasEdit/bias_scripts/aggregate_bias_type.py` for aggregation. 
- When using postag or causal data, simply modify the data and result dirs within the [edit section](#edit) of the script.

## Citation
```bibtex
@article{meng2022locating,
  title={Locating and Editing Factual Associations in {GPT}},
  author={Kevin Meng and David Bau and Alex Andonian and Yonatan Belinkov},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}

@article{zhang2024comprehensive,
  title={A Comprehensive Study of Knowledge Editing for Large Language Models},
  author={Zhang, Ningyu and Yao, Yunzhi and Tian, Bozhong and Wang, Peng and Deng, Shumin and Wang, Mengru and Xi, Zekun and Mao, Shengyu and Zhang, Jintian and Ni, Yuansheng and others},
  journal={arXiv preprint arXiv:2401.01286},
  year={2024}
}

@article{wang2023easyedit,
  title={Easyedit: An easy-to-use knowledge editing framework for large language models},
  author={Wang, Peng and Zhang, Ningyu and Xie, Xin and Yao, Yunzhi and Tian, Bozhong and Wang, Mengru and Xi, Zekun and Cheng, Siyuan and Liu, Kangwei and Zheng, Guozhou and others},
  journal={arXiv preprint arXiv:2308.07269},
  year={2023}
}

@article{yao2023editing,
  title={Editing Large Language Models: Problems, Methods, and Opportunities},
  author={Yao, Yunzhi and Wang, Peng and Tian, Bozhong and Cheng, Siyuan and Li, Zhoubo and Deng, Shumin and Chen, Huajun and Zhang, Ningyu},
  journal={arXiv preprint arXiv:2305.13172},
  year={2023}
}

@article{cheng2023edit,
  title={Can We Edit Multimodal Large Language Models?}, 
  author={Cheng, Siyuan and Tian, Bozhong and Liu, Qingbin and Chen, Xi and Wang, Yongheng and Chen, Huajun and Zhang, Ningyu},
  journal={arXiv preprint arXiv:2310.08475},
  year={2023}
}

@article{mao2023editing,
  title={Editing personality for llms},
  author={Mao, Shengyu and Zhang, Ningyu and Wang, Xiaohan and Wang, Mengru and Yao, Yunzhi and Jiang, Yong and Xie, Pengjun and Huang, Fei and Chen, Huajun},
  journal={arXiv preprint arXiv:2310.02168},
  year={2023}
}

@misc{knowlm,
  author = {Ningyu Zhang and Jintian Zhang and Xiaohan Wang and Honghao Gui and Kangwei Liu and Yinuo Jiang and Xiang Chen and Shengyu Mao and Shuofei Qiao and Yuqi Zhu and Zhen Bi and Jing Chen and Xiaozhuan Liang and Yixin Ou and Runnan Fang and Zekun Xi and Xin Xu and Lei Li and Peng Wang and Mengru Wang and Yunzhi Yao and Bozhong Tian and Yin Fang and Guozhou Zheng and Huajun Chen},
  title = {KnowLM Technical Report},
  year = {2023},
 url = {http://knowlm.zjukg.cn/},
}
```