# MAGM
The source code for our article "Multi-modal adaptive gated mechanism for visual question answering"

To run the code, you can refer to the relevant content on [MCAN](https://github.com/MILVLG/mcan-vqa) and [OpenVQA](https://github.com/MILVLG/openvqa) or their [MODEL ZOO](https://openvqa.readthedocs.io/en/latest/basic/model_zoo.html).

#### Offline Evaluation

Offline evaluation only support the VQA 2.0 *val* split. If you want to evaluate on the VQA 2.0 *test-dev* or *test-std* split, please see [Online Evaluation](#Online-Evaluation).

There are two ways to start:

(Recommend)

```bash
$ python3 run.py --RUN='val' --CKPT_V=str --CKPT_E=int
```

or use the absolute path instead:

```bash
$ python3 run.py --RUN='val' --CKPT_PATH=str
```

#### Online Evaluation

The evaluations of both the VQA 2.0 *test-dev* and *test-std* splits are run as follows:

```bash
$ python3 run.py --RUN='test' --CKPT_V=str --CKPT_E=int
```

Result files are stored in ```results/result_test/result_run_<'PATH+random number' or 'VERSION+EPOCH'>.json```

You can upload the obtained result json file to [Eval AI](https://eval.ai/web/challenges/challenge-page/830/overview) to evaluate the scores on *test-dev* and *test-std* splits.

# Citation
If this repository is helpful for your research, we'd really appreciate it if you could cite the following paper:
```
@article{xu2023multi,
  title={Multi-modal adaptive gated mechanism for visual question answering},
  author={Xu, Yangshuyi and Zhang, Lin and Shen, Xiang},
  journal={Plos one},
  volume={18},
  number={6},
  pages={e0287557},
  year={2023},
  publisher={Public Library of Science San Francisco, CA USA}
}
```
