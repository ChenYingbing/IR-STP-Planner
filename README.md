# IR-STP: Enhancing Autonomous Driving with Interaction Reasoning in Spatio-Temporal Planning

<h3 align="center">
  <a href="https://arxiv.org/abs/2311.02850">arxiv</a> | <a href="https://github.com/ChenYingbing/IR-STP-Planner">web</a>
</h3>
**ABSTRACT**: Considerable research efforts have been devoted to the development of motion planning algorithms, which form a cornerstone of the autonomous driving system (ADS). Nonetheless, acquiring an interactive and secure trajectory for the ADS remains challenging due to the complex nature of interaction modeling in planning. Modern planning methods still employ a uniform treatment of prediction outcomes and solely rely on collision-avoidance strategies, leading to suboptimal planning performance. To address this limitation, this paper presents a novel prediction-based interactive planning framework for autonomous driving. Our method incorporates interaction reasoning into spatio-temporal (s-t) planning by defining interaction conditions and constraints. Specifically, it records and continually updates interaction relations for each planned state throughout the forward search. We assess the performance of our approach alongside state-of-the-art methods in the CommonRoad environment. Our experiments include a total of 232 scenarios, with variations in the accuracy of prediction outcomes, modality, and degrees of planner aggressiveness. The experimental findings demonstrate the effectiveness and robustness of our method. It leads to a reduction of collision times by approximately 17.6% in 3-modal scenarios, along with improvements of nearly 7.6% in distance completeness and 31.7% in the fail rate in single-modal scenarios. For the community's reference, our code is accessible at https://github.com/ChenYingbing/IR-STP-Planner.


## Content List:

1. [News](#news)
2. [Introduction](#intro)
3. [Tutorials](#tutorials)



## News <a name="news"></a>

- 15/01/2024 Accepted by IEEE T-ITS.
- 07/11/2023 Initialize paper link
- 15/08/2023 Submitted.
- 28/07/2023 Initialization.

The code and tutorial have been released, any questions or inquiries are welcomed.



## Introduction <a name="intro"></a>

The introduction is coming soon.

If you find this repository useful, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
@article{chen2023ir,
  title={IR-STP: Enhancing Autonomous Driving with Interaction Reasoning in Spatio-Temporal Planning},
  author={Chen, Yingbing and Cheng, Jie and Gan, Lu and Wang, Sheng and Liu, Hongji and Mei, Xiaodong and Liu, Ming},
  journal={arXiv preprint arXiv:2311.02850},
  year={2023}
}
```



# Tutorials <a name="tutorials"></a>

Code includes
1. Training/Evaluation codes of pgp prediction networks implemented in Commonroad Env.
2. The proposed IR-STP planning method.
3. Evaluation tools Commonroad Env., including solution caching as well as metric extraction.

Tutorials see documentation [./docs/tutorial.md](./docs/tutorial.md).


