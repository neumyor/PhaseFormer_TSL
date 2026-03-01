# PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting

Official implementation of **PhaseFormer**, a novel time series forecasting model that shifts the paradigm from traditional temporal patches to the **phase domain**.

This repository has been refactored for full compatibility with the [Time-Series Library](https://github.com/thuml/Time-Series-Library) framework, ensuring seamless integration for researchers and practitioners. For the original implementation, please visit [neumyor/PhaseFormer](https://github.com/neumyor/PhaseFormer).

---

## 🚀 Key Features

Unlike standard segment-based forecasting, PhaseFormer leverages the periodic nature of time series data through:

* **Phase Tokenization**: Segments time series into fixed cycles based on `period_len`, encoding each cycle as a discrete phase token.
* **Cross-Phase Interaction**: Utilizes a specialized routing mechanism to facilitate information exchange between different phases.
* **Phase Block Stacking**: Employs deep layers to predict future phase states, which are then reconstructed back into the temporal domain.

**The result?** Significant improvements in both computational efficiency and forecasting accuracy.

---

## 🛠️ Usage

### Environment Setup

Ensure you have the dependencies of the Time-Series Library installed. Then, you can run the provided scripts directly.

### Running Examples

Example script are provided in the /scripts directory. You can run PhaseFormer on Traffic by:

```bash
bash ./scripts/run_traffic.py
```

More scripts can be found in [neumyor/PhaseFormer](https://github.com/neumyor/PhaseFormer).

---

## 📚 Cite This Work

If you find PhaseFormer useful in your research, please cite our ICLR 2026 paper:

```bibtex
@inproceedings{niu2026phaseformer,
  title={PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting},
  author={Niu, Yunhao and Deng, Jie and Tong, Yunhai},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}

```


