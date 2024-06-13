# PILOT
SenSys 2023: [Physics-Informed Data Denoising for Real-Life Sensing Systems](https://dl.acm.org/doi/10.1145/3625687.3625811)

### Data

We prepare the training and test data in numpy format under the folder `./data`. Due to size limit, we upload the pre-processed data on [Google Drive](https://drive.google.com/drive/folders/1Sj2OomEUwVmAcp2ugtR501ApFIPPbNsA?usp=sharing).

For inertial navigation, the training data are high-quality acceleration `high_a.npy`, angular velocity `high_w.npy`, and position/orientation `high_vi.npy`; the test data are low-quality acceleration `low_a.npy`, angular velocity `low_w.npy`, and position/orientation `low_vi.npy`.

### Training and Evaluation

Run the script `main.py` for pre-training (phase I), training (phase 2), and evaluation.

* args.z: autoencoder latent dimensions

* args.pre_epochs: number of pre-training epochs

* args.epochs: number of training epochs

* args.rate: manually added noise ratio

```sh
python main.py
```

### Citation
If you find this useful, please cite our paper: "Physics-Informed Data Denoising for Real-Life Sensing Systems"
```
@inproceedings{zhang2023physics,
  title={Physics-Informed Data Denoising for Real-Life Sensing Systems},
  author={Zhang, Xiyuan and Fu, Xiaohan and Teng, Diyan and Dong, Chengyu and Vijayakumar, Keerthivasan and Zhang, Jiayun and Chowdhury, Ranak Roy and Han, Junsheng and Hong, Dezhi and Kulkarni, Rashmi and others},
  booktitle={Proceedings of the 21st ACM Conference on Embedded Networked Sensor Systems},
  pages={83--96},
  year={2023}
}
```