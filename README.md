# danger_detection
If you want tha basic understanding about audio classification. I would recommand to check-out [this playlist](https://www.youtube.com/playlist?list=PLhA3b2k8R3t0SYW_MhWkWS5fWg-BlYqWn), then come back to this repository.

Then set-up the environment using [this repository](https://github.com/seth814/Audio-Classification)

To split all audio in same length run:
```
python clean.py
```
If you want to change any parameter, change default value under ```__main__```

To train the models run:
```
python train.py
or 
python train.py --model_type resnet50
```
Models to run. i.e. conv1d, resnet50, lstm, mobilenetv2, inceptionv3, xception
To get accuracy of your model run:
```
python predict.py
or
python predict.py --model_type resnet50
```

If you find this repository and paper helpful, we would appreciate using the following citations:

```
@inproceedings{ashikuzzaman2021danger,
  title={Danger Detection for Women and Child Using Audio Classification and Deep Learning},
  author={Ashikuzzaman, Md and Fime, Awal Ahmed and Aziz, Abdul and Tasnima, Tanvira},
  booktitle={2021 5th International Conference on Electrical Information and Communication Technology (EICT)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}
```
