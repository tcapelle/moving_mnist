# Moving MNIST forecasting
> A little experiment using Convolutional RNNs to forecast moving MNIST digits.


```python
from fastai2.vision.all import *
from moving_mnist.models.conv_rnn import *
from moving_mnist.data import *
```

```python
torch.cuda.set_device(0)
torch.cuda.get_device_name()
```




    'Quadro RTX 8000'



## Install

It only uses fastai2 as dependency. Check how to install at https://github.com/fastai/fastai2

## Example:

We wil predict:
- `n_in`: 5 images
- `n_out`: 5 images  
- `n_obj`: 3 objects

```python
ds = MovingMNIST(DATA_PATH, n_in=5, n_out=5, n_obj=3)
```

```python
train_tl = TfmdLists(range(500), ImageTupleTransform(ds))
valid_tl = TfmdLists(range(100), ImageTupleTransform(ds))
```

```python
dls = DataLoaders.from_dsets(train_tl, valid_tl, bs=8,
                             after_batch=[Normalize.from_stats(imagenet_stats[0][0], 
                                                               imagenet_stats[1][0])]).cuda()
```

Left: Input, Right: Target

```python
dls.show_batch()
```


![png](docs/images/output_10_0.png)


```python
model = SimpleModel()
```

```python
model
```




    SimpleModel(
      (encoder): Encoder(
        (convs): ModuleList(
          (0): TimeDistributed(
            (module): ConvLayer(
              (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): ReLU()
            )
          )
          (1): TimeDistributed(
            (module): ConvLayer(
              (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
              (1): ReLU()
            )
          )
          (2): TimeDistributed(
            (module): ConvLayer(
              (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
              (1): ReLU()
            )
          )
        )
        (rnns): ModuleList(
          (0): ConvGRU_cell(in=16, out=64, ks=5)
          (1): ConvGRU_cell(in=64, out=96, ks=5)
          (2): ConvGRU_cell(in=96, out=96, ks=5)
        )
      )
      (decoder): Decoder(
        (head): TimeDistributed(
          (module): Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1))
        )
        (deconvs): ModuleList(
          (0): TimeDistributed(
            (module): UpsampleBLock(in=96, out=96, blur=False, act=ReLU(), attn=False, norm=None)
          )
          (1): TimeDistributed(
            (module): UpsampleBLock(in=96, out=96, blur=False, act=ReLU(), attn=False, norm=None)
          )
          (2): TimeDistributed(
            (module): ConvLayer(
              (0): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): ReLU()
            )
          )
        )
        (rnns): ModuleList(
          (0): ConvGRU_cell(in=96, out=96, ks=5)
          (1): ConvGRU_cell(in=96, out=96, ks=5)
          (2): ConvGRU_cell(in=96, out=64, ks=5)
        )
      )
    )



```python
class TargetSeq(Callback):
    def after_pred(self):
        self.learn.yb = (torch.stack(self.yb[0], dim=1),)
```

```python
learn = Learner(dls, model, loss_func=MSELossFlat(), cbs=[TargetSeq()])
```

```python
x,y = dls.one_batch()
```

```python
learn.lr_find()
```








    SuggestedLRs(lr_min=0.010000000149011612, lr_steep=1.737800812406931e-05)




![png](docs/images/output_16_2.png)


```python
learn.fit_one_cycle(1, 1e-4)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.045332</td>
      <td>0.872255</td>
      <td>00:12</td>
    </tr>
  </tbody>
</table>

