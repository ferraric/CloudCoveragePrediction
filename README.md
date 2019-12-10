# DSLAB2019_MeteoSwissCloud

Meteoswiss Cloud Coverage

https://docs.google.com/document/d/1Ur9mitm7LuSoFrJ0W7OYMioCn-sK-0JS7Dntxzdgxoo/edit


```python
CUDA_VISIBLE_DEVICES=2 python3 main_mean_var_from_mv.py  -c ../configs/mean_var_from_mean_var.json
```
```python
CUDA_VISIBLE_DEVICES=2 python3 main_mean_var_from_7.py -c ../configs/mean_var_7_quantiles.json
```
```python
CUDA_VISIBLE_DEVICES=2 python3 main_7_to_21_after_15000.py  -c ../configs/21_from_7_quantiles.json
```
```python
CUDA_VISIBLE_DEVICES="" python3 main_7_to_21_after_200.py  -c ../configs/21_from_7_quantiles.json
```
```python
CUDA_VISIBLE_DEVICES=1 python3 main_7_to_21_after_2000.py  -c ../configs/21_from_7_quantiles.json
```
```python
CUDA_VISIBLE_DEVICES=4 python3 main_7_to_21_after_5000.py  -c ../configs/21_from_7_quantiles.json
```