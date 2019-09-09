# MiCADO - Scaling Optimizer with Machine Learning Support

## Don't forget to set environment
From project root run
```source activate optimizer```

## Test program
From project root run
```python pure.py```

## Test flask
From project root run
```python helloMTA.py```

## Start program 
From project root run  
```python optimizer.py --cfg path/to_config_file```

```python optimizer.py --cfg config/config.yaml```  

```python optimizer.py --cfg config/config.yaml --host=192.168.0.60```  

```python optimizer.py --cfg config/config.yaml --host 0.0.0.0 --port 5000```


## Test REST API 
__GET /optimizer/hello__  
Test REST API.  
```curl -X GET http://193.224.59.115:5000/optimizer/hello```  

```curl -X GET http://193.224.59.115:5000/```  
  
__POST /optimizer/init__  
Initialize optimizer with the neccessary constants.  
```curl -X POST http://127.0.0.1:5000/optimizer/init --data-binary @test_files/optimizer_constants.yaml```  

```curl -X POST http://193.224.59.115:5000/optimizer/init --data-binary @test_files/optimizer_constants.yaml```
  
__POST /optimizer/sample__   
Send a new training sample.  
```curl -X POST http://127.0.0.1:5000/optimizer/sample --data-binary @test_files/metrics_sample_example.yaml``` 

```curl -X POST http://193.224.59.115:5000/optimizer/sample --data-binary @test_files/metrics_sample_example.yaml```  

```curl -X POST http://193.224.59.115:5000/optimizer/sample --data-binary @test_files/metrics_sample_example_up.yaml```  

```curl -X POST http://193.224.59.115:5000/optimizer/sample --data-binary @test_files/metrics_sample_example_down.yaml``` 

__GET /optimizer/advice__     
Get scaling advice.  
```curl -X GET http://127.0.0.1:5000/optimizer/advice```  

```curl -X GET http://193.224.59.115:5000/optimizer/advice```  

```curl -X GET http://193.224.59.115:5000/optimizer/advice?last=False```  
  
__GET /optimizer/training_data__  
Download zipped training data that contains both neural network and linear regression data.  
```curl -X GET http://127.0.0.1:5000/optimizer/training_data```  

## TEST CSV  
cd csv/csv_to_optimizer  
source env/csv_to_optimizer/bin/activate  
```python csv_to_optimizer.py```  


