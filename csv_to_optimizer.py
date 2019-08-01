import requests
from ruamel import yaml
import dateutil.parser as dp
import time

''' PLEASE, UPDATE THE VARIABLES BELOW! '''
inputcsvfilepath = 'grafana_data_export_long_running_test-short.csv' #input
outputcsvfilepath = 'result_for_grafana_data_export_long_running_test.csv' # ebbe tolja bele az output
separator=';' #updated automatically if file starts with sep= string
max_number_of_rows_to_process = 20000 # parameter ennyit dolgozhat fel
column_index_of_time = 0
column_index_of_nodecount = 10
config={'optimizer_endpoint':'http://193.224.59.115:5000',
        'wait_after_rest_call':1}
init_params = {
  "constants": {
    "min_vm_number": 1,
    "max_vm_number": 10,
    "max_delta_vm" : 2,
    "training_samples_required": 300,
    "max_number_of_scaling_activity": 100,
    "nn_stop_error_rate": 10.0,
    "input_metrics": list(), #will be inserted by the code
    "target_metrics": [ { "name": "avg latency (quantile 0.5)", 
                          "min_threshold" : 1000000, 
                          "max_threshold" : 4000000 } ]
    }
  }

''' DO NOT CHANGE PARAMETERS BELOW '''
columnnames=[]

def convert_isodate_to_seconds(ts):
  return dp.parse(ts).strftime('%s')

def generate_init():
  for index in range(len(columnnames)): 
    if columnnames[index] != init_params["constants"]["target_metrics"][0]["name"] and \
       index != column_index_of_time and \
       index != column_index_of_nodecount:
      oneinput = dict()
      oneinput["name"]=columnnames[index]
      init_params["constants"]["input_metrics"].append(oneinput)
  print("INIT structure: {}".format(init_params))
  return

def calling_rest_api_init():
  global init_params
  url = config.get('optimizer_endpoint')+'/optimizer/init'
  print('Calling optimizer REST API init() method: '+url)
  try:
    response = requests.post(url, data=yaml.dump(init_params))
  except Exception as e:
    print('Calling optimizer REST API init() method raised exception: '+str(e))
    return
  print('Response: '+str(response))
  return

def generate_sample(values=dict()):
  global columnnames
  sample = dict()
  sample['sample']=dict()
  sample['sample']['input_metrics']=[]
  sample['sample']['target_metrics']=[]

  for index in range(len(values)):
    if index == column_index_of_time:
      sample['sample']['timestamp'] = int(convert_isodate_to_seconds(values[index])) \
        if values[index] != "null" else None
      continue
    if index == column_index_of_nodecount:
      sample['sample']['vm_number'] = int(values[index]) \
        if values[index] != "null" else None
      continue
    values[index] = None if values[index]=="null" else float(values[index])
    onesample=dict()
    onesample['name']=columnnames[index]
    onesample['value']=values[index]
    if columnnames[index] == init_params["constants"]["target_metrics"][0]["name"]:
      sample['sample']['target_metrics'].append(onesample)
    else:
      sample['sample']['input_metrics'].append(onesample)
  for s in sample['sample']['input_metrics']:
    if s['value'] is None:
      return None
  for s in sample['sample']['target_metrics']:
    if s['value'] is None:
      return None
  if sample['sample']['vm_number'] is None:
    return None
  return sample

def calling_rest_api_sample(sample=dict()):
  global config
  url = config.get('optimizer_endpoint')+'/optimizer/sample'
  print('Calling optimizer REST API sample() method: '+url)
  try:
    response = requests.post(url, data=yaml.dump(sample))
  except Exception as e:
    print('(O) Calling optimizer REST API sample() method raised exception: '+str(e))
    return
  print('Response: '+str(response))
  return

def calling_rest_api_advice():
  global config
  url = config.get('optimizer_endpoint')+'/optimizer/advice'
  print('Calling optimizer REST API advice() method: '+url)
  try:
    response = requests.get(url)
  except Exception as e:
    print('(O) Calling optimizer REST API advice() method raised exception: '+str(e))
    return dict()
  print('Response: '+str(response))
  print('Response message: '+str(response.json()))
  return response.json()

def extract_separator(line): 
  global separator
  if len(line.split("sep=",1))==2:
    separator = line.split("sep=",1)[1].rstrip()
    print('Separator: "{0}"'.format(separator))
    return True
  else:
    print('Separator: "{0}"'.format(separator))
    return False

def extract_columnnames(line):
  global columnnames
  columnnames=line.rstrip().split(separator)
  columnnames[column_index_of_time]="timestamp"
  print('Columnnames: "{0}"'.format(columnnames))

def train_optimizer_with_csv():
  with open(inputcsvfilepath) as fp:  
    line = fp.readline()
    cnt = 1
   
    if extract_separator(line):
      line = fp.readline()
      cnt +=1

    extract_columnnames(line) 
    
    generate_init()
    calling_rest_api_init()

    line = fp.readline()
    cnt +=1
    while line and cnt < max_number_of_rows_to_process:
      print("-------------- {} --------------".format(cnt))
      print("Line {}".format(line.strip()))
      values=line.rstrip().split(separator)
      print("Values: {}".format(values))
      sample = generate_sample(values)
      print("Sample: {}".format(sample))
      if sample is not None:
        calling_rest_api_sample(sample)
      line = fp.readline()
      cnt += 1

#time.sleep(config.get('wait_after_rest_call'))

def test_optimizer_with_csv():
  with open(inputcsvfilepath,'r') as fp, open(outputcsvfilepath,'w') as outfile:
    outfile.write('Time;Latency;AktVM;AdvVM\n')
    line = fp.readline()
    cnt = 1
   
    if extract_separator(line):
      line = fp.readline()
      cnt +=1

    extract_columnnames(line) 

    line = fp.readline()
    cnt +=1
    while line and cnt < max_number_of_rows_to_process:
      print("-------------- {} --------------".format(cnt))
      print("Line {}".format(line.strip()))
      values=line.rstrip().split(separator)
      print("Values: {}".format(values))
      sample = generate_sample(values)
      print("Sample: {}".format(sample))
      if sample is not None:
        calling_rest_api_sample(sample)
        advice = calling_rest_api_advice()
        print("Advice: {}".format(advice))
        if advice['valid']:
          line_to_save = '{0};{1};{2};{3}'.format(sample['sample']['timestamp'],
                                                  sample['sample']['target_metrics'][0]['value'],
                                                  sample['sample']['vm_number'],
                                                  advice['vm_number'])
          outfile.write(line_to_save+'\n')
          print(line_to_save)
      line = fp.readline()
      cnt += 1


train_optimizer_with_csv()
# test_optimizer_with_csv()
