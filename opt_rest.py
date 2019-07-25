from flask import Flask, jsonify, request, send_file
from ruamel import yaml

import logging
import logging.config

import opt_config
import opt_utils

import pandas as pd

# import advice
# from advice_oop import Advice
# from train import TrainingUnit


app = Flask(__name__)

logger = None
config = None
training_unit = None
#advice = None
training_result = []

constants = {}
sample_number = 0
vm_number_prev = 0
vm_number_prev_kept = 0
min_same_length = 3
sample_data_temp = []

def init_service(cfg):
    global logger
    logger = logging.getLogger('optimizer')

    global config
    config = opt_config.OptimizerConfig(cfg, 'optimizer')


@app.route('/', methods=['GET'])
def index():
    return 'MiCado Optimizer Modul'


@app.route('/optimizer/hello', methods=['GET'])
def hello():
    return 'Hello Optimizer'


@app.route('/optimizer/init', methods=['POST'])
def init():
    logger.info('Loading constants from file...')
    constants_yaml = request.stream.read()
    if not constants_yaml:
        raise RequestException(400, 'Empty POST data')
    else:
        global constants
        constants = yaml.safe_load(constants_yaml).get('constants')
        logger.info(f'Constants received: {constants}')
        
        # és kell írni egy metodust arra is, hogy törölje azt a szerencsétlet (reset)

        logger.info('Saving constants...')
        opt_utils.write_yaml(config.constants_filename, constants)
        logger.info('Constants saved to "data/constants.yaml"')
        # TODO:
        # ezekre a vm_number_prev változókra valószínüleg nem lesz szükség ugyhogy majd törölhetem
        global vm_number_prev
        logger.info(f'VM_NUMBER_PREV BEFORE INIT: {vm_number_prev}')
        vm_number_prev = constants.get('initial_vm_number', 1)
        global vm_number_prev_kept
        vm_number_prev_kept = vm_number_prev
        logger.info(f'Initial vm number: {vm_number_prev}')
        logger.info(f'Vm number prev used: {vm_number_prev_kept}')

        logger.info('Preparing database for training data...')

        input_metrics = [metric.get('name')
                         for metric in constants.get('input_metrics')]
        target_metrics = [metric.get('name')
                          for metric in constants.get('target_metrics')]

        timestamp_col = ['timestamp']
        vm_cols = ['vm_number', 'vm_number_prev', 'vm_number_diff']
        worker_count = ['vm_number']

        logger.info('Creating a .csv file for neural network...')
        opt_utils.persist_data(
            # TODO:
            # ide a vm_numbert is hozzá kell adnom
            config.nn_filename, timestamp_col+input_metrics+worker_count+target_metrics, 'w')
        logger.info('File created')
        
        logger.debug('Creating a .csv file for linear regression...')
        opt_utils.persist_data(
            config.lr_filename, timestamp_col+input_metrics+vm_cols, 'w')
        logger.debug('File created')

        logger.info('Optimizer REST initialized successfully ')
    return jsonify('OK'), 200


@app.route('/optimizer/training_data', methods=['GET', 'POST'])
def training_data():
    if request.method == 'GET':
        logger.info('Sending zipped training data...')
        return 'Hello training_data GET'


@app.route('/optimizer/sample', methods=['POST'])
def sample():
    logger.info('Loading training sample...')
    
    sample_yaml = request.stream.read()
    if not sample_yaml:
        raise RequestException(400, 'Empty POST data')
    else:
        sample = yaml.safe_load(sample_yaml)
        logger.info(f'New sample received: {sample}')
        logger.info('Getting sample data...')
        input_metrics = [metric.get('value')
                         for metric in sample.get('sample').get('input_metrics')]
        target_metrics = [metric.get('value')
                          for metric in sample.get('sample').get('target_metrics')]
        vm_number = sample.get('sample').get('vm_number')
        timestamp_col = [sample.get('sample').get('timestamp')]
        logger.info('Sample data stored in corresponding variables.')
        logger.info('----------------------------------------------')
        logger.info(f'input_metrics = {input_metrics}')
        logger.info(f'target_metrics = {target_metrics}')
        logger.info(f'vm_number = {vm_number}')
        logger.info(f'timestamp_col = {timestamp_col}')
        logger.info('----------------------------------------------')

        print(timestamp_col+input_metrics+target_metrics+[vm_number])
        
        if None not in timestamp_col+input_metrics+target_metrics+[vm_number]: 
            logger.debug('Sample accepted.')
            
            # TODO:
            # Egyszerűen csak hozzá kéne fűznöm az nn_.csv-hez
            
            # ahogy elnézem az opt_utils modul importálva van.
            # tehát írnom kéne bele egy függvényt(aminek paraméterként)
            # átadom az új adatokat és hozzáfüzi az nn_.csv-hez
            
            df = opt_utils.readCSV(config.nn_filename)
            logger.info('----------------------------------------------')
            logger.info(f'pandas dataframe df.columns = {df.columns}')
            logger.info('----------------------------------------------')

            # Ne csak appendálja az adatokat hanem írja is vissza a csv-be
            tmp_df = df.append(pd.Series(timestamp_col+input_metrics+[vm_number]+target_metrics, index=df.columns ), ignore_index=True)
            
            print(timestamp_col+input_metrics+target_metrics)
            print(tmp_df.values)
            print(tmp_df.head())

            # Elmenteni ezt a tmp_df pandas dataframet ugyan abba a csv fájlba
            # tmp_df.to_csv('data/nn_training_data.csv', sep=',', encoding='utf-8', index=False)
            tmp_df.to_csv(config.nn_filename, sep=',', encoding='utf-8', index=False)
            logger.info('----------Data has been added and saved to csv file----------')
            
            # TODO:
            # Ha egy megadott számnál hosszabb a dataframe akkor kezdje el a tanítást
            logger.info(f'tmp_df.shape = {tmp_df.shape}')
            logger.info(f'tmp_df.shape[0] = {tmp_df.shape[0]}')
            
            # TRAINING
            # if( tmp_df.shape[0] > constants.get('training_samples_required', 10) ):
            if( tmp_df.shape[0] > 10 ):
                logger.info('There is enough data for start learning')
                global training_result
                # training_result = training_unit.train()
                # logger.debug(f'Training result:  {training_result}')
            else:
                logger.info('There is not enough data for start learning')
            
            
            


            logger.info('Samples received and processed.')   

        else:
            logger.error('Sample ignored because it contains empty value.')
    return jsonify('OK'), 200


@app.route('/optimizer/advice', methods=['GET'])
def get_advice():
    return 'Hello advice GET'

class RequestException(Exception):
    def __init__(self, status_code, reason, *args):
        super(RequestException, self).__init__(*args)
        self.status_code, self.reason = status_code, reason

    def to_dict(self):
        return dict(status_code=self.status_code,
                    reason=self.reason,
                    message=str(self))


@app.errorhandler(RequestException)
def handled_exception(error):
    global logger
    logger.error(f'An exception occured: {error.to_dict()}')
    return jsonify(error.to_dict())


@app.errorhandler(Exception)
def unhandled_exception(error):
    global logger
    import traceback as tb
    logger.error('An unhandled exception occured: %r\n%s',
                 error, tb.format_exc(error))
    response = jsonify(dict(message=error.args))
    response.status_code = 500
    return response