from flask import Flask, jsonify, request, send_file
from ruamel import yaml

import logging
import logging.config

import opt_config
import opt_utils

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


@app.route('/optimizer/hello', methods=['GET'])
def hello():
    return 'Hello Optimizer'


@app.route('/optimizer/init', methods=['POST'])
def init():
    logger.debug('Loading constants from file...')
    constants_yaml = request.stream.read()
    if not constants_yaml:
        raise RequestException(400, 'Empty POST data')
    else:
        global constants
        constants = yaml.safe_load(constants_yaml).get('constants')
        logger.debug(f'Constants received: {constants}')
        
        # nekem ezt nem elmentenem kéne, hanem appendálnom, ha éppen nincs ott a file
        # mondjuk a file neve legyen trainging data
        # és kell írni egy metodust arra is, hogy törölje azt a szerencsétlet (reset)

        logger.info('Saving constants...')
        opt_utils.write_yaml(config.constants_filename, constants)
        logger.info('Constants saved to "data/constants.yaml"')
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

        logger.info('Creating a .csv file for neural network...')
        opt_utils.persist_data(
            config.nn_filename, timestamp_col+input_metrics+target_metrics, 'w')
        logger.info('File created')


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
    
    # le kéne ellenőriznem, hogy létezik-e a csv file amibe beleírom
    # ha nem akkor parsolja azt a yaml filét
    # és hozzon létre az alapján egy csv filét a data könyvtárban
    # ha létezik, akkor appendálja hozzá az adatot
    # most hiba kezelésről, ha nem megfelelő formátumu a yaml nem beszélünk
    return 'Hello sample POST'


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