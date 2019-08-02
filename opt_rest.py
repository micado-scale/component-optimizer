from flask import Flask, jsonify, request, send_file
from ruamel import yaml

import logging
import logging.config

import opt_config
import opt_utils
import opt_trainer
import opt_advisor
import opt_advisor_old
import opt_trainer_backtest
import opt_advisor_backtest

import pandas as pd


app = Flask(__name__)

logger = None
config = None
training_unit = None
_last = True

training_result = []
target_metrics = None
constants = {}


def init_service(cfg):
    global logger
    logger = logging.getLogger('optimizer')

    global config
    config = opt_config.OptimizerConfig(cfg, 'optimizer')


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

        logger.info('Saving constants...')
        opt_utils.write_yaml(config.constants_filename, constants)
        logger.info('Constants saved to "data/constants.yaml"')

        logger.info('Preparing database for training data...')
        input_metrics = [metric.get('name')
                         for metric in constants.get('input_metrics')]
        
        global target_metrics
        target_metrics = [metric.get('name')
                          for metric in constants.get('target_metrics')]

        timestamp_col = ['timestamp']
        
        worker_count = ['vm_number']

        logger.info('Creating a .csv file for neural network...')
        
        opt_utils.persist_data(
            config.nn_filename, timestamp_col+input_metrics+worker_count+target_metrics, 'w')
        
        logger.info('File created')
        
        global opt_advisor
        opt_advisor.init(constants.get('target_metrics'))
        
        global opt_trainer
        opt_trainer.init(target_metrics, input_metrics, worker_count)

        logger.info('Optimizer REST initialized successfully ')
    return jsonify('OK'), 200


@app.route('/optimizer/sample', methods=['POST'])
def sample():
    
    constants = opt_utils.read_yaml('data/constants.yaml')
    
    logger.info('----------------------------------------------------------')
    logger.info('-------------------------- YAML --------------------------')
    logger.info(f'Constants received: {constants}')
    logger.info('-------------------------- YAML --------------------------')
    logger.info('----------------------------------------------------------')
    
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

        # print(timestamp_col+input_metrics+target_metrics+[vm_number])
        
        if None not in timestamp_col+input_metrics+target_metrics+[vm_number]: 
            logger.info('----------------------------------------------')
            logger.info('Sample accepted.')
            logger.info('----------------------------------------------')
            
            # itt csak beolvassa a csv fájlt és csinál belőle egy data framet
            df = opt_utils.readCSV(config.nn_filename)
            
            logger.info('----------------------------------------------')
            logger.info(f'pandas dataframe df.columns = {df.columns}')
            logger.info('----------------------------------------------')

            # Hozzáfűzöm az új adatokat a beolvasott csv-ből csinált data framehez
            tmp_df = df.append(pd.Series(timestamp_col+input_metrics+[vm_number]+target_metrics, index=df.columns ), ignore_index=True)

            # Elmenteni ezt a tmp_df pandas dataframet ugyan abba a csv fájlba
            tmp_df.to_csv(config.nn_filename, sep=',', encoding='utf-8', index=False)

            logger.info('----------------------------------------------')
            logger.info('Data has been added and saved to csv file')
            logger.info('----------------------------------------------')

            
            # Ha egy megadott számnál hosszabb a dataframe akkor kezdje el a tanítást
            logger.info(f'tmp_df.shape = {tmp_df.shape}')
            logger.info(f'tmp_df.shape[0] = {tmp_df.shape[0]}')
                        
            # TRAINING
            logger.info(constants.get('training_samples_required'))
            # if( tmp_df.shape[0] > constants.get('training_samples_required') ):
            _min_training = 300

            logger.info('----------------------------------------------')
            logger.info(f'Now we have rows = {tmp_df.shape[0]}')
            logger.info(f'Minimum number when training start = {_min_training}')
            logger.info('----------------------------------------------')

            if( tmp_df.shape[0] > _min_training ):

                # TODO:
                # Kivezetni hogy hány mintánként tanuljon
                # Comment: Nehogy már minden körben tanítsuk
                if( tmp_df.shape[0] % 1 == 0 ):

                    logger.info('----------------------------------------------')
                    logger.info(f'Now we have rows = {tmp_df.shape[0]}')
                    logger.info('We have enough data to start learning')
                    logger.info('----------------------------------------------')

                    logger.info('----------------------------------------------')
                    logger.info('Learning NN and Linear Regression Phase')
                    logger.info('----------------------------------------------')

                    
                    global training_result

                    logger.info('----------------------------------------------')
                    logger.info('opt_trainer.run(config.nn_filename)')
                    logger.info('----------------------------------------------')

                    training_result = opt_trainer.run(config.nn_filename, visualize = False)
                    
                    logger.info(f'Training result = {training_result}')
                    
                    # opt_advisor.run(config.nn_filename, last = True)
                    # Az opt_adviser_old.run() csak meghagytam, hogyha egy régi csv-t szerenénk tesztelni vele
                    # opt_advisor_old.run()
                    
            else:
                logger.info('----------------------------------------------')
                logger.info('There is not enough data for start learning')
                logger.info('----------------------------------------------')

            logger.info('----------------------------------------------')
            logger.info('Samples received and processed.')
            logger.info('----------------------------------------------')

        else:
            logger.info('----------------------------------------------')
            logger.info('Sample ignored cause it contains empty value.')
            logger.info('----------------------------------------------')
            
    return jsonify('OK'), 200




@app.route('/optimizer/advice', methods=['GET'])
def get_advice():
    global _last
    _last = request.args.get('last', default = True)
    
    logger.info('----------------------------------------------------------')
    logger.info('------------------------ ADVISOR -------------------------')
    logger.info(f'_last parameter variable set = {_last}')
    logger.info('------------------------ ADVISOR -------------------------')
    logger.info('----------------------------------------------------------')

    
    
    logger.info('----------------------------------------------------------')
    logger.info('------------------------ ADVISOR -------------------------')
    logger.info('Advisor get_advice() called')
    logger.info('------------------------ ADVISOR -------------------------')
    logger.info('----------------------------------------------------------')
    
    
    constants = opt_utils.read_yaml('data/constants.yaml')
    
    logger.info('----------------------------------------------------------')
    logger.info('-------------------------- YAML --------------------------')
    logger.info(f'Constants received: {constants}')
    logger.info('-------------------------- YAML --------------------------')
    logger.info('----------------------------------------------------------')

    
    logger.info('----------------------------------------------------------')
    logger.info('------------------------ ADVISOR -------------------------')
    logger.info('opt_advisor.init() called')
    logger.info('------------------------ ADVISOR -------------------------')
    logger.info('----------------------------------------------------------')
    
    global opt_advisor
    opt_advisor.init(constants.get('target_metrics'))

    
    logger.info('----------------------------------------------------------')
    logger.info('------------------------ ADVISOR -------------------------')
    logger.info('opt_utils.readCSV(config.nn_filename)')
    logger.info('------------------------ ADVISOR -------------------------')
    logger.info('----------------------------------------------------------')
    
    df = opt_utils.readCSV(config.nn_filename)
    
    logger.info('----------------------------------------------------------')
    logger.info('------------------------ ADVISOR -------------------------')
    logger.info(f'df.shape[0] = {df.shape[0]}')
    logger.info('------------------------ ADVISOR -------------------------')
    logger.info('----------------------------------------------------------')



    # Tehát igazából abban sem vagyok biztos, hogy az Adviser API
    # hívásánaál be kellene e olvasnaom a CSV-t
    # lehet, hogy ezt megtehetné maga az advisor is

    # Szerintem az kéne, hogy az adviser nem kap semmilyen előszürést
    # hanem saját maga megvizsgálja, hogy van-e elég adat és ha van
    # akkor ad javaslatot, ha nincs akkor a józsinak megfelelően
    # visszadja, hogy False
    
    ## opt_advisor_return = opt_advisor.run(config.nn_filename, last = _last)

    opt_advisor_return = 'Semmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmiii'
    
    # logger.info('---------------------------------------- opt_advisor_return ----------------------------------------')
    # logger.info(opt_advisor_return)
    # logger.info('---------------------------------------- opt_advisor_return ----------------------------------------')


    # print('---constans= ', constants.get('input_metrics'))
    # print('---constans= ', constants)
        
    logger.info('Get Advice recieved and processed.')
    
    return opt_advisor_return
    

@app.route('/', methods=['GET'])
def index():
    return 'MiCado Optimizer Modul'


@app.route('/optimizer/hello', methods=['GET'])
def hello():
    return 'Hello Optimizer'


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