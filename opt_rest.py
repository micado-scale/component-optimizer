from flask import Flask, jsonify, request, send_file, render_template, Response
from ruamel import yaml

import logging
import logging.config

import opt_config
import opt_utils
import opt_trainer
import opt_advisor

import pandas as pd
import numpy as np


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

logger = None
config = None
training_unit = None
target_metrics = None
input_metrics = None
training_result = ['No error', None]
constants = {}

outsource_metrics = ['AVG_RR', 'SUM_RR']         # Our example application this should leave as it is
learning_round = 1                               # learn after n new sample
_last = True
is_reportable = False

# Akkor is átveszi a sample-ből ha ott van, ha nincs
vm_number = None
target_variable = None


def init_service(cfg):
    global logger
    logger = logging.getLogger('optimizer')

    global config
    config = opt_config.OptimizerConfig(cfg, 'optimizer')


@app.route('/init', methods=['POST'])
def init():
    logger.info('Loading constants from file...')
    constants_yaml = request.stream.read()
    if not constants_yaml:
        raise RequestException(400, 'Empty POST data')
    else:
        # ## ------------------------------------------------------------------------------------------------------
        # ## Reset some variables
        # ## ------------------------------------------------------------------------------------------------------
        global training_result
        training_result = ['No error', None]

        global is_reportable
        is_reportable = False

        # ## ------------------------------------------------------------------------------------------------------
        # ## Load configuration data
        # ## ------------------------------------------------------------------------------------------------------
        global constants
        constants = yaml.safe_load(constants_yaml).get('constants')
        logger.info('-------------------------------------------')
        logger.info('            Constants received             ')
        logger.info('-------------------------------------------')

        logger.info('-------------- GET CONSTANTS --------------')
        for k, v in constants.items():
            logger.info(f' {k} = {v}')
        logger.info('-------------------------------------------')

        logger.info('-------------------------------------------')
        logger.info('            Saving constants               ')
        logger.info('-------------------------------------------')
        opt_utils.write_yaml(config.constants_filename, constants)
        logger.info('Constants saved to "data/constants.yaml"')

        # ## ------------------------------------------------------------------------------------------------------
        # ## Assaigne input and target varables
        # ## ------------------------------------------------------------------------------------------------------
        logger.info('-------------------------------------------')
        logger.info('   Preparing database for training data    ')
        logger.info('-------------------------------------------')

        global input_metrics
        input_metrics = [metric.get('name')
                         for metric in constants.get('input_metrics')]

        global target_metrics
        target_metrics = [metric.get('name')
                          for metric in constants.get('target_metrics')]

        timestamp_col = ['timestamp']
        worker_count = ['vm_number']

        # ## ------------------------------------------------------------------------------------------------------
        # ## Create Data store csv file or use existing depends on the cofiguration
        # ## ------------------------------------------------------------------------------------------------------
        if( constants.get('knowledge_base') == 'use_existing' ):
            logger.info('File NOT created mode - use_existing')

        elif( constants.get('knowledge_base') == 'build_new' ):
            opt_utils.persist_data( config.nn_filename, timestamp_col+input_metrics+worker_count+target_metrics, 'w')
            logger.info('File created mode - build new')

        logger.info('-------------------------------------------')
        logger.info('  Created a .csv file for neural network   ')
        logger.info('-------------------------------------------')
        logger.info(f'csv saved to  {config.nn_filename}        ')

        logger.info('-------------------------------------------')
        logger.info('      Reset output file for advice         ')
        logger.info('-------------------------------------------')
        logger.info(f'csv  {config.output_filename}  reseted')

        # ## ------------------------------------------------------------------------------------------------------
        # ## Reset or delete output file where advices were stored
        # ## ------------------------------------------------------------------------------------------------------

        opt_utils.reset_output(config.output_filename)

        # ## ------------------------------------------------------------------------------------------------------
        # ## Init opt_advisor
        # ## ------------------------------------------------------------------------------------------------------

        global opt_advisor
        opt_advisor.init(constants.get('target_metrics'), input_metrics, worker_count, outsource_metrics, config, constants)

        # ## ------------------------------------------------------------------------------------------------------
        # ## Init opt_trainer
        # ## ------------------------------------------------------------------------------------------------------

        global opt_trainer
        training_samples_required = constants.get('training_samples_required')
        opt_trainer.init(target_metrics, input_metrics, worker_count, training_samples_required, outsource_metrics)

        logger.info('--------------------------------------------------------------')
        logger.info('          Optimizer REST initialized successfully             ')
        logger.info('--------------------------------------------------------------')

    return jsonify('OK'), 200



@app.route('/sample', methods=['POST'])
def sample():

    constants = opt_utils.read_yaml('data/constants.yaml')

    logger.info('-------------------------- YAML --------------------------')
    logger.info(f'Constants received: {constants}')
    logger.info('-------------------------- YAML --------------------------')

    sample_yaml = request.stream.read()
    if not sample_yaml:
        raise RequestException(400, 'Empty POST data')
    else:
        sample = yaml.safe_load(sample_yaml)
        logger.info(f'New sample received: {sample}')

        logger.info('')
        logger.info('--------------------------------------------------------------')
        logger.info('                   Getting sample data...                     ')
        logger.info('--------------------------------------------------------------')

        input_metrics = [metric.get('value')
                         for metric in sample.get('sample').get('input_metrics')]


        target_metrics = [metric.get('value')
                          for metric in sample.get('sample').get('target_metrics')]


        global target_variable
        target_variable = target_metrics

        # Ez volt a jó megoldás, de Józsi valmiért a Polcy Keeperben a sample-n kívül küldi el a vm_number értéket
        # ezért ezt most átírom, csak egy próba ereéig
        global vm_number
        vm_number = sample.get('sample').get('vm_number')
        # vm_number = sample.get('vm_number')

        timestamp_col = [sample.get('sample').get('timestamp')]

        logger.info('')
        logger.info('Sample data is going to be stored in corresponding variables.')
        logger.info('--------------------------------------------------------------')
        logger.info(f'      input_metrics = {input_metrics}')
        logger.info(f'      target_metrics = {target_metrics}')
        logger.info(f'      vm_number = {vm_number}')
        logger.info(f'      timestamp_col = {timestamp_col}')
        logger.info(f'      target_variable = {target_variable}')
        # logger.info(f'      np.isnan(target_metrics[0]) = {np.isnan(target_metrics[0])}') # can cause error if array is empty or None
        logger.info('      ----------------------- sample -----------------------')
        logger.info(f'      {sample.get("sample")}')
        logger.info('      ----------------------- sample -----------------------')
        logger.info('')
        logger.info('--------------------------------------------------------------')


        # if None not in timestamp_col+input_metrics+target_metrics+[vm_number]: 
        # if( len(input_metrics) != 0 and len(target_metrics) != 0 and None not in timestamp_col+input_metrics+target_metrics+[vm_number]):
        if( len(input_metrics) == len(constants.get('input_metrics')) and len(target_metrics) != 0 and None not in timestamp_col+input_metrics+target_metrics+[vm_number] and np.isnan(target_metrics[0]) == False):

            logger.info('----------------------------------------------')
            logger.info('Sample accepted.')
            logger.info('----------------------------------------------')
            logger.info(f'      (constants.get("target_metrics") = {constants.get("target_metrics")}')
            logger.info(f'      len(target_metrics) = {len(target_metrics)}')
            logger.info(f'      len(constants.get("target_metrics")) = {len(constants.get("target_metrics"))}')
            logger.info('----------------------------------------------')

            # itt csak beolvassa a csv fájlt és csinál belőle egy data framet
            df = opt_utils.readCSV(config.nn_filename)

            logger.info('----------------------------------------------')
            logger.info('  df = opt_utils.readCSV(config.nn_filename)  ')
            logger.info(f'  pandas dataframe df.columns = {df.columns}')
            logger.info('----------------------------------------------')

            # Hozzáfűzöm az új adatokat a beolvasott csv-ből csinált data framehez
            tmp_df = df.append(pd.Series(timestamp_col+input_metrics+[vm_number]+target_metrics, index=df.columns ), ignore_index=True)

            # Elmenteni ezt a tmp_df pandas dataframet ugyan abba a csv fájlba
            tmp_df.to_csv(config.nn_filename, sep=',', encoding='utf-8', index=False)

            logger.info('----------------------------------------------')
            logger.info("tmp_df.to_csv(config.nn_filename, sep=',', encoding='utf-8', index=False)'")
            logger.info('Data has been added and saved to csv file')
            logger.info('----------------------------------------------')





            # ## ------------------------------------------------------------------------------------------------------
            # ## Start Training
            # ## ------------------------------------------------------------------------------------------------------

            # Ha egy megadott számnál hosszabb a dataframe akkor kezdje el a tanítást
            logger.info(f'tmp_df.shape = {tmp_df.shape}')
            logger.info(f'tmp_df.shape[0] = {tmp_df.shape[0]}')

            _min_training = constants.get('training_samples_required')
            #TODO:
            #Itt van egy kis diszkrepancia
            #Mivel a trainer maga végez egy kis adattisztitást, lehet, hogy itt 300-ra van állítva a df érték, de a trainer maga
            #dobja a null sorokat, ezért leheet, hogy ott kisebb lesz a df szám amikor elkezd tanulni (mivel itt a df 300 ott a null
            #dobása után mondjuk 243

            logger.info('----------------------------------------------')
            logger.info(f'Now we have rows = {tmp_df.shape[0]}')
            logger.info(f'Minimum number when training start = {constants.get("training_samples_required")}')
            logger.info(f'Minimum number when training start = {_min_training}')
            logger.info('----------------------------------------------')

            if( tmp_df.shape[0] >= _min_training ):

                # TODO:
                # Kivezetni hogy hány mintánként tanuljon
                # Comment: Nehogy már minden körben tanítsuk
                if( tmp_df.shape[0] % learning_round == 0 ):

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

                    logger.info(f'\n\nTraining result = {training_result}\n')

                    if( training_result[0] == 'No error' ):
                        global is_reportable
                        is_reportable = True

                    logger.info(f' is_reportable = {is_reportable}\n')

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


@app.route('/advice', methods=['GET'])
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


    # logger.info('----------------------------------------------------------')
    # logger.info('------------------------ ADVISOR -------------------------')
    # logger.info('opt_utils.readCSV(config.nn_filename)')
    # logger.info('------------------------ ADVISOR -------------------------')
    # logger.info('----------------------------------------------------------')

    # df = opt_utils.readCSV(config.nn_filename)

    # logger.info('----------------------------------------------------------')
    # logger.info('------------------------ ADVISOR -------------------------')
    # logger.info(f'df.shape[0] = {df.shape[0]}')
    # logger.info('------------------------ ADVISOR -------------------------')
    # logger.info('----------------------------------------------------------')

    # Tehát igazából abban sem vagyok biztos, hogy az Adviser API
    # hívásánaál be kellene e olvasnaom a CSV-t
    # lehet, hogy ezt megtehetné maga az advisor is

    # Szerintem az kéne, hogy az adviser nem kap semmilyen előszürést
    # hanem saját maga megvizsgálja, hogy van-e elég adat és ha van
    # akkor ad javaslatot, ha nincs akkor a józsinak megfelelően
    # visszadja, hogy False

    # opt_advisor_return = opt_advisor.run(config.nn_filename, last = _last)

    # Felmerült az a probléma, hogy néha nem kapunk elfogadható mintát
    # a Sample API-ban, ezért két változót átadok az Advisornak
    # akkor is ha ezek nem lesznek letárolva az adatok között
    opt_advisor_return = opt_advisor.run(config.nn_filename, vm_number, target_variable, last = _last, training_result = training_result)

    logger.info('----------------------------------------------------------')
    logger.info('------------------ opt_advisor_return --------------------')
    logger.info(f'opt_advisor_return with message: {opt_advisor_return}')
    logger.info('------------------ opt_advisor_return --------------------')
    logger.info('----------------------------------------------------------')

    logger.info('Get Advice recieved and processed.')

    return opt_advisor_return


@app.route('/report', methods=['GET'])
def report():
    global is_reportable
    logger.info('----------------------------------------------------------')
    logger.info('        application report method has been called         ')
    logger.info('----------------------------------------------------------')
    if( is_reportable == True ):
        logger.info('        application is reportable         ')
        return render_template('index.html')
    else:
        logger.info('        application is NOT reportable         ')
        return 'There is not enough sample to get report'


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


@app.errorhandler(NameError)
def name_error_exception(error):
    global logger
    logger.error(f'An exception occured: {error}')
    return jsonify('NameError'), 400


@app.errorhandler(Exception)
def unhandled_exception(error):
    global logger
    import traceback as tb
    logger.error('An unhandled exception occured: %r\n%s',
                 error, tb.format_exc(error))
    response = jsonify(dict(message=error.args))
    response.status_code = 500
    return response
