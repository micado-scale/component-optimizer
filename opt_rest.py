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
    logger.info('Loading training sample...')
    
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
            logger.debug('Sample accepted.')
            
            # TODO:
            # Egyszerűen csak hozzá kéne fűznöm az nn_.csv-hez
            
            # ahogy elnézem az opt_utils modul importálva van.
            # tehát írnom kéne bele egy függvényt(aminek paraméterként)
            # átadom az új adatokat és hozzáfüzi az nn_.csv-hez

            # itt csak beolvassa a csv fájlt és csinál belőle egy data framet
            df = opt_utils.readCSV(config.nn_filename)
            logger.info('----------------------------------------------')
            logger.info(f'pandas dataframe df.columns = {df.columns}')
            logger.info('----------------------------------------------')

            # Hozzáfűzöm az új adatokat a beolvasott csv-ből csinált data framehez
            tmp_df = df.append(pd.Series(timestamp_col+input_metrics+[vm_number]+target_metrics, index=df.columns ), ignore_index=True)
            
            # print(timestamp_col+input_metrics+target_metrics)
            # print(tmp_df.values)
            # print(tmp_df.head())

            # Elmenteni ezt a tmp_df pandas dataframet ugyan abba a csv fájlba
            # tmp_df.to_csv('data/nn_training_data.csv', sep=',', encoding='utf-8', index=False)
            tmp_df.to_csv(config.nn_filename, sep=',', encoding='utf-8', index=False)
            logger.info('----------Data has been added and saved to csv file----------')
            
            # Ha egy megadott számnál hosszabb a dataframe akkor kezdje el a tanítást
            logger.info(f'tmp_df.shape = {tmp_df.shape}')
            logger.info(f'tmp_df.shape[0] = {tmp_df.shape[0]}')
                        
            # TRAINING
            logger.info(constants.get('training_samples_required'))
            # if( tmp_df.shape[0] > constants.get('training_samples_required') ):
            if( tmp_df.shape[0] > 4 ):

                logger.info('There is enough data for start learning')

                # TODO:
                # Kivezetni hogy hány mintánként tanuljon
                # Comment: Nehogy már minden körben tanítsuk
                if( tmp_df.shape[0] % 1 == 0 ):

                    logger.info('----------Learning Neural Network and Linear Regression Phase----------')
                    
                    global training_result

                    training_result = opt_trainer.run(config.nn_filename, visualize = False)
                    
                    logger.info(f'Training result = {training_result}')
                    
                    
                    # TODO:
                    # A traningin_results-ba leginkább a tanulást leíró metrikákat kéne tenni
                    # semmi esetre sem az adatok becslését
                    
                    # TODO:
                    # Más kérdés, hogy a metrikákat akár minden alkalommal el lehet kérni
                    # Akár tanítás nélkül is, vagy azért mert el vannak tárolva
                    # valahol a trainerben, vagy mert relatíve könnyű öket kiszámolni
                    # de életszerűbbnek tartom azt a helyzetet, ha csak akkor kérjük el amikor
                    # tanulás is történ vagy ezt is bizonyos időközönként és nem minden lépésben
                    
                    
                    # TODO:
                    # Jó lenne ha ez a metodus tényleg csak az éppen aktuális adatokat kapná meg
                    # Ellenben back-test-hez kimondottan jó lenne ha komplet csv elérési utat adnék neki
                    # Vagy akár megkaphatja a komplet adatokat is
                    
                    # TODO:
                    # Ez el fog hasalni ha kevés eset van ezért belevarni magába az opt_advisor--ba
                    # hogyha még meg is hívják, akkor olvassa be a csv fájlt amiből dolgoznia kell
                    # de ha annak hossza rövidebb egy megadott értéknék akkor ne hajtsa végre
                    # és térjen vissza valamilyen üzenettel
                    
                    # opt_advisor.run(config.nn_filename, last = True)
                    
                    # Az opt_adviser_old.run() csak meghagytam, hogyha egy régi csv-t szerenénk tesztelni vele
                    
                    # opt_advisor_old.run()
                    
                    # opt_advisor.run(tmp_df[:-1])

            else:
                logger.info('There is not enough data for start learning')

            logger.info('Samples received and processed.')   

        else:
            logger.error('Sample ignored because it contains empty value.')
    return jsonify('OK'), 200




@app.route('/optimizer/advice', methods=['GET'])
def get_advice():
    global _last
    _last = request.args.get('last', default = True)
    
    logger.info(f'_last variable set = {_last}')

    logger.info('Advisor get_advice() called')

    constants = opt_utils.read_yaml('data/constants.yaml')
    # logger.debug('-------------------------- YAML --------------------------')
    # logger.debug(f'Constants received: {constants}')
    # logger.debug('-------------------------- YAML --------------------------')
    
    global opt_advisor
    opt_advisor.init(constants.get('target_metrics'))

    # Nos igazából ennek a modulnak semmilyen adatot nem kell kapnia
    # ugyanis kiolvassa az adatokat egy korábban eltárolt fájlból
    
    # Ha a file tartalma kisebb mint egy előre meghatározott érték
    # akkor nem fut le vagy olyan értékkel tér vissza amit a felhasználó
    # értelmezni tud
    
    df = opt_utils.readCSV(config.nn_filename)
    # logger.debug('----------------------------------------------')
    # logger.debug(f'pandas dataframe df.columns = {df.columns}')
    # logger.debug('----------------------------------------------')

    # TODO:
    # Ha egy megadott számnál hosszabb a dataframe akkor adjon tanácsot különben ne
    # logger.debug(f'df.shape = {df.shape}')
    # logger.debug(f'df.shape[0] = {df.shape[0]}')
    # logger.debug(f'constants.get("training_samples_required") = {constants.get("training_samples_required")})
            

    # logger.info('There is enough data for get advice')
    # logger.info('---------Get Advice Phase----------')

    opt_advisor_return = opt_advisor.run(config.nn_filename, last = _last)

    # logger.info('---------------------------------------- opt_advisor_return ----------------------------------------')
    # logger.info(opt_advisor_return)
    # logger.info('---------------------------------------- opt_advisor_return ----------------------------------------')

    # Az opt_adviser_old.run() csak meghagytam, hogyha egy régi csv-t szerenénk tesztelni vele                    
    # opt_advisor_old.run()


    # print('---constans= ', constants.get('input_metrics'))
    # print('---constans= ', constants)
        
    logger.info('Get Advice recieved and processed.')
    
    return opt_advisor_return
    


@app.route('/optimizer/backtest', methods=['POST'])
def backtest():
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
            
            # TRAINING AND TESTING
            # if( tmp_df.shape[0] > constants.get('training_samples_required', 10) ):
            if( tmp_df.shape[0] > 10 ):
                logger.info('There is enough data for start learning')
                global training_result
                # TODO:
                # Csináljunk egy függvényt valahová akinek odaadhatom a tmp_df dataframet
                # az eredményt tároljuk el a global training_result változóban
                logger.info('----------Learning Neural Network and Linear Regression Phase----------')

                # training_result = opt_trainer.run()

                opt_trainer_backtest.run(config.nn_filename)

                opt_advisor_backtest.run()

                # TODO:
                # A traningin_results-ba leginkább a tanulást leíró metrikákat kéne tenni
                # semmi esetre sem az adatok becslését
                    
            else:
                logger.info('There is not enough data for start learning')

            logger.info('Samples received and processed.')   

        else:
            logger.error('Sample ignored because it contains empty value.')
    return jsonify('OK'), 200




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