from utils import setup_logger
LOGGER_DISABLED = {
'nnloss':False
, 'elmloss':False
}
run_folder = './loss/'
logger_nn = setup_logger('logger_nn', run_folder + '/logger_nn.log')
logger_nn.disabled = LOGGER_DISABLED['nnloss']

logger_elm = setup_logger('logger_elm', run_folder + '/logger_elm.log')
logger_elm.disabled = LOGGER_DISABLED['elmloss']