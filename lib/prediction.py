#import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
from seetings import settings
s = settings()

def get_error_measures(denormal_y, denormal_predicted):

    mae = np.mean(np.absolute(denormal_y - denormal_predicted))
    rmse = np.sqrt((np.mean((np.absolute(denormal_y - denormal_predicted)) ** 2)))
    nrsme_max_min = 100 * rmse / (denormal_y.max() - denormal_y.min())
    nrsme_mean = 100 * rmse / (denormal_y.mean())

    return mae, rmse, nrsme_max_min, nrsme_mean

def draw_graph_station(dataset, yTest, yTestPred, visualise=1, ax=None, drawu = True):

    #dataset is a class that contain the mean and variety of the data
    udenormalYTest = dataset.denormalize_data(yTest)
    udenormalPredicted = dataset.denormalize_data(yTestPred)

    mae, rmse, nrmse_maxMin, nrmse_mean = get_error_measures(udenormalYTest, udenormalPredicted)
    print('MAE = %7.7s - RMSE = %7.7s - nrmse_maxMin = %7.7s - nrmse_mean = %7.7s'%(mae, rmse, nrmse_maxMin, nrmse_mean))

    if visualise:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.plot(udenormalYTest, label='Real', color='blue')
        ax.plot(udenormalPredicted, label='Predicted', color='red')


    plt.savefig(s.log_path + '/result.png')
    return mae, rmse, nrmse_maxMin, nrmse_mean

#def draw_graph_all_stations(output_dir, dataset, n_stations, yTest, yTestPred):
    maeRmse = np.zeros((n_stations, 4))

    for staInd in range(n_stations):
        fig, ax = plt.subplots(figsize=(20, 10))
        maeRmse[staInd] = draw_graph_station(dataset, yTest, yTestPred, staInd, visualise=1, ax=ax)
        plt.xticks(range(0, len(yTest), 100))
        filename = '{}/finalEpoch_{}'.format(output_dir, staInd)
        plt.savefig('{}.png'.format(filename))

    errMean = maeRmse.mean(axis=0)
    print('OUTPUT : ', maeRmse.mean(axis=0))
