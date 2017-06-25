import matplotlib.pyplot as plt
import seaborn
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default=5)
args = parser.parse_args()

log_directory = 'log'
plot_directory= 'plot'

if args.dataset == 5:

	for i in range(5):
		training_val_loss = np.genfromtxt(os.path.join(log_directory, str(i), 'log_attention', 'log_curve.txt'), delimiter=',')
	
		xaxis = training_val_loss[:, 0]
		training_loss = training_val_loss[:, 1]
		val_loss = training_val_loss[:, 2]

		output_plot = os.path.join(plot_directory, 'plot_'+str(i)+'.png')

		plt.figure()
		plt.plot(xaxis, training_loss, 'r-', label='Training loss')
		plt.plot(xaxis, val_loss, 'b-', label='Validation loss')
		plt.title('Training and Validation loss vs Epoch for dataset ' + str(i))
		plt.legend()
		plot_file = os.path.join(plot_directory, 'training_curve_'+str(i)+'.png')
		plt.savefig(plot_file, bbox_inches='tight')

else:
	i = args.dataset
	training_val_loss = np.genfromtxt(os.path.join(log_directory, str(i), 'log_attention', 'log_curve.txt'), delimiter=',')

        xaxis = training_val_loss[:, 0]
        training_loss = training_val_loss[:, 1]
        val_loss = training_val_loss[:, 2]

        output_plot = os.path.join(plot_directory, 'plot_'+str(i)+'.png')

        plt.figure()
        plt.plot(xaxis, training_loss, 'r-', label='Training loss')
        plt.plot(xaxis, val_loss, 'b-', label='Validation loss')
        plt.title('Training and Validation loss vs Epoch for dataset ' + str(i))
        plt.legend()
        plot_file = os.path.join(plot_directory, 'training_curve_'+str(i)+'.png')
        plt.savefig(plot_file, bbox_inches='tight')



