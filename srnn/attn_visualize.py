'''
Visualization script for the attention structural RNN model

Author: Anirudh Vemula
Date: 8th May 2017
'''
import numpy as np
import matplotlib.pyplot as plt
import pickle


def plot_attention(true_pos_nodes, pred_pos_nodes, nodes_present, observed_length, attn_weights, name, plot_directory):
    traj_length, numNodes, _ = true_pos_nodes.shape

    traj_data = {}
    for tstep in range(traj_length):
        pred_pos = pred_pos_nodes[tstep, :]
        true_pos = true_pos_nodes[tstep, :]

        for ped in range(numNodes):
            if ped not in traj_data and tstep < observed_length:
                traj_data[ped] = [[], []]

            if ped in nodes_present[tstep]:
                traj_data[ped][0].append(true_pos[ped, :])
                traj_data[ped][1].append(pred_pos[ped, :])

    for j in traj_data:
        # For each ped
        for tstep in range(traj_length):
            if tstep < observed_length:
                # Observed part
                continue
            # Predicted part
            


def main():
    save_directory = 'save/save_attention'
    plot_directory = 'plot/plot_attention_viz'

    f = open(save_directory+'/results.pkl', 'rb')
    results = pickle.load(f)

    for i in range(len(results)):
        # For each sequence
        print i
        true_pos_nodes = results[i][0]
        pred_pos_nodes = results[i][1]
        nodes_present = results[i][2]
        observed_length = results[i][3]
        attn_weights = results[i][4]

        name = 'sequence' + str(i)

        plot_attention(true_pos_nodes, pred_pos_nodes, nodes_present, observed_length, attn_weights, name, plot_directory)


if __name__ == '__main__':
    main()
