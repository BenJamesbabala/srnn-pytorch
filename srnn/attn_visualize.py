'''
Visualization script for the attention structural RNN model

Author: Anirudh Vemula
Date: 8th May 2017
'''
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import seaborn


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

    fig = plt.figure()
    for tstep in range(traj_length):
        if tstep < observed_length:
            # Observed part
            continue
        # Predicted part
        # Create a plot for current prediction tstep
        # TODO for now, just plot the attention for the first step
        if len(nodes_present[observed_length-1]) == 1:
            # Only one pedestrian, so no spatial edges
            continue
        for ped in nodes_present[observed_length-1]:
            true_traj_ped = (np.array(traj_data[ped][0]) + 1) / 2
            pred_traj_ped = (np.array(traj_data[ped][1]) + 1) / 2

            peds_other = attn_weights[tstep-observed_length][ped][1]
            attn_w = attn_weights[tstep-observed_length][ped][0]

            # fig = plt.figure()
            ax = fig.gca()
            c = 'r'
            # ipdb.set_trace()
            list_of_points = range(true_traj_ped[:, 0].size-1)
            plt.plot(true_traj_ped[:, 0], true_traj_ped[:, 1], color=c, linestyle='solid', linewidth=1, marker='o', markevery=list_of_points)
            plt.scatter(true_traj_ped[-1, 0], true_traj_ped[-1, 1], color='b', marker='D')
            # plt.plot(pred_traj_ped[:, 0], pred_traj_ped[:, 1], color=c, linestyle='dashed', marker='x', linewidth=1)

            for ind_ped, ped_o in enumerate(peds_other):
                true_traj_ped_o = (np.array(traj_data[ped_o][0]) + 1) / 2
                pred_traj_ped_o = (np.array(traj_data[ped_o][1]) + 1) / 2

                weight = attn_w[ind_ped]

                c = np.random.rand(3)
                list_of_points = range(true_traj_ped_o[:, 0].size-1)
                plt.plot(true_traj_ped_o[:, 0], true_traj_ped_o[:, 1], color=c, linestyle='solid', linewidth=1, marker='o', markevery=list_of_points)
                plt.scatter(true_traj_ped_o[-1, 0], true_traj_ped_o[-1, 1], color='b', marker='D')
                circle = plt.Circle((true_traj_ped_o[-1, 0], true_traj_ped_o[-1, 1]), weight*0.1, fill=False, color='b', linewidth=2)
                ax.add_artist(circle)
                # plt.plot(pred_traj_ped_o[:, 0], pred_traj_ped_o[:, 1], color=c, linestyle='dashed', marker='x', linewidth=2*weight)

            # plt.ylim((1, 0))
            # plt.xlim((0, 1))
            ax.set_aspect('equal')
            # plt.show()
            plt.savefig(plot_directory+'/'+name+'_'+str(ped)+'.png')
            # plt.close('all')
            plt.clf()

        break
    plt.close('all')


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--test_dataset', type=int, default=0,
                        help='test dataset index')

    # Parse the parameters
    args = parser.parse_args()

    save_directory = 'save/'
    save_directory += str(args.test_dataset) + '/save_attention/'
    plot_directory = 'plot/plot_attention_viz/'+str(args.test_dataset)

    f = open(save_directory+'results.pkl', 'rb')
    results = pickle.load(f)

    for i in range(len(results)):
        # For each sequence
        print(i)
        true_pos_nodes = results[i][0]
        pred_pos_nodes = results[i][1]
        nodes_present = results[i][2]
        observed_length = results[i][3]
        attn_weights = results[i][4]

        name = 'sequence' + str(i)

        plot_attention(true_pos_nodes, pred_pos_nodes, nodes_present, observed_length, attn_weights, name, plot_directory)


if __name__ == '__main__':
    main()
