    # trajectory_x = np.zeros((T, total_UE)) # shape(sequence_length, total_UE)
    # trajectory_y = np.zeros((T, total_UE))

    # # Trajectory
    # trajectory_x[0] = locux
    # trajectory_y[0] = locuy
    # for t in range(1, T):
    #     for i in range(total_UE):
    #         move_x = np.random.normal(0, 1) * np.random.choice([-1,1]) * multi_distance[a]
    #         move_y = np.random.normal(0, 1) * np.random.choice([-1,1]) * multi_distance[a]
    #         trajectory_x[t, i] = trajectory_x[0, i] + move_x
    #         trajectory_y[t, i] = trajectory_y[0, i] + move_y
            
    # # Plot trajectory
    # for i in range(total_UE):
    #     plt.plot(trajectory_x.T[i], trajectory_y.T[i])
    # plt.scatter(locrux, locruy)
    # plt.title('UE Trajectory')
    # plt.grid()
    # plt.show()