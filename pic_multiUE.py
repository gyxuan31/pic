import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
np.set_printoptions(precision=2, suppress=True)

# load parameters
params = loadmat('multi_UE.mat')

T = int(params['T'].squeeze())
T_ref = 20
num_RU = int(params['num_RU'].squeeze())
num_RB = int(params['num_RB'].squeeze())
num_ref = int(params['num_ref'].squeeze())
gamma = params['gamma'].squeeze()
num_setreq = int(params['num_setreq'].squeeze())
B = float(params['B'].squeeze())
P = params['P'].squeeze()
sigmsqr = params['sigmsqr'].squeeze()
eta = float(params['eta'].squeeze())
predicted_len = int(params['predicted_len'].squeeze())
rayleigh_gain = params['rayleigh_gain']
multi_num_UE = params['multi_num_UE'].squeeze()
num_UE = [6, 12, 18, 24, 30, 36]
distance = params['multi_distance_true'].squeeze()

num_point = 6 # number of UE group

# load output total_UE=[9 18 27]
output = loadmat('multi_output.mat')
multi_rec_dr_random = output['multi_rec_dr_random'].squeeze()
multi_rec_dr_avg = output['multi_rec_dr_avg'].squeeze()
multi_rec_dr_op = output['multi_rec_dr_op'].squeeze()
multi_rec_e_random = output['multi_rec_e_random'].squeeze()
multi_rec_e_avg = output['multi_rec_e_avg'].squeeze()
multi_rec_e_op = output['multi_rec_e_op'].squeeze()

# load output total_UE=[36 45]
output1 = loadmat('multi_output1.mat')
multi_rec_dr_random1 = output1['multi_rec_dr_random'].squeeze()
multi_rec_dr_avg1 = output1['multi_rec_dr_avg'].squeeze()
multi_rec_dr_op1 = output1['multi_rec_dr_op'].squeeze()
multi_rec_e_random1 = output1['multi_rec_e_random'].squeeze()
multi_rec_e_avg1 = output1['multi_rec_e_avg'].squeeze()
multi_rec_e_op1 = output1['multi_rec_e_op'].squeeze()

# load output total_UE=[6 12 24 30]
params_sup = loadmat('multi_UE_sup.mat')
multi_num_UE_sup = params_sup['multi_num_UE'].squeeze()
distance_sup = params_sup['multi_distance_true'].squeeze()
output1 = loadmat('multi_output_sup.mat')
multi_rec_dr_random_sup = output1['multi_rec_dr_random'].squeeze()
multi_rec_dr_avg_sup = output1['multi_rec_dr_avg'].squeeze()
multi_rec_dr_op_sup = output1['multi_rec_dr_op'].squeeze()
multi_rec_e_random_sup = output1['multi_rec_e_random'].squeeze()
multi_rec_e_avg_sup = output1['multi_rec_e_avg'].squeeze()
multi_rec_e_op_sup = output1['multi_rec_e_op'].squeeze()

util_op_mean = np.zeros(num_point)
util_random_mean = np.zeros(num_point)
util_avg_mean = np.zeros(num_point)
dr_random = np.zeros(num_point)
dr_avg = np.zeros(num_point)
dr_op = np.zeros(num_point)

for a in range(1,2): # *18 total_UE=[9 18 27]  final[6 12 18(2) 24 30 36(5)]
    total_UE = multi_num_UE[a] * num_RU

    util_random = []
    util_avg = []
    util_op = []

    for t in range(num_ref, num_ref+T_ref):
        e_op = np.array(multi_rec_e_op[a,t,:total_UE,:]) #(T, total_UE, num_RB)
        e_random = np.array(multi_rec_e_random[a,t,:total_UE,:])
        e_avg =  np.array(multi_rec_e_avg[a,t,:total_UE,:])
        # RANDOM
        util_random_list = np.any(e_random, axis=0)  # (num_RB,)
        temp = np.sum(util_random_list)
        util_random.append(temp / float(num_RB))

        # AVG
        util_avg_list = np.any(e_avg, axis=0)
        util_avg.append(np.sum(util_avg_list) / float(num_RB))

        # OP
        util_op_list = np.any(e_op, axis=0)
        util_op.append(np.sum(util_op_list) / float(num_RB))
        # print(np.sum(util_op_list))
        
    idx = 2
    
    util_op_mean[idx] = np.mean(np.array(util_op))
    util_random_mean[idx] = np.mean(np.array(util_random))
    util_avg_mean[idx] = np.mean(np.array(util_avg))
    
    dr_op[idx] = multi_rec_dr_op[a] / total_UE
    dr_avg[idx] = multi_rec_dr_avg[a] / total_UE
    dr_random[idx] = multi_rec_dr_random[a] / total_UE



for a in range(3,4): # *36 total_UE=[9 18 27 / 36 45] final[6 12 18(2) 24 30 36(5)]
    total_UE = multi_num_UE[a] * num_RU

    util_random = []
    util_avg = []
    util_op = []

    for t in range(13, 13+T_ref):
        e_op = np.array(multi_rec_e_op1[a,t,:total_UE,:]) #(T, total_UE, num_RB)
        e_random = np.array(multi_rec_e_random[a,t,:total_UE,:])
        e_avg =  np.array(multi_rec_e_avg[a,t,:total_UE,:])
        # RANDOM
        util_random_list = np.any(e_random, axis=0)  # (num_RB,)
        temp = np.sum(util_random_list)
        util_random.append(temp / float(num_RB))

        # AVG
        util_avg_list = np.any(e_avg, axis=0)
        util_avg.append(np.sum(util_avg_list) / float(num_RB))

        # OP
        util_op_list = np.any(e_op, axis=0)
        util_op.append(np.sum(util_op_list) / float(num_RB))
        # print(np.sum(util_op_list))
        
    idx = 5
    
    util_op_mean[idx] = np.mean(np.array(util_op))
    util_random_mean[idx] = np.mean(np.array(util_random))
    util_avg_mean[idx] = np.mean(np.array(util_avg))

    dr_op[idx] = multi_rec_dr_op1[a-3] / total_UE
    dr_avg[idx] = multi_rec_dr_avg[a] / total_UE
    dr_random[idx] = multi_rec_dr_random[a] / total_UE

for a in range(4): # total_UE=[6 12 24 30] final[6 12 18(2) 24 30 36(5)]
    total_UE = multi_num_UE_sup[a] * num_RU

    util_random = []
    util_avg = []
    util_op = []

    for t in range(num_ref, num_ref+T_ref):
        e_op = np.array(multi_rec_e_op_sup[a,t,:total_UE,:]) #(T, total_UE, num_RB)
        e_random = np.array(multi_rec_e_random_sup[a,t,:total_UE,:])
        e_avg =  np.array(multi_rec_e_avg_sup[a,t,:total_UE,:])
        # RANDOM
        util_random_list = np.any(e_random, axis=0)  # (num_RB,)
        temp = np.sum(util_random_list)
        util_random.append(temp / float(num_RB))

        # AVG
        util_avg_list = np.any(e_avg, axis=0)
        util_avg.append(np.sum(util_avg_list) / float(num_RB))

        # OP
        util_op_list = np.any(e_op, axis=0)
        util_op.append(np.sum(util_op_list) / float(num_RB))
        # print(np.sum(util_op_list))
        
    if a == 2 or a == 3:
        idx = a + 1
    else:
        idx = a
        
    util_op_mean[idx] = np.mean(np.array(util_op))
    util_random_mean[idx] = np.mean(np.array(util_random))
    util_avg_mean[idx] = np.mean(np.array(util_avg))

    dr_op[idx] = multi_rec_dr_op_sup[a] / total_UE
    dr_avg[idx] = multi_rec_dr_avg_sup[a] / total_UE
    dr_random[idx] = multi_rec_dr_random_sup[a] / total_UE

# print(dr_op)
# print(dr_avg)

# Plot - Utilization
plt.figure()
plt.plot(util_random_mean, label='Random', marker='D', markersize=6, color='#3480b8')
plt.plot(util_avg_mean, label='Average', marker='D', markersize=6, color='#8fbc8f')
plt.plot(util_op_mean, label='MPC', marker='D', markersize=6, color='#c82423')
plt.xlabel('UE number')
plt.ylabel('RB Utilization (%)')
xtick = [a for a in num_UE]
plt.xticks([a for a in range(len(num_UE))], xtick)
plt.ylim((0,1.1))
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Plot - Geometric Mean of Data Rate
plt.figure()
plt.plot(dr_random, label='Random', marker='D', markersize=6, color='#3480b8')
plt.plot(dr_avg, label='Average', marker='D', markersize=6, color='#8fbc8f')
plt.plot(dr_op, label='MPC', marker='D', markersize=6, color='#c82423')
plt.xlabel('UE number')
plt.ylabel('Geometric Mean of Data Rate')
plt.xticks([a for a in range(len(num_UE))], xtick)
plt.legend()
plt.grid()
plt.show()

# Plot - every RU
fig, axes = plt.subplots(1, num_RU, constrained_layout=True)

for rho in range(num_RU):
    util_ru_op = np.zeros(len(num_UE))
    util_ru_random = np.zeros(len(num_UE))
    util_ru_avg = np.zeros(len(num_UE))
    for a in [1,3]:
        total_UE = int(multi_num_UE[a] * num_RU)
        dist = distance[a,:,:total_UE,:].reshape((T, total_UE, num_RU))
        
        util_op = np.zeros(T_ref)
        util_random = np.zeros(T_ref)
        util_avg = np.zeros(T_ref)
        
        for t in range(13, T_ref+13):
            if a < 3:
                e_op = np.array(multi_rec_e_op[a,t,0:total_UE,:]) #(T, total_UE, num_RB)
                e_random = np.array(multi_rec_e_random[a,t,0:total_UE,:])
                e_avg =  np.array(multi_rec_e_avg[a,t,0:total_UE,:])
            else:
                e_op = np.array(multi_rec_e_op1[a,t,0:total_UE,:]) #(T, total_UE, num_RB)
                e_random = np.array(multi_rec_e_random[a,t,0:total_UE,:])
                e_avg =  np.array(multi_rec_e_avg[a,t,0:total_UE,:])
                
            
            # calculate UE connect which RU
            user_RU_norm = np.zeros(total_UE, dtype=int)
            for i in range(total_UE):
                temp = np.zeros(num_RU)
                for j in range(num_RU):
                    temp[j] = dist[t, i, j]
                user_RU_norm[i] = np.argmin(temp)

            RU_UE_norm = []
            for r in range(num_RU):
                idx = np.where(user_RU_norm == r)[0]
                RU_UE_norm.append(idx)

            # RANDOM
            e_ran = e_random[RU_UE_norm[rho], :]
            util_random_list = np.any(e_ran, axis=0)
            util_random[t-13] = np.sum(util_random_list) / float(num_RB)

            # AVG
            e_av = e_avg[RU_UE_norm[rho], :]
            util_avg_list = np.any(e_av, axis=0)
            util_avg[t-13] = np.sum(util_avg_list) / float(num_RB)

            # OP
            e_o = e_op[RU_UE_norm[rho], :]
            util_op_list = np.any(e_o, axis=0)
            util_op[t-13] = float(np.sum(util_op_list) / float(num_RB))
        if a == 1:
            idx = 2
        else:
            idx = 5    
        util_ru_op[idx] = np.mean(util_op)
        util_ru_random[idx] = np.mean(np.array(util_random))
        util_ru_avg[idx] = np.mean(np.array(util_avg))
        
    for a in range(4): # len(multi_num_UE)
        total_UE = int(multi_num_UE_sup[a] * num_RU)
        dist = distance_sup[a,:,:total_UE,:].reshape((T, total_UE, num_RU))

        util_op = np.zeros(T_ref)
        util_random = np.zeros(T_ref)
        util_avg = np.zeros(T_ref)
        
        for t in range(T_ref):
            e_op = np.array(multi_rec_e_op_sup[a,t,0:total_UE,:]) #(T, total_UE, num_RB)
            e_random = np.array(multi_rec_e_random_sup[a,t,0:total_UE,:])
            e_avg =  np.array(multi_rec_e_avg_sup[a,t,0:total_UE,:])
                
            # calculate UE connect which RU
            user_RU_norm = np.zeros(total_UE, dtype=int)
            for i in range(total_UE):
                temp = np.zeros(num_RU)
                for j in range(num_RU):
                    temp[j] = dist[t, i, j]
                user_RU_norm[i] = np.argmin(temp)

            RU_UE_norm = []
            for r in range(num_RU):
                idx = np.where(user_RU_norm == r)[0]
                RU_UE_norm.append(idx)

            # RANDOM
            e_ran = e_random[RU_UE_norm[rho], :]
            util_random_list = np.any(e_ran, axis=0)
            util_random[t] = np.sum(util_random_list) / float(num_RB)

            # AVG
            e_av = e_avg[RU_UE_norm[rho], :]
            util_avg_list = np.any(e_av, axis=0)
            util_avg[t] = np.sum(util_avg_list) / float(num_RB)

            # OP
            e_o = e_op[RU_UE_norm[rho], :]
            util_op_list = np.any(e_o, axis=0)
            util_op[t] = float(np.sum(util_op_list) / float(num_RB))
        
        if a == 2 or a == 3:
            idx = a + 1
        else:
            idx = a
        util_ru_op[idx] = np.mean(util_op)
        util_ru_random[idx] = np.mean(np.array(util_random))
        util_ru_avg[idx] = np.mean(np.array(util_avg))

    ax = axes[rho]
    ax.plot(util_ru_random, linewidth=1.5, color='#3480b8', label='Static Allocation', marker='D', markersize=6)
    ax.plot(util_ru_avg, linewidth=1.5, color='#8fbc8f', label='Average Allocation', marker='D', markersize=6)
    ax.plot(util_ru_op, linewidth=1.5, color='#c82423', label='MPC-based Allocation', marker='D', markersize=6)

    ax.set_ylim(0, 1)
    ax.set_xlabel('UE number')
    ax.set_ylabel(f'RB Utilization of RU {rho+1} (%)')
    ax.grid(True)
    ax.set_xticks([a for a in range(len(num_UE))])
    ax.set_xticklabels(xtick)
    
axes[rho].legend(loc='lower right')

plt.show()