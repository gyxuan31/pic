import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
np.set_printoptions(precision=2, suppress=True)

# load parameters
params = loadmat('multi_distance1.mat') # 1 5；2000 6
T = int(params['T'].squeeze())

num_RU = int(params['num_RU'].squeeze())
num_RB = int(params['num_RB'].squeeze())
num_ref = int(params['num_ref'].squeeze())
gamma = params['gamma'].squeeze()
B = float(params['B'].squeeze())
P = params['P'].squeeze()
sigmsqr = params['sigmsqr'].squeeze()
eta = float(params['eta'].squeeze())
rayleigh_gain = params['rayleigh_gain']
total_UE = int(params['total_UE'].squeeze())
multi_distance = params['multi_distance_true'].squeeze()

num_point = params['num_point'].squeeze()
loss = (4*np.pi*1e9/(3*1e8))**(-eta)

T_ref = T-num_ref
T_ref = 20
# load output

output1 = loadmat('multiDis_output5.mat')
multi_rec_dr_random_sup = output1['multi_rec_dr_random'].squeeze()
multi_rec_dr_pso_sup = output1['multi_rec_dr_pso'].squeeze()
multi_rec_dr_avg_sup = output1['multi_rec_dr_avg'].squeeze()
multi_rec_dr_op_sup = output1['multi_rec_dr_op'].squeeze()
multi_rec_e_random_sup = output1['multi_rec_e_random'].squeeze()
multi_rec_e_pso_sup = output1['multi_rec_e_pso'].squeeze()
multi_rec_e_avg_sup = output1['multi_rec_e_avg'].squeeze()
multi_rec_e_op_sup = output1['multi_rec_e_op'].squeeze()
multi_mean_avg = output1['multi_mean_avg'].squeeze()
multi_mean_op = output1['multi_mean_op'].squeeze()
multi_mean_random = output1['multi_mean_random'].squeeze()
multi_mean_pso = output1['multi_mean_pso'].squeeze()

xtick = [1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00]
plt.figure()
plt.plot(multi_mean_random, label='Random', marker='D', markersize=6, color='#3480b8') 
plt.plot(multi_mean_avg, label='Average', marker='D', markersize=6, color='#8fbc8f')
plt.plot(multi_mean_op, label='MPC', marker='D', markersize=6, color='#c82423')
plt.plot(multi_mean_pso, label='pso', marker='D', markersize=6, color='gray')
plt.xlabel('Standard Deviation of the Distance from UE to Serving RU (km)')
plt.ylabel('Mean of Data Rate')
plt.xticks([a for a in range(0,num_point,2)], xtick)
plt.legend(loc='lower right')
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x * 1e-6:.1f}'))
plt.grid()
plt.show()

util_op_mean = np.zeros(num_point)
util_random_mean = np.zeros(num_point)
util_avg_mean = np.zeros(num_point)
util_pso_mean = np.zeros(num_point)
dr_random = np.zeros(num_point)
dr_avg = np.zeros(num_point)
dr_op = np.zeros(num_point)
dr_pso = np.zeros(num_point)

# for a in range(num_point):
#     print(a)
#     mean_op = []
#     mean_random = []
#     mean_avg = []
#     distance = np.squeeze(multi_distance[a, :, :, :])
#     for t in range(T_ref):
#         e_op = np.array(multi_rec_e_op_sup[a,t,:,:]) #(T, total_UE, num_RB)
#         e_random = np.array(multi_rec_e_random_sup[a,t,:,:])
#         e_avg =  np.array(multi_rec_e_avg_sup[a,t,:,:])
#         data_rate_op = np.zeros((T_ref, total_UE))
#         data_rate_random = np.zeros((T_ref, total_UE))
#         data_rate_avg = np.zeros((T_ref, total_UE))
        
#         user_RU_norm = np.zeros(total_UE, dtype=int)

#         for i in range(total_UE):
#             temp = np.zeros(num_RU)
#             for j in range(num_RU):
#                 temp[j] = distance[t + num_ref, i, j]
#             user_RU_norm[i] = np.argmin(temp)  # Python 中返回的是 0-based index
#         RU_UE_norm = [[] for _ in range(num_RU)]
#         for r in range(num_RU):
#             idx = np.where(user_RU_norm == r)[0]  # 返回满足条件的索引数组
#             RU_UE_norm[r] = idx.tolist()  # 存储为列表
        
#         for n in range(total_UE):
#             for k in range(num_RB):
#                 if e_op[n, k] >= 0.5:
#                     signal = (
#                         P * distance[t + num_ref, n, user_RU_norm[n]] *
#                         rayleigh_gain[n, k] * loss
#                     )

#                     interference = 0.0
#                     for others in range(total_UE):
#                         if others != n and e_op[others, k] >= 0.5 and user_RU_norm[others] != user_RU_norm[n]:
#                             for i in range(num_RU):
#                                 interference += (
#                                     P * distance[t + num_ref, n, user_RU_norm[i]] *
#                                     rayleigh_gain[n, k] * loss
#                                 )

#                     SINR = signal / (interference + sigmsqr)
#                     data_rate_op[t,n] += B * np.log(1 + SINR)
#                 if e_random[n, k] == 1:
#                     signal = (
#                         P * distance[t + num_ref, n, user_RU_norm[n]] *
#                         rayleigh_gain[n, k] * loss
#                     )
#                     interference = 0.0
#                     for others in range(total_UE):
#                         if others != n and e_random[others, k] == 1 and user_RU_norm[others] != user_RU_norm[n]:
#                             for i in range(num_RU):
#                                 interference += (
#                                     P * distance[t + num_ref, n, user_RU_norm[i]] *
#                                     rayleigh_gain[n, k] * loss
#                                 )
#                     SINR = signal / (interference + sigmsqr)
#                     data_rate_random[t,n] += B * np.log(1 + SINR)
#                 if e_avg[n, k] == 1:
#                     signal = (
#                         P * distance[t + num_ref, n, user_RU_norm[n]] *
#                         rayleigh_gain[n, k] * loss
#                     )
#                     interference = 0.0
#                     for others in range(total_UE):
#                         if others != n and e_avg[others, k] == 1 and user_RU_norm[others] != user_RU_norm[n]:
#                             for i in range(num_RU):
#                                 interference += (
#                                     P * distance[t + num_ref, n, user_RU_norm[i]] *
#                                     rayleigh_gain[n, k] * loss
#                                 )
#                     SINR = signal / (interference + sigmsqr)
#                     data_rate_avg[t,n] += B * np.log(1 + SINR)
#     avg_a = np.mean(data_rate_avg, axis=0)
#     avg_b = np.sum(np.mean(data_rate_avg, axis=0))/total_UE
#     random_a = np.mean(data_rate_random, axis=0)
#     random_b = np.sum(np.mean(data_rate_random, axis=0))/total_UE
#     op_a = np.mean(data_rate_op, axis=0)
#     op_b = np.sum(np.mean(data_rate_op, axis=0))/total_UE
#     mean_random.append(random_b)
#     mean_op.append(op_b)
#     mean_avg.append(avg_b)


for a in range(num_point):
    util_random = []
    util_avg = []
    util_op = []
    util_pso = []

    for t in range(T_ref):
        e_op = np.array(multi_rec_e_op_sup[a,t,:,:]) #(T, total_UE, num_RB)
        e_random = np.array(multi_rec_e_random_sup[a,t,:,:])
        e_avg =  np.array(multi_rec_e_avg_sup[a,t,:,:])
        e_pso = np.array(multi_rec_e_pso_sup[a,t,:,:])
        # print(e_op)
        # RANDOM
        util_random_list = np.any(e_random, axis=0)  # (num_RB,)
        util_random.append(np.sum(util_random_list) / float(num_RB))

        # AVG
        util_avg_list = np.any(e_avg, axis=0)
        util_avg.append(np.sum(util_avg_list) / float(num_RB))

        # OP
        # temp = np.sum(e_op, axis=1) # test-RB num UE get
        util_op_list = np.any(e_op, axis=0)
        util_op.append(np.sum(util_op_list) / float(num_RB))
        # print(np.sum(util_op_list))
        
        # PSO
        util_pso_list = np.any(e_pso, axis=0)
        util_pso.append(np.sum(util_pso_list) / float(num_RB))
        
    idx = a
        
    util_op_mean[idx] = np.mean(np.array(util_op))
    util_random_mean[idx] = np.mean(np.array(util_random))
    util_avg_mean[idx] = np.mean(np.array(util_avg))
    util_pso_mean[idx] = np.mean(np.array(util_pso))

    dr_op[idx] = (np.e ** multi_rec_dr_op_sup[a])**(1/total_UE)
    dr_avg[idx] = (np.e ** multi_rec_dr_avg_sup[a])**(1/total_UE)
    dr_random[idx] = (np.e ** multi_rec_dr_random_sup[a])**(1/total_UE)
    dr_pso[idx] = (np.e ** multi_rec_dr_pso_sup[a])**(1/total_UE)

# print(dr_op)
# print(dr_avg)
print(multi_rec_dr_op_sup)
# Plot - Geometric Mean of Data Rate
plt.figure()
plt.plot(dr_random, label='Random', marker='D', markersize=5, color='#3480b8') 
plt.plot(dr_avg, label='Average', marker='D', markersize=5, color='#8fbc8f')
plt.plot(dr_pso, label='PSO', marker='D', markersize=5, color='gray')
plt.plot(dr_op, label='MPC', marker='D', markersize=5, color='#c82423')
plt.xlabel('Standard Deviation of the Distance from UE to Serving RU (km)')
plt.ylabel('Geometric Mean of Data Rate (Mbps)')
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x * 1e-6:.1f}'))
plt.xticks([a for a in range(0,num_point,2)], xtick)
plt.legend(loc='upper right')
plt.grid()

# random=[]
# avg=[]
# op=[]
# pso=[]
# for i in range(11):
#     random.append(dr_random[i*2])
#     avg.append(dr_avg[i*2])
#     op.append(dr_op[i*2])
#     pso.append(dr_op[i*2])
# plt.figure()
# plt.plot(random, label='Random', marker='D', markersize=6, color='#3480b8') 
# plt.plot(avg, label='Average', marker='D', markersize=6, color='#8fbc8f')
# plt.plot(op, label='MPC', marker='D', markersize=6, color='#c82423')
# plt.xlabel('Standard Deviation of the Distance from UE to Serving RU (km)')
# plt.ylabel('Geometric Mean of Data Rate (Mbps)')
# ax = plt.gca()
# ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x * 1e-6:.1f}'))
# # tick = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
# # plt.xticks([a for a in range(6)], tick)
# plt.legend(loc='upper right')
# plt.grid()

# Plot - Utilization
# resource efficiency
eff_random = dr_random/util_random_mean
eff_avg = dr_avg/util_avg_mean
eff_op = dr_op/util_op_mean

# plt.figure()
# plt.plot(eff_random, label='Random', marker='D', markersize=6, color='#3480b8')
# plt.plot(eff_avg, label='Average', marker='D', markersize=6, color='#8fbc8f')
# plt.plot(eff_op, label='MPC', marker='D', markersize=6, color='#c82423')
# plt.xlabel('Coverage Area of UE')
# plt.ylabel('Resource Efficiency')

# plt.xticks([a for a in range(num_point)], xtick)
# plt.legend(loc='lower right') # loc='lower right'
# plt.grid()
# plt.show()



# Plot - every RU
fig, axes = plt.subplots(1, num_RU, constrained_layout=True)

for rho in range(num_RU):
    util_ru_op = np.zeros(num_point)
    util_ru_random = np.zeros(num_point)
    util_ru_avg = np.zeros(num_point)
    util_ru_pso = np.zeros(num_point)
        
    for a in range(num_point): # len(multi_num_UE)
        dist = multi_distance[a,:,:total_UE,:].reshape((T, total_UE, num_RU))

        util_op = np.zeros(T_ref)
        util_random = np.zeros(T_ref)
        util_avg = np.zeros(T_ref)
        util_pso = np.zeros(T_ref)
        
        for t in range(T_ref):
            e_op = np.array(multi_rec_e_op_sup[a,t,0:total_UE,:]) #(T, total_UE, num_RB)
            e_random = np.array(multi_rec_e_random_sup[a,t,0:total_UE,:])
            e_avg =  np.array(multi_rec_e_avg_sup[a,t,0:total_UE,:])
            e_pso =  np.array(multi_rec_e_pso_sup[a,t,0:total_UE,:])
                
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
            
            # PSO
            e_ps = e_pso[RU_UE_norm[rho], :]
            util_pso_list = np.any(e_ps, axis=0)
            util_pso[t] = float(np.sum(util_pso_list) / float(num_RB))

        idx = a
        util_ru_op[idx] = np.mean(util_op)
        util_ru_random[idx] = np.mean(np.array(util_random))
        util_ru_avg[idx] = np.mean(np.array(util_avg))
        util_ru_pso[idx] = np.mean(np.array(util_pso))

    ax = axes[rho]
    ax.plot(util_ru_random, linewidth=1.5, color='#3480b8', label='Static Allocation', marker='D', markersize=4)
    ax.plot(util_ru_avg, linewidth=1.5, color='#8fbc8f', label='Average Allocation', marker='D', markersize=4)
    ax.plot(util_ru_pso, linewidth=1.5, color='gray', label='pso', marker='D', markersize=4)
    ax.plot(util_ru_op, linewidth=1.5, color='#c82423', label='MPC-based Allocation', marker='D', markersize=4)

    ax.set_ylim(0, 1)
    
    ax.set_ylabel(f'RB Utilization of RU {rho+1} (%)')
    ax.grid(True)
    ax.set_xticks([a for a in range(0,num_point,2)])
    ax.set_xticklabels(xtick)
axes[1].set_xlabel('Standard Deviation of the Distance from UE to Serving RU (km)')
axes[rho].legend(loc='lower right')

plt.show()