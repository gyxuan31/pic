import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
np.set_printoptions(precision=2, suppress=True)

# load parameters
params = loadmat('multi_UE_tr.mat') # multi_UE_sup1
T = 200# int(params['T'].squeeze())
num_RU = int(params['num_RU'].squeeze())
num_RB = int(params['num_RB'].squeeze())
num_ref = int(params['num_ref'].squeeze())
gamma = params['gamma'].squeeze()
num_setreq = int(params['num_setreq'].squeeze())
B = 2880*1e3
P = params['P'].squeeze()
sigmsqr = 10**((-174-30)/10) * B
eta = float(params['eta'].squeeze())
predicted_len = int(params['predicted_len'].squeeze())
rayleigh_gain = params['rayleigh_gain']
multi_num_UE = params['multi_num_UE'].squeeze()
distance_sup = params['multi_distance_true'].squeeze()
fc = 1 * 1e9 # 2 GHz
loss = (4*np.pi*fc/(3*1e8))**(-2)

num_point = 9 #len(multi_num_UE) # number of UE group

T_ref = T-num_ref
# T_ref = 5

# load output

output1 = loadmat('UE1.mat')
output = loadmat('UE1_nolstm.mat')
# random-op_n, avg-pso_n, fmincon-hun_n
multi_rec_dr_op = output['multi_rec_dr_op'].squeeze() 
multi_rec_dr_pso = output['multi_rec_dr_pso'].squeeze()
multi_rec_dr_hun = output['multi_rec_dr_hun'].squeeze()

multi_rec_dr_op_n = output1['multi_rec_dr_op'].squeeze() # no lstm
multi_rec_dr_pso_n = output1['multi_rec_dr_pso'].squeeze()
multi_rec_dr_hun_n = output1['multi_rec_dr_hun'].squeeze()

multi_rec_e_random_sup = output['multi_rec_e_random'].squeeze()
multi_rec_e_avg_sup = output['multi_rec_e_avg'].squeeze()
multi_rec_e_op_sup = output['multi_rec_e_op'].squeeze()
multi_rec_e_pso_sup = output['multi_rec_e_avg'].squeeze()
multi_rec_e_fmincon_sup = output['multi_rec_e_avg'].squeeze()
multi_rec_e_hun_sup = output['multi_rec_e_hun'].squeeze()

util_op_mean = np.zeros(num_point)
util_random_mean = np.zeros(num_point)
util_avg_mean = np.zeros(num_point)
util_pso_mean = np.zeros(num_point)
util_fmincon_mean = np.zeros(num_point)
util_hun_mean = np.zeros(num_point)

dr_op = np.zeros(num_point)
dr_pso = np.zeros(num_point)
dr_op_n = np.zeros(num_point)
dr_pso_n = np.zeros(num_point)
dr_hun = np.zeros(num_point)
dr_hun_n = np.zeros(num_point)

for a in range(num_point): # total_UE=[6 12 24 30] final[6 12 18(2) 24 30 36(5)]
    total_UE = multi_num_UE[a] * num_RU

    util_random = []
    util_avg = []
    util_op = []
    util_pso = []
    util_fmincon = []
    util_hun = []

    for t in range(T_ref):
        e_op = np.array(multi_rec_e_op_sup[a,t,:total_UE,:]) #(T, total_UE, num_RB)
        e_random = np.array(multi_rec_e_random_sup[a,t,:total_UE,:])
        e_avg =  np.array(multi_rec_e_avg_sup[a,t,:total_UE,:])
        e_pso = np.array(multi_rec_e_pso_sup[a,t,:total_UE,:])
        e_fmin = np.array(multi_rec_e_fmincon_sup[a,t,:total_UE,:])
        e_hun = np.array(multi_rec_e_hun_sup[a,t,:total_UE,:])
        
        # RANDOM
        util_random_list = np.any(e_random, axis=0)  # (num_RB,)
        util_random.append(np.sum(util_random_list) / float(num_RB))

        # AVG
        util_avg_list = np.any(e_avg, axis=0)
        util_avg.append(np.sum(util_avg_list) / float(num_RB))

        # OP
        util_op_list = np.any(e_op, axis=0)
        util_op.append(np.sum(util_op_list) / float(num_RB))
        # print(np.sum(util_op_list))
        
        # PSO
        util_pso_list = np.any(e_pso, axis=0)
        util_pso.append(np.sum(util_pso_list) / float(num_RB))
        
        # fmincon
        util_fmincon_list = np.any(e_fmin, axis=0)
        util_pso.append(np.sum(util_fmincon_list) / float(num_RB))
        
        # HUN
        util_hun_list = np.any(e_hun, axis=0)
        util_hun.append(np.sum(util_hun_list) / float(num_RB))
        
    idx = a
        
    util_op_mean[idx] = np.mean(np.array(util_op))
    util_random_mean[idx] = np.mean(np.array(util_random))
    util_avg_mean[idx] = np.mean(np.array(util_avg))
    util_pso_mean[idx] = np.mean(np.array(util_pso))
    util_fmincon_mean[idx] = np.mean(np.array(util_fmincon))
    util_hun_mean[idx] = np.mean(np.array(util_hun))

    # dr_op[idx] = multi_rec_dr_op_sup[a] / total_UE
    # dr_avg[idx] = multi_rec_dr_avg_sup[a] / total_UE
    # dr_random[idx] = multi_rec_dr_random_sup[a] / total_UE
    
    dr_op_n[idx] = (np.e ** multi_rec_dr_op_n[a])**(1/total_UE)
    dr_pso[idx] = (np.e ** multi_rec_dr_pso[a])**(1/total_UE)
    dr_op[idx] = (np.e ** multi_rec_dr_op[a])**(1/total_UE)
    dr_pso_n[idx] = (np.e ** multi_rec_dr_pso_n[a])**(1/total_UE)
    dr_hun[idx] = (np.e ** multi_rec_dr_hun[a])**(1/total_UE)
    dr_hun_n[idx] = (np.e ** multi_rec_dr_hun_n[a])**(1/total_UE)
    print(dr_op[idx])

print(multi_rec_dr_op_n)
print(multi_rec_dr_hun_n)
# print(dr_avg)
print(dr_hun_n)

# Plot - Geometric Mean of Data Rate
plt.figure()
plt.plot(dr_pso_n, label='PSO-N', marker='D', markersize=5, color='#3480b8', linestyle='--')
plt.plot(dr_pso, label='MPC-PSO', marker='D', markersize=5, color='#3480b8')

plt.plot(dr_hun_n, label='HUN-N', marker='D', markersize=5, color='#c82423', linestyle='--')
plt.plot(dr_hun, label='MPC-HUN', marker='D', markersize=5, color='#c82423') # #ED7D31

plt.plot(dr_op_n, label='GA-N', marker='D', markersize=5, color='#FFC000', linestyle='--')
plt.plot(dr_op, label='MPC-GA', marker='D', markersize=5, color='#FFC000')




plt.xlabel('UE number')
plt.ylabel('Geometric Mean of Data Rate (Mbps)')
xtick = [a*num_RU for a in multi_num_UE[:num_point]]
plt.xticks([a for a in range(num_point)], xtick)
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x * 1e-6:.1f}'))
# for i, y in enumerate(dr_random):
#     plt.text(i, y, f'{y*1e-6:.1f}', ha='center', va='bottom')
# for i, y in enumerate(dr_avg):
#     plt.text(i, y, f'{y*1e-6:.1f}', ha='center', va='bottom')
# for i, y in enumerate(dr_op):
#     plt.text(i, y, f'{y*1e-6:.1f}', ha='center', va='bottom')
plt.legend()
plt.grid()
plt.show()

'''
# Plot - Utilization
# resource efficiency
eff_random = dr_random/util_random_mean
eff_avg = dr_avg/util_avg_mean
eff_op = dr_op/util_op_mean

plt.figure()
plt.plot(eff_random, label='Random', marker='D', markersize=6, color='#3480b8')
plt.plot(eff_avg, label='Average', marker='D', markersize=6, color='#8fbc8f')
plt.plot(eff_op, label='MPC', marker='D', markersize=6, color='#c82423')
plt.xlabel('UE number')
plt.ylabel('Resource Efficiency')

plt.xticks([a for a in range(num_point)], xtick)
plt.legend(loc='upper right')
plt.grid()
plt.show()
'''


# Plot - every RU
fig, axes = plt.subplots(1, num_RU, constrained_layout=True)

for rho in range(num_RU):
    util_ru_op = np.zeros(len(multi_num_UE))
    util_ru_random = np.zeros(len(multi_num_UE))
    util_ru_avg = np.zeros(len(multi_num_UE))
    util_ru_pso = np.zeros(len(multi_num_UE))
    util_ru_fmincon = np.zeros(len(multi_num_UE))
    util_ru_hun = np.zeros(len(multi_num_UE))
    
    ru_op = np.zeros(len(multi_num_UE))
    ru_random = np.zeros(len(multi_num_UE))
    ru_avg = np.zeros(len(multi_num_UE))
    ru_pso = np.zeros(len(multi_num_UE))
    ru_hun = np.zeros(len(multi_num_UE))
        
    for a in range(num_point): # len(multi_num_UE)
        total_UE = int(multi_num_UE[a] * num_RU)
        dist = distance_sup[a,:T,:total_UE,:].reshape((T, total_UE, num_RU))

        util_op = np.zeros(T_ref)
        util_random = np.zeros(T_ref)
        util_avg = np.zeros(T_ref)
        util_pso = np.zeros(T_ref)
        util_fmin = np.zeros(T_ref)
        util_hun = np.zeros(T_ref)
        
        for t in range(T_ref):
            e_op = np.array(multi_rec_e_op_sup[a,t,0:total_UE,:]) #(T, total_UE, num_RB)
            e_random = np.array(multi_rec_e_random_sup[a,t,0:total_UE,:])
            e_avg =  np.array(multi_rec_e_avg_sup[a,t,0:total_UE,:])
            e_pso =  np.array(multi_rec_e_pso_sup[a,t,0:total_UE,:])
            e_fmincon =  np.array(multi_rec_e_fmincon_sup[a,t,0:total_UE,:])
            e_hun = np.array(multi_rec_e_hun_sup[a,t,0:total_UE,:])
            
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
            e_p = e_pso[RU_UE_norm[rho],:]
            util_pso_list = np.any(e_p, axis=0)
            util_pso[t] = float(np.sum(util_pso_list) / float(num_RB))
            
            # fmincon
            e_fminc = e_fmincon[RU_UE_norm[rho],:]
            util_fmincon_list = np.any(e_fminc, axis=0)
            util_fmin[t] = float(np.sum(util_fmincon_list) / float(num_RB))
            
            # HUN
            e_h = e_hun[RU_UE_norm[rho],:]
            util_hun_list = np.any(e_h, axis=0)
            util_hun[t] = float(np.sum(util_hun_list) / float(num_RB))
            
            '''
            # Calculate data rate for every RU
            UE_num = len(RU_UE_norm[rho])
            
            data_rate_op = np.zeros((T_ref, UE_num))
            data_rate_random = np.zeros((T_ref, UE_num))
            data_rate_avg = np.zeros((T_ref, UE_num))
            data_rate_hun = np.zeros((T_ref, UE_num))
            data_rate_pso = np.zeros((T_ref, UE_num))
            for n in range(UE_num):
                for k in range(num_RB):
                    if e_o[n, k] >= 0.5:
                        signal = (
                            P * dist[t + num_ref, n, user_RU_norm[n]] *
                            rayleigh_gain[n, k] * loss
                        )
                        interference = 0.0
                        for others in range(total_UE):
                            if others != n and e_op[others, k] >= 0.5 and user_RU_norm[others] != user_RU_norm[n]:
                                for i in range(num_RU):
                                    interference += (
                                        P * dist[t + num_ref, n, user_RU_norm[i]] *
                                        rayleigh_gain[n, k] * loss
                                    )

                        SINR = signal / (interference + sigmsqr)
                        data_rate_op[t,n] += B * np.log(1 + SINR)
                    if e_ran[n, k] == 1:
                        signal = (
                            P * dist[t + num_ref, n, user_RU_norm[n]] *
                            rayleigh_gain[n, k] * loss
                        )
                        interference = 0.0
                        for others in range(total_UE):
                            if others != n and e_random[others, k] == 1 and user_RU_norm[others] != user_RU_norm[n]:
                                for i in range(num_RU):
                                    interference += (
                                        P * dist[t + num_ref, n, user_RU_norm[i]] *
                                        rayleigh_gain[n, k] * loss
                                    )
                        SINR = signal / (interference + sigmsqr)
                        data_rate_random[t,n] += B * np.log(1 + SINR)
                    if e_avg[n, k] == 1:
                        signal = (
                            P * dist[t + num_ref, n, user_RU_norm[n]] *
                            rayleigh_gain[n, k] * loss
                        )
                        interference = 0.0
                        for others in range(total_UE):
                            if others != n and e_avg[others, k] == 1 and user_RU_norm[others] != user_RU_norm[n]:
                                for i in range(num_RU):
                                    interference += (
                                        P * dist[t + num_ref, n, user_RU_norm[i]] *
                                        rayleigh_gain[n, k] * loss
                                    )
                        SINR = signal / (interference + sigmsqr)
                        data_rate_avg[t,n] += B * np.log(1 + SINR)
                    if e_p[n, k] == 1:
                        signal = (
                            P * dist[t + num_ref, n, user_RU_norm[n]] *
                            rayleigh_gain[n, k] * loss
                        )
                        interference = 0.0
                        for others in range(total_UE):
                            if others != n and e_avg[others, k] == 1 and user_RU_norm[others] != user_RU_norm[n]:
                                for i in range(num_RU):
                                    interference += (
                                        P * dist[t + num_ref, n, user_RU_norm[i]] *
                                        rayleigh_gain[n, k] * loss
                                    )
                        SINR = signal / (interference + sigmsqr)
                        data_rate_avg[t,n] += B * np.log(1 + SINR)
                        
                    if e_h[n, k] == 1:
                        signal = (
                            P * dist[t + num_ref, n, user_RU_norm[n]] *
                            rayleigh_gain[n, k] * loss
                        )
                        interference = 0.0
                        for others in range(total_UE):
                            if others != n and e_avg[others, k] == 1 and user_RU_norm[others] != user_RU_norm[n]:
                                for i in range(num_RU):
                                    interference += (
                                        P * dist[t + num_ref, n, user_RU_norm[i]] *
                                        rayleigh_gain[n, k] * loss
                                    )
                        SINR = signal / (interference + sigmsqr)
                        data_rate_avg[t,n] += B * np.log(1 + SINR)

        ru_op[a] = np.exp(np.mean(np.log(1+np.mean(data_rate_op, axis=0))))
        ru_avg[a] = np.exp(np.mean(np.log(1+np.mean(data_rate_avg, axis=0))))
        ru_random[a] = np.exp(np.mean(np.log(1+np.mean(data_rate_random, axis=0))))
        ru_hun[a] = np.exp(np.mean(np.log(1+np.mean(data_rate_hun, axis=0))))
        ru_pso[a] = np.exp(np.mean(np.log(1+np.mean(data_rate_pso, axis=0))))
        
        # ru_op[a] = np.prod(np.mean(data_rate_op, axis=0)) ** (1 / UE_num)
        # ru_avg[a] = np.prod(np.mean(data_rate_avg, axis=0)) ** (1 / UE_num)
        # ru_random[a] = np.prod(np.mean(data_rate_random, axis=0)) ** (1 / UE_num)
        # ru_hun[a] = np.prod(np.mean(data_rate_hun, axis=0)) ** (1 / UE_num)
        # ru_pso[a] =np.prod(np.mean(data_rate_pso, axis=0)) ** (1 / UE_num)
            

        idx = a
        util_ru_op[idx] = ru_op[a] / np.mean(util_op)
        util_ru_random[idx] = ru_random[a] / np.mean(np.array(util_random))
        util_ru_avg[idx] = ru_avg[a] / np.mean(np.array(util_avg))
        util_ru_pso[idx] = ru_pso[a] / np.mean(np.array(util_pso))
        # util_ru_fmincon[idx] = np.mean(np.array(util_fmin))
        util_ru_hun[idx] = ru_hun[a] / np.mean(np.array(util_hun))
        '''
        idx = a
        util_ru_op[idx] = np.mean(util_op)
        util_ru_random[idx] = np.mean(np.array(util_random))
        util_ru_avg[idx] = np.mean(np.array(util_avg))
        util_ru_pso[idx] = np.mean(np.array(util_pso))
        # util_ru_fmincon[idx] = np.mean(np.array(util_fmin))
        util_ru_hun[idx] = np.mean(np.array(util_hun))

    ax = axes[rho]
    ax.plot(util_ru_random, linewidth=1.5, color='#3480b8', label='Static Allocation', marker='D', markersize=5)
    ax.plot(util_ru_avg, linewidth=1.5, color='#8fbc8f', label='Average Allocation', marker='D', markersize=5)
    ax.plot(util_ru_op, linewidth=1.5, color='#c82423', label='MPC-GA Allocation', marker='D', markersize=5)
    ax.plot(util_ru_pso, linewidth=1.5, color='gray', label='MPC-PSO', marker='D', markersize=5)
    # ax.plot(util_ru_fmincon, linewidth=1.5, color='#FFC000', label='MPC-fmincon', marker='D', markersize=5)
    ax.plot(util_ru_hun, linewidth=1.5, color='#FF99CC', label='MPC-HUN', marker='D', markersize=5)

    # ax.set_ylim(0, 1)
    ax.set_xlabel('UE number')
    ax.set_ylabel(f'RB Utilization of RU {rho+1} (%)')
    ax.grid(True)
    ax.set_xticks([a for a in range(num_point)])
    ax.set_xticklabels(xtick)
    
axes[rho].legend(loc='lower right')

plt.show()