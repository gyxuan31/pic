import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from scipy.io import savemat
np.random.seed(1)

num_ref = 5
predicted_len = 3
num_RU = 3
num_RB = 25 # num RB/RU

T = 55

gamma = 3
num_setreq = 3
B = 2880*1e3 # 12*300 kHz
fc = 1 * 1e9 # 2 GHz
P = 0.2 # W
sigmsqr = 4*1e-12 
eta = 2

multi_num_UE = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# multi_num_UE = [2, 4, 8, 10] # UERU, under one RU 6 12 18 24 30 36
# distance_true.shape(T, total_UE, num_RU)
# prediction.shape(T-num_ref, predicted_len, total_UE, num_RU)
multi_distance_true = np.zeros((len(multi_num_UE), T, multi_num_UE[-1]*num_RU, num_RU),dtype=float) # shape(len(multi_num_UE), T, multi_num_UE[i], num_RU)
multi_prediction = np.zeros((len(multi_num_UE), T-num_ref, multi_num_UE[-1]*num_RU, num_RU),dtype=float) # shape(len(multi_num_UE), T, predicted_len, multi_num_UE[i], num_RU)

for a in range(len(multi_num_UE)):
    UERU = multi_num_UE[a] # num of UE under every RU
    total_UE = UERU * num_RU
    
    # Rayleigh fading
    X = np.random.randn(total_UE, num_RB) # real
    Y = np.random.randn(total_UE, num_RB) # img
    H = (X + 1j * Y) / np.sqrt(2)   # H.shape = (total_UE, num_RB)
    # rayleigh_gain = np.abs(H)**2     # |h|^2
    rayleigh_gain = np.ones((total_UE, num_RB))

    # Location
    locrux = [-1.732*1000, 0, 1.732*1000]
    locruy = [-1*1000, 2*1000, -1*1000]
    locux = np.random.randn(total_UE) * 2000 - 1000 # * multi_distance[a] - multi_distance[a]/2
    locuy = np.random.randn(total_UE) * 2000 - 1000 # * multi_distance[a] - multi_distance[a]/2
    # plt.scatter(locrux,locruy)
    # plt.scatter(locux,locuy)
    # plt.show()
    trajectory_x = np.zeros((T, total_UE)) # shape(sequence_length, total_UE)
    trajectory_y = np.zeros((T, total_UE))

    # Trajectory
    trajectory_x[0] = locux
    trajectory_y[0] = locuy
    for t in range(T):
        for i in range(total_UE):
            trajectory_x[t, i] = np.random.normal(loc=0, scale=500)
            trajectory_y[t, i] = np.random.normal(loc=0, scale=500)
                     
    # Plot trajectory
    for i in range(total_UE):
        plt.plot(trajectory_x.T[i], trajectory_y.T[i])
    plt.scatter(locrux, locruy)
    plt.title('UE Trajectory')
    plt.grid()
    # plt.show()

    # Distance
    distance_true = np.zeros((T, total_UE, num_RU))
    for t in range(T):
        for i in range(total_UE):
            for j in range(num_RU):
                dis = np.sqrt((trajectory_x[t, i] - locrux[j]) ** 2 + (trajectory_y[t, i] - locruy[j]) ** 2)
                distance_true[t, i, j] = dis

    
    # Train
    X = []
    Y = []
    for i in range(T - num_ref - predicted_len):
        x_seq = distance_true[i:i + num_ref, :, :]  # (num_ref, total_UE, num_RU)
        y_seq = distance_true[i+num_ref:i + num_ref + predicted_len, :, :]  # (predicted_len, total_UE, num_RU)
        X.append(x_seq)
        Y.append(y_seq)

    X = np.array(X)  # (samples, num_ref, total_UE, num_RU)
    Y = np.array(Y)  # (samples, predicted_len, total_UE, num_RU)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    samples = X.shape[0]
    X_flat = X.reshape(-1, total_UE * num_RU) # shape(-1, total_UE*num_RU) - normalize
    X_scaled = scaler_x.fit_transform(X_flat).reshape(samples, num_ref, total_UE, num_RU)

    Y_flat = Y.reshape(-1, total_UE * num_RU)
    Y_scaled = scaler_y.fit_transform(Y_flat).reshape(samples, predicted_len, total_UE, num_RU)

    X_train = torch.tensor(X_scaled, dtype=torch.float32)
    Y_train = torch.tensor(Y_scaled, dtype=torch.float32)

    class LSTMModel(nn.Module):
        def __init__(self, total_UE, num_RU, num_ref, predicted_len):
            super().__init__()
            hidden_dim = 64
            self.lstm = nn.LSTM(input_size=total_UE*num_RU, hidden_size=hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, total_UE * num_RU * predicted_len)
            self.total_UE = total_UE
            self.num_RU = num_RU
            self.num_ref = num_ref
            self.predicted_len = predicted_len

        def forward(self, x):
            batch_size = x.size(0)
            x = x.view(batch_size, self.num_ref, -1)  # (batch, num_ref, total_UE*num_RU)
            lstm_out, _ = self.lstm(x)               # (batch, num_ref, hidden_dim)
            last_out = lstm_out[:, -1, :]            # (batch, hidden_dim)
            y = self.fc(last_out)                    # (batch, total_UE*num_RU*predicted_len)
            y = y.view(batch_size, self.predicted_len, self.total_UE, self.num_RU)
            return y


    model = LSTMModel(total_UE, num_RU, num_ref, predicted_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(300):
        model.train()
        output = model(X_train)
        loss = loss_fn(output, Y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss = {loss.item():.4f}")


    model.eval()

    # Predict
    prediction = [] # shape(T-num_ref, predicted_len, total_UE, num_RU)
    prediction_avg = [] # shape(T-num_ref, total_UE, num_RU)
    for t in range(T-num_ref):
        ref = distance_true[t:t+num_ref, :, :].reshape(1, num_ref, total_UE, num_RU)
        ref_scaled = scaler_x.transform(ref.reshape(-1, total_UE * num_RU)).reshape(1, num_ref, total_UE, num_RU)
        ref_tensor = torch.tensor(ref_scaled, dtype=torch.float32)

        with torch.no_grad():
            pred_scaled = model(ref_tensor).numpy()  # (1, predicted_len, total_UE, num_RU)
            pred_flat = pred_scaled.reshape(-1, total_UE * num_RU)
            pred = scaler_y.inverse_transform(pred_flat).reshape(predicted_len, total_UE, num_RU)
            prediction.append(pred)
        for u in range(total_UE):
            for i in range(num_RU):
                multi_prediction[a,t,u,i] = 0.85*prediction[t][0][u][i]+0.1*prediction[t][1][u][i]+0.05*prediction[t][2][u][i] 
                                                        # shape(len(multi_num_UE), T, predicted_len, multi_num_UE[i], num_RU)
    multi_distance_true[a,:,:total_UE,:] = distance_true # shape(len(multi_num_UE), T, multi_num_UE[i], num_RU)
 
loss = (4*np.pi*fc/(3*1e8))**(-2)
multi_distance_true = multi_distance_true
multi_prediction = multi_prediction

savemat('multi_UE_sup1.mat', {
    'T': T,
    'num_RU': num_RU,
    'multi_num_UE': multi_num_UE,
    'num_RB': num_RB,
    'num_ref': num_ref,
    'gamma': gamma,
    'num_setreq': num_setreq,
    'B': B,
    'P': P,
    'sigmsqr': sigmsqr,
    'eta': eta,
    'predicted_len': predicted_len,
    'rayleigh_gain': rayleigh_gain,
    'loss': loss,
    
    'multi_distance_true': multi_distance_true,
    'multi_prediction': multi_prediction
    })

print(multi_distance_true)
print(multi_prediction)

plt.figure() # prediction & true distance
pred_array = np.array(prediction)  # shape: (T - num_ref, predicted_len, total_UE, num_RU)
pred_distance = []
for i in range(pred_array.shape[0] - predicted_len + 1):
    pred_distance.append(pred_array[i, 0, 1,1])

plt.plot(pred_distance, 'b', label='Predicted Distance')
true_distance = distance_true[num_ref:num_ref + (T - num_ref - predicted_len + 1),1,1]  # (T - num_ref - predicted_len + 1,)
plt.plot(true_distance.flatten(), 'r--', label='True Distance')

plt.xlabel("Time Step")
plt.ylabel("Distance")
plt.xticks(np.arange(0, len(true_distance)+1, 3))
plt.title("True vs Predicted Distance")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
