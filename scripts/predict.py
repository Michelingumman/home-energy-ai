class EnergyDemandPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim,
            num_layers,
            batch_first=True
        )
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 24)  # 24-hour prediction
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.dense(lstm_out[:, -1, :])
        return predictions