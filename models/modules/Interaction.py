import torch
import torch.nn as nn
import torch.nn.functional as F

class InteractionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(InteractionLayer, self).__init__()
        
        self.ffn_V = FFN(input_dim, hidden_dim)
        self.ffn_L = FFN(input_dim, hidden_dim)
        
    def forward(self, V, L, v_mask, l_mask):
        A_LV = torch.bmm(self.ffn_V(V), self.ffn_L(L).permute(0, 2, 1)) / torch.sqrt(torch.tensor(V.size(-1), dtype=torch.float))  # t * n
        A_VV = torch.bmm(self.ffn_V(V), self.ffn_L(V).permute(0, 2, 1)) / torch.sqrt(torch.tensor(V.size(-1), dtype=torch.float))  # t * t


        A = torch.cat([A_LV, A_VV], dim=-1)  # b, t, (n + t)
        V_next = torch.bmm(F.softmax(A, dim=1), torch.cat([self.ffn_L(L), self.ffn_V(V)], dim=1))

        return A, V_next


class InteractionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers= 3):
        super(InteractionEncoder, self).__init__()

        self.layers = nn.ModuleList([InteractionLayer(input_dim, hidden_dim) for _ in range(num_layers)])

    def forward(self, V, L, v_mask, l_mask):
        # V_i = V
        for layer in self.layers:
            A, V = layer(V, L, v_mask, l_mask)

        return V, A


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FFN, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.linear2(self.relu(self.linear1(x)))
        return x



