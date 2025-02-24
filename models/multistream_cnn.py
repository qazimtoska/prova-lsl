import torch
import torch.nn as nn

class MultiStream1DCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_classes: int = 3,
        n_streams: int = 4,
        start_kernel: int = 7,
        stream_depth: int = 2,
        pool_output_size: int = 32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.n_classes   = n_classes
        self.n_streams   = n_streams
        self.start_kernel= start_kernel
        self.stream_depth= stream_depth
        self.pool_output_size = pool_output_size

        # Costruiamo gli stream
        streams = []
        for i in range(n_streams):
            k_size = start_kernel + 2*i  # es. 7,9,11,13
            stream = self._build_stream(in_channels, k_size, stream_depth)
            streams.append(stream)
        self.streams = nn.ModuleList(streams)

        # Pooling adattivo per uniformare la lunghezza temporale a pool_output_size
        self.adapool = nn.AdaptiveMaxPool1d(pool_output_size)

        # Calcoliamo il numero di feature in input al classificatore
        # Ogni stream ha 64 canali in output, dimensione tempo = pool_output_size
        in_feats = 64 * pool_output_size * n_streams
        hidden_dim = 256

        # Classificatore fully-connected
        self.classifier = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, n_classes),
        )

    def _build_stream(self, in_ch, kernel_size, depth):
        """
        Crea una serie di blocchi Conv1d->ReLU->Conv1d->ReLU->MaxPool, ripetuti 'depth' volte.
        """
        layers = []
        c_in = in_ch
        for _ in range(depth):
            c_out = 64
            layers.append(nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.Conv1d(c_out, c_out, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            c_in = c_out
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        x: [B, in_channels, time]
        """
        outs = []
        for stream in self.streams:
            s_out = stream(x)            # => [B,64, time_ridotta]
            s_out = self.adapool(s_out)  # => [B,64,32]
            s_out = s_out.view(s_out.size(0), -1)  # => [B, 64*32]
            outs.append(s_out)
        # Concateniamo le feature di tutti gli stream
        concat = torch.cat(outs, dim=1)   # => [B, 64*32*n_streams]
        # Classificazione finale
        logits = self.classifier(concat)  # => [B, 3]
        return logits