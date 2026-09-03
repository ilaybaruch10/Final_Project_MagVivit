import torch
import torch.nn as nn

class SharedMacroTransformer(nn.Module):
    DOMAIN_RR = 0
    DOMAIN_TEMP = 1
    COND_EXPOSED = 0
    COND_BLANKET = 1

    def __init__(self, d_model=512, n_heads=8, depth=6, dropout=0.1,
                 ff_mult=4, max_len=700, use_domain_token=False):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult*d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.use_domain_token = use_domain_token
        # Spec section 17: no CLS token. RR is predicted densely from the
        # contextualized frame tokens, not from one pooled global vector, so a CLS
        # token has nothing to do in the RR-only configuration.
        self.domain_embed = nn.Embedding(2, d_model)
        self.cond_embed = nn.Embedding(2, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))

    def make_seq(self, z, domain_id: int, cond_id):
        """Build [COND] z1..zT, with positional embeddings on the frame tokens only."""
        B, T, D = z.shape
        # Spec section 19: COND is not a temporal frame, so it must not receive a
        # temporal position. Adding pos_embed to the whole sequence (the previous
        # behaviour) both shifted COND by P0 and offset every frame by one slot,
        # so z_t carried P_{t+1}.
        z = z + self.pos_embed[:, :T, :]
        cond = self.cond_embed(cond_id.view(B, 1).to(z.device))
        if self.use_domain_token:
            dom_ids = torch.full((B, 1), int(domain_id), device=z.device, dtype=torch.long)
            dom = self.domain_embed(dom_ids)
            return torch.cat([dom, cond, z], dim=1)      # (B,T+2,D)
        return torch.cat([cond, z], dim=1)               # (B,T+1,D) -- spec section 20

    def frame_start_index(self):
        """Index of the first frame token, i.e. how many leading tokens to drop.

        Spec section 23: the contextualized COND output is discarded before the
        dense RR head, leaving exactly T frame vectors.
        """
        return 2 if self.use_domain_token else 1

    def forward_tokens(self, z, domain_id: int, cond_id):
        return self.encoder(self.make_seq(z, domain_id, cond_id))

    def forward_once(self, z, domain_id: int, cond_id):
        """Deprecated: pooled single-vector output for the disabled Temperature path.

        This used to return the CLS representation. With CLS removed (spec section
        17) index 0 is the COND token, which is a class embedding and not a summary
        of the clip -- returning it would be silently wrong. Spec section 39 requires
        the Temperature head to become dense (B,200) before it is re-enabled, so this
        fails loudly instead of guessing.
        """
        raise NotImplementedError(
            "forward_once() relied on the CLS token, removed per spec section 17. "
            "Per spec section 39 the Temperature path must produce a dense (B,T) "
            "output from the frame tokens before include_temp_path is re-enabled."
        )
