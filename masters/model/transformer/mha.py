import einops
from torch import LongTensor, Tensor, nn

from ..utils import pe_from_name


class MHA(nn.Module):
    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        output_dropout: float = 0.0,
        relative_pe: str | None = None,
        relative_pe_kwargs: dict = {},
        relative_pe_shared: bool = False,
    ):
        super(MHA, self).__init__()

        assert embed_size % num_heads == 0, ":("
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.relative_pe_shared = relative_pe_shared

        self.W_q = nn.Linear(embed_size, embed_size)
        self.W_k = nn.Linear(embed_size, embed_size)
        self.W_v = nn.Linear(embed_size, embed_size)
        self.W_o = nn.Linear(embed_size, embed_size)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.output_dropout = nn.Dropout(output_dropout)

        if relative_pe is not None:
            if self.relative_pe_shared:
                self.relative_pe = pe_from_name("relative", relative_pe)
                # RPE is shared, don't instantiate it here
            else:
                relative_pe_kwargs.setdefault("embed_size", self.head_dim)
                self.relative_pe = pe_from_name("relative", relative_pe)(
                    **relative_pe_kwargs
                )
        else:
            self.relative_pe = None

        self.scale = self.head_dim**-0.5

    def forward(
        self,
        x: Tensor,
        mask: LongTensor | None = None,
        rpe_mtx: Tensor | None = None,
    ) -> Tensor:
        Q = einops.rearrange(self.W_q(x), "b n (h d) -> b h n d", h=self.num_heads)
        K = einops.rearrange(self.W_k(x), "b n (h d) -> b h n d", h=self.num_heads)
        V = einops.rearrange(self.W_v(x), "b n (h d) -> b h n d", h=self.num_heads)
        A = einops.einsum(Q, K, "b h i d, b h j d -> b h i j") * self.scale

        if self.relative_pe is not None:
            if self.relative_pe_shared:
                assert rpe_mtx is not None, "rpe_mtx needed for shared RPE"
                P = rpe_mtx
            else:
                assert rpe_mtx is None, "rpe_mtx not needed for per-layer RPE"
                P = self.relative_pe(x)

            if self.relative_pe.coupled:
                A += einops.einsum(
                    Q, P, f"b h i d, {self.relative_pe.shape} -> b h i j"
                )
            else:
                A += P  # [h i j] or [i j]

        # mask: [b j] -> [b 1 1 j]
        if mask is not None:
            attention_mask = mask[:, None, None, :]
            A = A.masked_fill(attention_mask == 0, float("-inf"))

        S = self.attn_dropout(A.softmax(dim=-1))

        C = einops.einsum(S, V, "b h i j, b h j d -> b h i d")

        O = self.W_o(einops.rearrange(C, "b h n d -> b n (h d)"))

        return self.output_dropout(O)
