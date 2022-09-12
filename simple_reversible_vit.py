from types import SimpleNamespace

import yahp as hp
import torch
from torch import nn
from torch.autograd import Function as Function
from einops import rearrange
from einops.layers.torch import Rearrange

# A one-file implementation of the reversible VIT architecture
# Lacking:
# - Stochastic Depth (for now)
# - Dropout (never used)
# - Positional Embeddings (adding now)

def norm(dim: int):
    return nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)


def mlp(in_features: int, hidden_features: int, out_features: int):
    # If you need dropout, just get more data ...
    # None of the ViT configs (that I checked) in PySlowFast
    # use dropout
    # (Possibly b/c they are using stochastic depth instead ...)
    return nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
    )


def mlp_block(dim: int, mlp_ratio: int):
    return nn.Sequential(
            norm(dim),
            mlp(in_features=dim, hidden_features=dim * mlp_ratio, out_features=dim),
    )


# Taken from lucidrain's ViT repository
class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        # TODO: The original paper says sqrt(d_k)
        # but FBAI + lucidrains do something else
        self.scale = head_dim ** -0.5

        self.to_probabilities = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

    def forward(self, x):
        b, n_patches, dim = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        assert qkv[0].shape == x.shape

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        assert q.shape == (b, self.heads, n_patches, dim // self.heads)

        attn_matrix = (q @ k.transpose(-1, -2)) * self.scale

        assert attn_matrix.shape[-1] == attn_matrix.shape[-2]
        assert attn_matrix.shape[-1] == n_patches

        probs = self.to_probabilities(attn_matrix)

        reaveraged_values = probs @ v
        reaveraged_values = rearrange(reaveraged_values, "b h n d -> b n (h d)")

        return reaveraged_values


def attention_block(*, dim: int, heads: int):
    return nn.Sequential(norm(dim), Attention(dim, heads))


class ReversibleBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        mlp_ratio,
        drop_path_rate: float,
    ):
        super().__init__()
        self.drop_path_rate = drop_path_rate
	# No residual connections on purpose:
	# the paper says the two-stream architecture
	# has builtin skip connections
        self.F = attention_block(
	    # dim should be divisible by 2
            dim=dim // 2,
            heads=heads,
        )
        self.G = mlp_block(
            dim=dim // 2,
            mlp_ratio=mlp_ratio,
        )
        # self.seeds = {}

    # def seed_cuda(self, key):
    #     """
    #     Fix seeds to allow for stochastic elements such as
    #     dropout to be reproduced exactly in activation
    #     recomputation in the backward pass.
    #     """

    #     # randomize seeds
    #     # use cuda generator if available
    #     if (
    #         hasattr(torch.cuda, "default_generators")
    #         and len(torch.cuda.default_generators) > 0
    #     ):
    #         # GPU
    #         device_idx = torch.cuda.current_device()
    #         seed = torch.cuda.default_generators[device_idx].seed()
    #     else:
    #         # CPU
    #         seed = int(torch.seed() % sys.maxsize)

    #     self.seeds[key] = seed
    #     torch.manual_seed(self.seeds[key])

    def forward(self, X_1, X_2):
        """
        forward pass equations:
        Y_1 = X_1 + Attention(X_2), F = Attention
        Y_2 = X_2 + MLP(Y_1), G = MLP
        """

        # Y_1 : attn_output
        f_X_2 = self.F(X_2)

        assert f_X_2.shape == X_2.shape

        # self.seed_cuda("droppath")
        # f_X_2_dropped = drop_path(
        #     f_X_2, drop_prob=self.drop_path_rate, training=self.training
        # )

        # Y_1 = X_1 + f(X_2)
        # Y_1 = X_1 + f_X_2_dropped
        Y_1 = X_1 + f_X_2

        # free memory
        del X_1

        g_Y_1 = self.G(Y_1)

        # torch.manual_seed(self.seeds["droppath"])
        # g_Y_1_dropped = drop_path(
        #     g_Y_1, drop_prob=self.drop_path_rate, training=self.training
        # )

        # Y_2 = X_2 + g(Y_1)
        # Y_2 = X_2 + g_Y_1_dropped
        Y_2 = X_2 + g_Y_1

        del X_2

        return Y_1, Y_2

    def backward_pass(
        self,
        Y_1,
        Y_2,
        dY_1,
        dY_2,
    ):
        """
        equation for activation recomputation:
        X_2 = Y_2 - G(Y_1), G = MLP
        X_1 = Y_1 - F(X_2), F = Attention
        """
	# TODO: I don't fully understand
	# why this works ... specific questions around
	# how the gradients dX_1 and dX_2 are being calculated

        # temporarily record intermediate activation for G
        # and use them for gradient calculcation of G
        with torch.enable_grad():

            Y_1.requires_grad = True

            g_Y_1 = self.G(Y_1)
            assert g_Y_1.shape == Y_1.shape

            # torch.manual_seed(self.seeds["droppath"])
            # g_Y_1 = drop_path(
            #     g_Y_1, drop_prob=self.drop_path_rate, training=self.training
            # )

            g_Y_1.backward(dY_2, retain_graph=True)

        # activation recomputation is by design and not part of
        # the computation graph in forward pass.
        with torch.no_grad():

            X_2 = Y_2 - g_Y_1
            del g_Y_1

            dY_1 = dY_1 + Y_1.grad
            Y_1.grad = None

        # record F activations and calc gradients on F
        with torch.enable_grad():
            X_2.requires_grad = True

            f_X_2 = self.F(X_2)

            # torch.manual_seed(self.seeds["droppath"])
            # f_X_2 = drop_path(
            #     f_X_2, drop_prob=self.drop_path_rate, training=self.training
            # )

            f_X_2.backward(dY_1, retain_graph=True)

        # propagate reverse computed acitvations at the start of
        # the previou block for backprop.s
        with torch.no_grad():
            X_1 = Y_1 - f_X_2

            del f_X_2, Y_1
            dY_2 = dY_2 + X_2.grad

            X_2.grad = None
            X_2 = X_2.detach()

        return X_1, X_2, dY_1, dY_2


class RevBackProp(Function):
    @staticmethod
    def forward(
        ctx,
        x,
        layers,
    ):
        X_1, X_2 = torch.chunk(x, 2, dim=-1)

        for layer in layers:
            X_1, X_2 = layer(X_1, X_2)

        all_tensors = [X_1.detach(), X_2.detach()]

        ctx.save_for_backward(*all_tensors)
        ctx.layers = layers

        return torch.cat([X_1, X_2], dim=-1)

    @staticmethod
    def backward(ctx, dx):
        dX_1, dX_2 = torch.chunk(dx, 2, dim=-1)
        assert dX_1.shape == dX_2.shape

        X_1, X_2 = ctx.saved_tensors
        layers = ctx.layers

        for _, layer in enumerate(layers[::-1]):
            X_1, X_2, dX_1, dX_2 = layer.backward_pass(
                Y_1=X_1,
                Y_2=X_2,
                dY_1=dX_1,
                dY_2=dX_2,
            )

        dx = torch.cat([dX_1, dX_2], dim=-1)

        del dX_1, dX_2, X_1, X_2

        return dx, None, None


# def patch_embed()
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
#             nn.Linear(patch_dim, dim),
#         )


def patch_embed(dim_out: int, patch_size: int, img_size: int):
    assert (
        img_size % patch_size == 0
    ), f"img_size: {img_size} not divisible by patch_size: {patch_size}"
    return nn.Sequential(
        nn.Conv2d(
            3,
            dim_out,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            padding=(0, 0),
        ),
	Rearrange("b dim_out w_patched h_patched -> b (w_patched h_patched) dim_out")
    )


class ReversibleViTParams(hp.Hparams):
    depth: int = hp.required("# of transformer blocks")
    model_dim: int = hp.required("width of internal representation")
    heads: int = hp.required("# of attention of heads")
    patch_size: int = hp.required("width/height of patch")


class ReversibleVIT(nn.Module):
    def __init__(self, cfg, img_size: int):
        super().__init__()

        depth, model_dim, heads, mlp_ratio = (
            cfg.depth,
            cfg.model_dim,
            cfg.heads,
            cfg.mlp_ratio,
        )
        assert (
            model_dim % 2 == 0
        ), f"model_dim must be divisible by 2 for reversible ViT"
        self.patchify = patch_embed(model_dim // 2, cfg.patch_size, img_size)
        num_patches = (img_size // cfg.patch_size) ** 2
        self.pos_embed = nn.Parameter(
		torch.zeros(
			1,
			num_patches,
			model_dim // 2,
		)
	)
	# Initialization taken from PySlowFast
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            block = ReversibleBlock(
                dim=model_dim,
                heads=heads,
                mlp_ratio=mlp_ratio,
                drop_path_rate=None,
            )
            self.blocks.append(block)
        self.norm = norm(model_dim)

    def forward(self, x):
        patches = self.patchify(x)
        patches += self.pos_embed

        concat = torch.cat([patches, patches], dim=-1)
        concat = RevBackProp.apply(concat, self.blocks)

        concat = self.norm(concat)
        concat = concat.mean(1)

        return concat


if __name__ == "__main__":
    cfg = SimpleNamespace(
        **dict(
            depth=6,
            model_dim=768,
            heads=8,
            patch_size=16,
	    mlp_ratio=4,
        )
    )

    img_size = 16 * 14
    model = ReversibleVIT(cfg, img_size=img_size)

    x = torch.randn((1, 3, img_size, img_size))
    y = model(x)
    print("Forward finished")
    y.sum().backward()
    print("Backward finished")
