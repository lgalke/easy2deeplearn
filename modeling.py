import torch
import torch.nn as nn
import torch.nn.functional as F

from egg import core



class RelaxedLinear(nn.Linear):

    """Overwrite to allow second, ignored, input"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, __aux_input=None):
        return super().forward(x)


class TiedLinear(nn.Linear):

    """TiedLinear: A linear layer with tied weights to another linear layer"""

    def __init__(self, other: nn.Linear, bias=True, device=None, dtype=None):
        """Initializes a tied linear layer from its origin linear layer

        :other: nn.Module
        :bias: whether to use an extra bias term (no reuse)

        """
        nn.Linear.__init__(
            self,
            other.out_features,
            other.in_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        # We re-use the other linear's weight matrix transposed
        self.weight = other.weight

        # Still we need a new bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.bias = None

        self.reset_parameters()

    def forward(self, x):
        h = x @ self.weight
        if self.bias is not None:
            h += self.bias
        return h

    def reset_parameters(self):
        # we only reset the bias here
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def __repr__(self):
        return f"TiedLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"


class TiedEmbedding(nn.Embedding):

    """An Embedding layer tied to another linear layer"""

    def __init__(self, other, **kwargs):
        """Inits the embedding and ties weights to the original linear layer, no bias.

        :other: nn.Linear to be tied to

        """
        nn.Embedding.__init__(self, other.in_features, other.out_features, **kwargs)
        self.weight = other.weight

    def forward(self, x):
        """Embeds the input as usual but then returns its transpose"""
        return F.embedding(
            x,
            self.weight.t(),  # Access Linear's weights **transposed**
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


def tie_weights(
    sender: nn.Module,
    receiver: nn.Module,
    within_sender=True,
    between_sender_and_receiver=True,
):
    """Tie weights between sender and receiver"""
    print("Tying weights")
    if within_sender:
        print("Tying sender embedding and sender decoder")
        assert hasattr(sender, "hidden_to_output") and isinstance(
            sender.hidden_to_output, nn.Linear
        )
        assert hasattr(sender, "embedding")
        if isinstance(sender.embedding, nn.Linear):
            bias = sender.hidden_to_output.bias is not None
            print(f"... using TiedLinear with bias={bias}")
            sender.hidden_to_output = TiedLinear(sender.embedding, bias=bias)
        elif isinstance(sender.embedding, nn.Embedding):
            sender.hidden_to_output.weight = sender.embedding.weight
        else:
            raise AssertionError("Unknown type of sender.embedding")

    if between_sender_and_receiver:
        print("Tying sender embedding and receiver embedding")
        assert hasattr(sender, "embedding") and hasattr(receiver, "embedding")
        if (
            isinstance(sender.embedding, nn.Embedding)
            and isinstance(receiver.embedding, nn.Embedding)
        ) or (
            isinstance(sender.embedding, nn.Linear)
            and isinstance(receiver.embedding, nn.Linear)
        ):
            receiver.embedding.weight = sender.embedding.weight
        elif isinstance(sender.embedding, nn.Linear) and isinstance(
            receiver.embedding, nn.Embedding
        ):
            print(f"... using TiedEmbedding")
            receiver.embedding = TiedEmbedding(
                sender.embedding,
                padding_idx=receiver.embedding.padding_idx,
                max_norm=receiver.embedding.max_norm,
                norm_type=receiver.embedding.max_norm,
                scale_grad_by_freq=receiver.embedding.scale_grad_by_freq,
                sparse=False,
            )
        else:
            raise AssertionError("Could not resolve type mismatch")




class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size,
        output_dim,
        num_layers=2,
        dropout=0.5,
        act_fn=F.relu,
    ):
        super().__init__()
        assert num_layers >= 2, "Just use a Linear layer..."
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_size))
        for __ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_dim))

        self.act_fn = act_fn
        self.drop = nn.Dropout(dropout)

    def forward(self, x, input=None, aux_input=None):
        h = x
        for i, layer in enumerate(self.layers):

            h = layer(h)

            if i != (len(self.layers) - 1):
                h = self.act_fn(h)
                h = self.drop(h)

        return h


class TiedRnnReinforce(nn.Module):

    """Agent that shares parameters between discriminative and generative tasks"""

    def __init__(
        self,
        input2hidden,
        hidden2output,
        vocab_size,
        embed_dim,
        hidden_size,
        max_len,
        num_layers=1,
        cell="rnn",
        deterministic=False,
    ):
        super().__init__()
        self.input2hidden = input2hidden
        self.hidden2output = hidden2output

        # Generative Language Model
        # the wrapped agent returns the initial hidden state for a RNN cell
        # input_dim -> hidden_size

        self.sender = core.RnnSenderReinforce(
            self.input2hidden,
            vocab_size,
            embed_dim,
            hidden_size,
            max_len,
            num_layers=num_layers,
        )

        # Discriminative Language Model
        # calls the wrapped agent with the hidden state
        # As the wrapped agent does not sample, it has to be trained via regular back-propagation
        self.receiver = core.RnnReceiverDeterministic(
            self.hidden2output,
            vocab_size,
            embed_dim,
            hidden_size,
            cell=cell,
            num_layers=num_layers,
        )

        tie_weights(self.sender, self.receiver)

    def forward(
        self,
        sender_input=None,
        aux_input=None,
        message=None,
        receiver_input=None,
        lengths=None,
    ):
        if message is not None:
            outputs = self.receiver(
                message, input=receiver_input, aux_input=aux_input, lengths=length
            )
        else:
            assert sender_input is not None
            outputs = self.sender(sender_input, aux_input=aux_input)
        return outputs


class TiedRnnGS(nn.Module):

    """Docstring for TiedRnnSenderReceiverGS."""

    def __init__(
        self,
        input2hidden: nn.Module,
        hidden2output: nn.Module,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        max_len: int,
        temperature: float,
        cell: str = "rnn",
        trainable_temperature: bool = False,
        straight_through: bool = False,
        tied_weights: str = "all",
    ):
        nn.Module.__init__(self)

        self.input2hidden = input2hidden  # MLP / Linear
        self.hidden2output = hidden2output  # MLP / Linear

        self.sender = core.RnnSenderGS(  # Sender RNN
            self.input2hidden,
            vocab_size,
            embed_dim,
            hidden_size,
            max_len,
            temperature,
            cell=cell,
            trainable_temperature=trainable_temperature,
            straight_through=straight_through,
        )

        # self.receiver = core.RnnReceiverGS(
        #     self.hidden2output, vocab_size, embed_dim, hidden_size, cell=cell
        # )
        self.receiver = core.RnnEncoder(  # Receiver
            vocab_size, embed_dim, hidden_size, cell=cell, num_layers=1
        )

        # Handle tied_weights argument
        assert tied_weights in [True, "all", "between", "within", "none", False, None]
        if tied_weights == "all" or tied_weights == True:  # True is an alias for 'all'
            tie_weights(self.sender, self.receiver)
        elif tied_weights == "between":
            # Tie only between sender and receiver, but not sender input & output layers
            tie_weights(
                self.sender,
                self.receiver,
                within_sender=False,
                between_sender_and_receiver=True,
            )
        elif tied_weights == "within":
            # Tie only within sender
            tie_weights(
                self.sender,
                self.receiver,
                within_sender=True,
                between_sender_and_receiver=False,
            )
        else:  # Falsy ['none', False, None]
            print("[TiedRnnGS/warning] No tied weights, was this intended?")

    def forward(
        self,
        sender_input=None,
        aux_input=None,
        message=None,
        receiver_input=None,
    ):
        if message is not None:
            # print(message)
            # print(message.shape)

            # outputs = self.receiver(message, input=receiver_input, aux_input=aux_input)
            last_hidden = self.receiver(message)
            outputs = self.hidden2output(last_hidden)
        else:
            assert sender_input is not None
            outputs = self.sender(
                sender_input, aux_input=aux_input
            )  # bsz, seqlen, vocab_size
        return outputs
