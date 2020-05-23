"""

https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html

"""
import torch
import torch.nn as nn

class Attention(nn.Module):

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError("Invalid attention type selected")

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """

        :param query: [batch_size, output_length, dimensions]
        :param context: [batch_size, output_length, query_length]
        :return:
        output [batch_size, output_legth, dimensions]
        weights [batch_size, output_length, query_length]
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == 'general':
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # 1. Compute a score for each encoder state
        attention_scores = torch.bmm(query, context.transpose(1,2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        print("ATT SCORES")
        print(attention_scores.size())

        # 2. Compute the attention weights
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # 3. Compute the new context vector S
        mix = torch.bmm(attention_weights, context)

        # 4. Concatenate context vector with output of previous time step
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        #
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return mix, output, attention_weights

#%%

if __name__ == "__main__":
    attention = Attention(7, "dot")
    query = torch.randn(5, 1, 7)    # puede ser el embedding del usuario?
    context = torch.randn(5, 6, 7)

    print("~~~~ QUERY SIZE (user / item embedding) ~~~~")
    print(query.size())
    print("**** QUERY ****")
    print(query)

    print("@@@@ CONTEXT (sentence embedding) @@@@")
    print(context.size())
    print("^^^^ CONTEXT ^^^^")
    print(context)


    mix, output, weight = attention(query, context)
    print("!!!! OUTPUT SIZE (context S) !!!!")
    print(output.size())

    print("%%%% WEIGHT SIZE (weight for each word in sentence) %%%%")
    print(weight.size())

    """
    emb  = nn.Embedding(10, 7)
    lookup_tensor = torch.tensor([0], dtype=torch.long)
    res = emb(lookup_tensor)
    print(res)
    print(res.size())
    """

    print("OUTPUT~~~~`")
    print(output)
    print("~~~WEIGHT")
    print(weight)
    print("$$$$ WEIGHT SUM $$$$")
    print(torch.sum(weight, dim=2))

    print("&&&& MIX SIZE &&&&")
    print(mix.size())
    print("&&&& MIX &&&&")
    print(mix)




