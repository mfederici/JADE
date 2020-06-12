from torch.autograd import Function


class ScaleGrad(Function):
    @staticmethod
    def forward(ctx, input_, coeff):
        ctx.save_for_backward(input_)
        ctx.coeff = coeff
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = ctx.coeff * grad_output
        return grad_input, None