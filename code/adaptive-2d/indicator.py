import torch


def gradients(input, output):
    return torch.autograd.grad(outputs=output, inputs=input, grad_outputs = torch.ones_like(output),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]


def indicator_res_same(net_plus, net_minus, p, pde):
    '''interior'''
    # p.requires_grad_()
    #v1
    output = torch.where(pde.phi(p), net_plus(p), net_minus(p))
    diff_coeff = pde.diffusion_coefficient_judge(p)
    grad = gradients(p, output) * diff_coeff
    delta = gradients(p, grad)
    delta = torch.sum(delta, dim=1)
    delta = delta.reshape(-1, 1)
    rhs = pde.rhs_judge(p)
    res = torch.abs(-delta - rhs)

    #v2 xr_plus and xr_minus
    # dim = pde.dimension()
    # xr_sign = pde.phi(xr)
    # mask_plus = xr_sign
    # mask_minus = ~xr_sign
    # xr_plus = torch.masked_select(xr, mask_plus)
    # xr_plus = xr_plus.reshape(-1, dim)
    # xr_minus = torch.masked_select(xr, mask_minus)
    # xr_minus = xr_minus.reshape(-1, dim)

    # diff_coeff_r_plus = pde.diffusion_plus(xr_plus)
    # diff_coeff_r_minus = pde.diffusion_minus(xr_minus)
    # output_r_plus = net_plus(xr_plus)
    # output_r_minus = net_minus(xr_minus)
    #
    # grad_r_plus = gradients(xr_plus, output_r_plus)*diff_coeff_r_plus
    # delta_r_plus = gradients(xr_plus, grad_r_plus)
    # delta_r_plus = torch.sum(delta_r_plus, dim=1)
    # delta_r_plus = delta_r_plus.reshape(-1, 1)
    #
    # grad_r_minus = gradients(xr_minus, output_r_minus)*diff_coeff_r_minus
    # delta_r_minus = gradients(xr_minus, grad_r_minus)
    # delta_r_minus = torch.sum(delta_r_minus, dim=1)
    # delta_r_minus = delta_r_minus.reshape(-1, 1)
    #
    # rhs_r_plus = pde.rhs_plus(xr_plus)
    # rhs_r_minus = pde.rhs_minus(xr_minus)
    # res_r_plus = torch.abs(-delta_r_plus - rhs_r_plus)
    # res_r_minus = torch.abs(-delta_r_minus - rhs_r_minus)
    # res_r = torch.cat((res_r_plus, res_r_minus), dim=0)
    # print('res_r[:10]', res_r[:10])
    # print('res_r_plus.shape', res_r_plus.shape, '\n', 'res_r_minus.shape', res_r_minus.shape)
    return res

def indicator_res_own(net_plus, net_minus, xr, xf, xb, pde):
    xr.requires_grad_()
    xf.requires_grad_()
    xb.requires_grad_()

    '''interior'''
    # output_r = torch.where(pde.phi(xr), net_plus(xr), net_minus(xr))
    # diff_coeff_r = pde.diffusion_coefficient_judge(xr)
    # grad_r = gradients(xr, output_r) * diff_coeff_r
    # delta_r = gradients(xr, grad_r)
    # delta_r = torch.sum(delta_r, dim=1)
    # delta_r = delta_r.reshape(-1, 1)
    # rhs_r = pde.rhs_judge(xr)
    # res_r = torch.abs(-delta_r - rhs_r)
    # res_r = torch.square(res_r)

    #v2
    dim = pde.dimension()
    xr_sign = pde.phi(xr)
    mask_plus = xr_sign
    mask_minus = ~xr_sign
    xr_plus = torch.masked_select(xr, mask_plus)
    xr_plus = xr_plus.reshape(-1, dim)
    xr_minus = torch.masked_select(xr, mask_minus)
    xr_minus = xr_minus.reshape(-1, dim)

    diff_coeff_r_plus = pde.diffusion_plus(xr_plus)
    diff_coeff_r_minus = pde.diffusion_minus(xr_minus)
    output_r_plus = net_plus(xr_plus)
    output_r_minus = net_minus(xr_minus)

    grad_r_plus = gradients(xr_plus, output_r_plus)*diff_coeff_r_plus
    delta_r_plus = gradients(xr_plus, grad_r_plus)
    delta_r_plus = torch.sum(delta_r_plus, dim=1)
    delta_r_plus = delta_r_plus.reshape(-1, 1)

    grad_r_minus = gradients(xr_minus, output_r_minus)*diff_coeff_r_minus
    delta_r_minus = gradients(xr_minus, grad_r_minus)
    delta_r_minus = torch.sum(delta_r_minus, dim=1)
    delta_r_minus = delta_r_minus.reshape(-1, 1)

    rhs_r_plus = pde.rhs_plus(xr_plus)
    rhs_r_minus = pde.rhs_minus(xr_minus)
    res_r_plus = torch.square(-delta_r_plus - rhs_r_plus)
    res_r_minus = torch.square(-delta_r_minus - rhs_r_minus)
    res_r = torch.cat((res_r_plus, res_r_minus), dim=0)

    '''Interface'''
    output_f_plus = net_plus(xf)
    output_f_minus = net_minus(xf)
    output_f_val_jump = output_f_plus - output_f_minus #value jump

    grad_f_plus = gradients(xf, output_f_plus)
    grad_f_minus = gradients(xf, output_f_minus)
    interface_normal = pde.interface_normal(xf)
    beta_plus = pde.diffusion_plus(xf)
    beta_minus = pde.diffusion_minus(xf)
    flux_plus = torch.sum(grad_f_plus * interface_normal, dim=1)
    flux_plus = flux_plus.reshape(-1, 1)
    flux_plus *= beta_plus
    flux_minus = torch.sum(grad_f_minus * interface_normal, dim=1)
    flux_minus = flux_minus.reshape(-1, 1)
    flux_minus *= beta_minus
    output_f_flux_jump = flux_plus - flux_minus #flux jump

    fun_jump = pde.function_jump_p(xf)
    flux_jump = pde.flux_jump_q(xf)
    res_f = torch.square(output_f_val_jump - fun_jump) + torch.square(output_f_flux_jump - flux_jump)

    '''Boundary'''
    output_b = torch.where(pde.phi(xb), net_plus(xb), net_minus(xb))
    bnd_val = pde.BC_Dirichlet(xb)
    # res_b = torch.abs()
    res_b = torch.square(output_b - bnd_val)

    #v1-together
    res = torch.cat((res_r, res_f, res_b), dim=0)
    return res
    #v2-rank only
    # return res_r, res_f, res_b


def indicator_grad(net_plus, net_minus, xr, xf, xb, pde):
# def indicator_grad(net_plus, net_minus, xr, xf, pde):

    xr.requires_grad_()
    xf.requires_grad_()
    xb.requires_grad_()
    # #xr-grad-v2
    # dim = pde.dimension()
    # xr_sign = pde.phi(xr)
    # mask_plus = xr_sign
    # mask_minus = ~xr_sign
    # xr_plus = torch.masked_select(xr, mask_plus)
    # xr_plus = xr_plus.reshape(-1, dim)
    # xr_minus = torch.masked_select(xr, mask_minus)
    # xr_minus = xr_minus.reshape(-1, dim)
    # output_r_plus = net_plus(xr_plus)
    # output_r_minus = net_minus(xr_minus)
    # grad_r_plus = gradients(xr_plus, output_r_plus)
    # grad_r_minus = gradients(xr_minus, output_r_minus)
    # grad_r = torch.cat((grad_r_plus, grad_r_minus), dim=0)
    # print('grad_r.shape', grad_r.shape)
    # print('grad_r', grad_r)
#xr-grad-v1
    output_r = torch.where(pde.phi(xr), net_plus(xr), net_minus(xr))
    grad_r = gradients(xr, output_r)
    # # print('grad_r', grad_r)
    grad_r_magnitude = torch.norm(grad_r, dim=1)

    # #xb-grad
    output_b = net_plus(xb)
    grad_b = gradients(xb, output_b)
    grad_b_magnitude = torch.norm(grad_b, dim=1)

    #xf-grad
    grad_f_plus = gradients(xf, net_plus(xf))
    plus_magnitude = torch.norm(grad_f_plus, dim=1)
    grad_f_minus = gradients(xf, net_minus(xf))
    minus_magnitude = torch.norm(grad_f_minus, dim=1)
    max_magnitude, _ = torch.max(torch.stack([plus_magnitude, minus_magnitude], dim=1), dim=1)
    grad_f_magnitude = max_magnitude
    # grad_all_magnitude = torch.cat((grad_r_magnitude, grad_f_magnitude), dim=0)
    grad_all_magnitude = torch.cat((grad_r_magnitude, grad_f_magnitude, grad_b_magnitude), dim=0)

#   print(f'grad_all_magnitude.shape={grad_all_magnitude.shape}')
    #       f'\n,'
    #       f'grad_r_magnitude= {grad_r_magnitude.shape}, '
    #       f'grad_f_magnitude={grad_f_magnitude.shape} '
    # f'grad_b_magnitude={grad_b_magnitude.shape}')
    return grad_all_magnitude

