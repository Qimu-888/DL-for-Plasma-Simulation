from sunflower_v2 import *
# from train_refine_v3 import *
from mayavi import mlab
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def visualize_2d(h):
    # box = torch.tensor([0, 1, 0, 1], dtype=torch.float64) #straight
    box = torch.tensor([-1, 1, -1, 1], dtype=torch.float64)
    box = box.detach().numpy()
    x1 = torch.linspace(box[0], box[1], h)
    x2 = torch.linspace(box[2], box[3], h)
    X, Y = torch.meshgrid(x1, x2)
    Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)

    real_minus = torch.where(pde.phi(Z), torch.nan, pde.exact_minus(Z)).reshape(h, h)
    real_plus = torch.where(pde.phi(Z), pde.exact_plus(Z), torch.nan).reshape(h, h)
    pred_minus = torch.where(pde.phi(Z), torch.nan, net_minus(Z)).reshape(h, h)
    pred_plus = torch.where(pde.phi(Z), net_plus(Z), torch.nan).reshape(h, h)

    real = pde.exact_judge(Z).reshape(h, h)
    pred = torch.where(pde.phi(Z), net_plus(Z), net_minus(Z)).reshape(h, h)

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f'Comparison-Adaptive Sampling', y=0.92)
    ax = fig.add_subplot(121, projection='3d')
    # surf1 = ax.plot_surface(X, Y, real_plus, cmap='jet', vmin = real.min(), vmax= real.max(), alpha=1)
    surf1 = ax.plot_surface(X, Y, real_plus, cmap='jet', vmin = real.min(), vmax= real.max(), alpha=0.76)
    surf2 = ax.plot_surface(X, Y, real_minus, cmap='jet', vmin = real.min(), vmax= real.max(), alpha= 1)
    ax.view_init(elev=34, azim=-76)
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=5)
    ax.set_title('Real Values')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=5)


    ax1 = fig.add_subplot(122, projection='3d')
    # surf3 = ax1.plot_surface(X, Y, pred_plus.detach().numpy(), cmap='jet', vmin = pred.min(), vmax= pred.max(),alpha=1)

    surf3 = ax1.plot_surface(X, Y, pred_plus.detach().numpy(), cmap='jet', vmin = pred.min(), vmax= pred.max(),alpha=0.76)
    ax1.plot_surface(X, Y, pred_minus.detach().numpy(), cmap='jet', vmin = pred.min(), vmax= pred.max(), alpha=1)
    ax1.view_init(elev=34, azim=-76)
    ax1.locator_params(axis='x', nbins=5)  # 减少x轴的刻度数量
    ax1.locator_params(axis='y', nbins=5)  # 减少y轴的刻度数量
    ax1.set_title('Predicted Values')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    fig.colorbar(surf3, ax=ax1, shrink=0.5, aspect=5)

    '''save'''
    # if epochs > 5:
    #     compare_path = (f'{save_path_flower}{epochs}epochs(bp, bm) = ({bp}, {bm})(gamma_b, gamma_f)=({gamma_b}, {gamma_f}(Nr, Nb, Nf) = ({Nr}, {Nb}, {Nf})'
    #                     f'(m, depth)=({m}, {depth}.pdf')
    #     fig.savefig(compare_path)


    abs_error = torch.abs(real - pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    abs_error_heatmap = ax.imshow(abs_error.detach().numpy(), cmap='jet', extent=[box[0], box[1], box[2], box[3]],
                                  origin='lower')
    ax.set_title('Absolute Error (Adaptive)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    cbar = fig.colorbar(abs_error_heatmap, ax=ax)
    min_val = abs_error_nitsche.detach().min()
    max_val = abs_error_nitsche.detach().max()
    num_ticks = 10
    tick_values = np.linspace(min_val, max_val, num_ticks)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f'{tick:.3f}' for tick in tick_values * 100])
    cbar.ax.text(0.1, 1.04, r'$\times 10^{-2}$', transform=cbar.ax.transAxes, fontsize=10, verticalalignment='top',
                 horizontalalignment='left')
    cbar.set_label('Absolute Error')
    plt.tight_layout()
    return abs_error

box = torch.tensor([-1, 1, -1, 1], dtype=torch.float64)
from DUNM.Module.visualize_2d_surface import abs_error_nitsche
'''v2'''
abs_error_adapt = visualize_2d(300)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
heatmap1 = ax1.imshow(abs_error_nitsche.detach().numpy() , cmap='jet', extent=[box[0], box[1], box[2], box[3]],  origin='lower')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Nitsche Method Error')

heatmap2 = ax2.imshow(abs_error_adapt.detach().numpy() , cmap='jet', extent=[box[0], box[1], box[2], box[3]], origin='lower')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Adaptive Method Error')

min_val = min(abs_error_nitsche.detach().numpy() .min(), abs_error_adapt.detach().numpy() .min())
max_val = max(abs_error_nitsche.detach().numpy() .max(), abs_error_adapt.detach().numpy() .max())

#flower
# cbar = fig.colorbar(heatmap2, ax=[ax1, ax2])
# heatmap1.set_cmap(heatmap2.get_cmap())
# heatmap1.set_clim(heatmap2.get_clim())

#sun
cbar = fig.colorbar(heatmap1, ax=[ax1, ax2])
heatmap2.set_cmap(heatmap1.get_cmap())
heatmap2.set_clim(heatmap1.get_clim())

num_ticks = 10
tick_values = np.linspace(min_val, max_val, num_ticks)
cbar.set_ticks(tick_values)
cbar.set_ticklabels([f'{tick:.3f}' for tick in tick_values*100])
cbar.ax.text(0.1, 1.04, r'$\times 10^{-2}$', transform=cbar.ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')
plt.show()




