import matplotlib.pyplot as plt
import pandas as pd
import time
import matplotlib
# matplotlib.use('Qt5Agg')  # 使用R Qt5Agg 后端
import itertools
from matplotlib.lines import Line2D

from DUNM.Module.Res_Net import ResNet
from DUNM.Module.Res_Net import FCNet

from indicator import *
'''#jump2-1'''
from DUNM.Module.Loss_fun_work_v1 import *
#2d
from DUNM.Module.jump2minus1_2d import *

def setup_seed(seed):
    torch.manual_seed(seed)  # Results of random operations on the CPU will be deterministic and reproducible.
setup_seed(888888)

def weights_init_normal(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.normal_(m.weight, 0, 0.5)
        torch.nn.init.constant_(m.bias, 0.0)

def l2_error(u, uh):
    num = torch.sqrt(torch.mean(torch.pow(u-uh, 2)))
    den = torch.sqrt(torch.mean(torch.pow(u, 2)))
    error = num / den
    return error

#v1
def interface_points(N_f=32):
    theta_initial = torch.linspace(0, 2 * torch.pi, N_f, dtype=torch.float32)
    theta_initial = theta_initial.reshape(-1, 1)
    r = 1/2 + torch.sin(5*theta_initial)/7
    x_initial = r * torch.cos(theta_initial)
    y_initial = r * torch.sin(theta_initial)
    xf = torch.cat((x_initial, y_initial), dim=1)
    xf = torch.cat((xf, torch.zeros(xf.size(0), 1)), dim=-1)
    return xf

def interface_points_add(xf, n_insert=1):
    N_f = xf.shape[0]
    new_xf = []
    for i in range(N_f):
        theta_start = torch.atan2(xf[i, 1], xf[i, 0]).item()
        theta_end = torch.atan2(xf[(i + 1) % N_f, 1], xf[(i + 1) % N_f, 0]).item()
        if theta_end < theta_start:
            theta_end += 2 * torch.pi
        theta_insert = torch.linspace(theta_start, theta_end, n_insert + 2, dtype=torch.float32)[:-1]
        #flower shape
        r_insert = 1 / 2 + torch.sin(5 * theta_insert) / 7
        x_insert = r_insert * torch.cos(theta_insert)
        y_insert = r_insert * torch.sin(theta_insert)
        new_xf.append(torch.stack([x_insert, y_insert], dim=1))
    new_xf = torch.cat(new_xf, dim=0)
    distances = torch.cdist(xf, new_xf).float()
    overlap_indices = torch.any(distances < 1e-2, dim=0)
    new_xf = new_xf[~overlap_indices]
    new_xf = torch.cat((new_xf, torch.zeros(new_xf.size(0), 1)), dim=-1)
    return new_xf

def generate_grid_points(h):
    x_coords = torch.linspace(-1, 1, steps=h)
    y_coords = torch.linspace(-1, 1, steps=h)
    X, Y = torch.meshgrid(x_coords, y_coords,  indexing = 'xy')
    grid_points_2d = torch.stack([X, Y], dim=-1)
    grid_points_2d = grid_points_2d.view(-1, 2)
    grid_points_3d = torch.cat((grid_points_2d, torch.zeros(grid_points_2d.size(0), 1)), dim=-1)
    return grid_points_3d

def add_grid(points, refinement_factor, h):#pass in points-3d
    refined_points = []
    for i, point in enumerate(points):
        count = point[-1].item()
        initial_spacing = (1 / (h - 1)) / (2 ** count)
        for dx in np.linspace(-initial_spacing, initial_spacing, refinement_factor):
            for dy in np.linspace(-initial_spacing, initial_spacing, refinement_factor):
                new_point = point[:3] + torch.tensor([dx, dy,0], dtype=points.dtype)
                new_point[-1] = count + 1
                if not any(torch.allclose(new_point[:2], p[:2]) for p in points):                    refined_points.append(new_point)
    refined_points = [p for p in refined_points if -1 <= p[0] <= 1 and -1 <= p[1] <= 1]
    refined_points = torch.stack(refined_points)
    refined_points, _ = torch.unique(refined_points, dim=0, return_inverse=True)
    refined_points = refined_points.clone().detach().requires_grad_(True)
    return refined_points

def initial_data(h=10, Nf=32):
    grid_points = generate_grid_points(h)
    boundary_mask = (torch.abs(grid_points[:,:2]).max(dim=1)[0] == 1.0).bool() #only max tensor not indices
    xb = grid_points[boundary_mask]
    xr = grid_points[~boundary_mask]
    xf = interface_points(Nf)
    xr = xr.clone().detach().requires_grad_(True)
    xf = xf.clone().detach().requires_grad_(True)
    xb = xb.clone().detach().requires_grad_(True)
    return xr, xf, xb
def classify_x_all(x_all):
    r = torch.norm(x_all[:, :2], dim=1)
    theta = torch.atan2(x_all[:, 1], x_all[:, 0])
    interface_r = 0.5 + torch.sin(5*theta)/7
    interface_mask = torch.isclose(r, interface_r, atol=0).bool()
    boundary_threshold = 1.0
    boundary_mask = (torch.abs(x_all[:,:2]).max(dim=1)[0] == boundary_threshold).bool() #only max tensor not indices
    combined_mask = interface_mask | boundary_mask
    interior_mask = ~combined_mask
    xb = x_all[boundary_mask]
    xf = x_all[interface_mask]
    xr = x_all[interior_mask]
    xr = xr.clone().detach().requires_grad_(True)
    xf = xf.clone().detach().requires_grad_(True)
    xb = xb.clone().detach().requires_grad_(True)
    return xr, xf, xb


def train_only(epochs, gamma_b, gamma_f, xr, xf, xb, net_plus, net_minus, opt1, opt2):
    loss_list = []
    l2_list = []
    x_t = pde.interior_points(10000)
    u_exact = pde.exact_judge(x_t)

    for i in range(epochs):
        loss_energy, loss_interior, loss_interface, loss_bc =\
        loss_nitsche(net_plus, net_minus, xr, xf, xb, pde, gamma_b, gamma_f)
        '''train pde network, record loss'''
        opt1.zero_grad()
        opt2.zero_grad()
        loss_energy.backward()
        opt1.step()
        opt2.step()
        if i % 500 == 0:
            loss_list.append(loss_energy.item())
            u_pred = torch.where(pde.phi(x_t), net_plus(x_t), net_minus(x_t))
            relative_l2 = l2_error(u_exact, u_pred)
            l2_list.append(relative_l2.item())
            print(f'At iteration = {i}, pde_loss = {loss_energy}, l2_error={relative_l2}, interior_loss = {loss_interior},'
                  f'bc loss = {loss_bc}, interface loss = {loss_interface}')
    min_l2error = min(l2_list)
    with torch.no_grad():
       x_t = pde.interior_points(1200)
       u = pde.exact_judge(x_t)
       uh = torch.where(pde.phi(x_t), net_plus(x_t), net_minus(x_t))
       uh = uh.reshape(-1, 1)
       err_l2 = l2_error(u, uh)
    print(f'Relative l2 error of last epoch of', err_l2)
    return net_plus, net_minus, l2_list, loss_list, err_l2

def train_adapt_grid( h, Nf, re_level, re_factor,num_insert, epochs,
                      gamma_b, gamma_f,
                      m=10, depth=3, d=2):
    # net_plus = ResNet(d, m, 1, depth=depth)
    # net_minus = ResNet(d, m, 1, depth=depth)
    net_plus = FCNet(d, m, 1, depth=depth)
    net_minus = FCNet(d, m, 1, depth=depth)

    opt1 = torch.optim.Adam(net_plus.parameters(), lr=1e-3)
    opt2 = torch.optim.Adam(net_minus.parameters(), lr=1e-3)

    xr, xf, xb = initial_data(h, Nf)
    print('xr.size(0), xb.size(0)', xr.size(0), xb.size(0))
    l2_list = []
    total_l2_list = []
    total_loss_list = []
    df = []
    i = 0
    while i < re_level:
        i +=1
        old_point = torch.cat((xr, xf, xb), dim=0)
        start_time = time.time()
        net_plus, net_minus, sub_l2_list, sub_loss_list, err_l2 = train_only(epochs, gamma_b, gamma_f, xr[:,:2], xf[:,:2], xb[:,:2], net_plus, net_minus, opt1, opt2)
        l2_list.append(err_l2)
        total_l2_list.append(sub_l2_list)
        total_loss_list.append(sub_loss_list)

        '''add and store more sampling points'''
        grid_add = add_grid(torch.cat((xr, xb), dim=0), re_factor, h)
        # grid_add = generate_new_points(torch.cat((xr, xb), dim=0), re_factor)

        xr_add, _, xb_add = classify_x_all(grid_add)
        xf_add = interface_points_add(xf[:,:2], num_insert)#Nf*3
        # print(f'xf_add.shape={xf_add.shape}, xr_add.shape={xr_add.shape}, xb_add.shape={xb_add.shape}')

        # '''v1'''
        # '''compute residual'''
        #v1: pass xr,xf,xb into｜-\delta u -f|
        # x_all = torch.cat((xr_add, xf_add, xb_add), dim=0)
        # res = indicator_res_same(net_plus, net_minus, x_all, pde)

        # '''v2 xr,xf,xb own residual but cat and rank together'''
        # res = indicator_res_own(net_plus, net_minus, xr_add, xf_add, xb_add, pde)

        # v3-rank grad
        res = indicator_grad(net_plus, net_minus, xr_add[:,:2], xf_add[:,:2], xb_add[:,:2], pde)#v1
        # res = indicator_grad(net_plus, net_minus, torch.cat((xr_add[:,:2], xb_add[:,:2]), dim=0), xf_add[:,:2], pde)#v2
        res_list = res.tolist()
        num_top10 = int(0.10 * len(res_list))
        sorted_indices = sorted(range(len(res_list)), key=lambda k: res_list[k], reverse=True)
        x_all_add = torch.cat((xr_add, xf_add, xb_add), dim=0)#N*3
        x_all_sorted_points = x_all_add[sorted_indices]
        x_top_ten = x_all_sorted_points[:num_top10]
        x_top_ten, _ = torch.unique(x_top_ten, dim=0, return_inverse=True)
        # print('x_top_ten.shape', x_top_ten.shape)
        # print('x_top_ten', x_top_ten)
        distances = torch.cdist(x_top_ten[:, :2], x_top_ten[:, :2]).float()
        unique_points = []
        seen = set()
        threshold = 1e-3
        for a in range(x_top_ten.size(0)):
            if a not in seen:
                distances_to_a = distances[a, :]
                distances_to_a[a] = float('inf')  # 排除自身
                if torch.all(distances_to_a > threshold):
                    unique_points.append(x_top_ten[a])
                else:
                    unique_points.append(x_top_ten[a])
                    for j, distance in enumerate(distances_to_a):
                        if distance <= threshold:
                            seen.add(j)
        unique_points = torch.stack(unique_points)
        x_top_ten = unique_points
        # print('AFTER x_top_ten.shape', x_top_ten.shape)
        xr_top, xf_top, xb_top = classify_x_all(x_top_ten)#N*3 with the 3rd:+1
        print(f'xr_top.shape={xr_top.shape}, xf_top.shape={xf_top.shape}, xb_top.shape={xb_top.shape}')
        xr = torch.cat((xr, xr_top), dim=0)
        xf = torch.cat((xf, xf_top), dim=0)
        xb = torch.cat((xb, xb_top), dim=0)
        end_time = time.time()
        print(f'(ADD already) xr.shape={xr.shape}, xf.shape={xf.shape}, xb.shape={xb.shape}')
        add_point = x_top_ten
        refine_time = end_time - start_time
        result = {'Refine Level': i, 'l2 error': err_l2.item(), 'Num of (xr,xf,xb)': [xr.size(0),xf.size(0), xb.size(0)],'Training time':refine_time}
        df.append(result)
        plt.figure()
        plt.scatter(old_point[:, 0].detach().numpy(), old_point[:, 1].detach().numpy(), color='red', label='old')
        plt.scatter(add_point[:, 0].detach().numpy(), add_point[:, 1].detach().numpy(), color='blue', label='add', s=20)
        plt.title(f"Top 10% Points (ranked by gradients) for level {i}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/flower/flower_{i}.pdf')
        plt.show()
    df = pd.DataFrame(df)
    return net_plus, net_minus, df, total_l2_list, total_loss_list

'''flower'''
pde = flowerInterface()
re_level = 3
epochs = 1500

re_factor = 3
num_insert = 2
h = 10
Nf = 32
gamma_b = 3000
gamma_f = 100

m = 64
depth = 4
# m = 10
# depth = 3

net_plus, net_minus, df, total_l2_list, total_loss_list = train_adapt_grid(h, Nf, re_level, re_factor, num_insert, epochs, gamma_b, gamma_f, m, depth)
final_time = df['Training time'].sum()
df[f'Final Time of {epochs * re_level} epochs'] = None
df.at[0, f'Final Time of {epochs * re_level} epochs'] = final_time
df[f'(gamma_b, gamma_f)'] = None
df.at[0, f'(gamma_b, gamma_f)'] = [gamma_b, gamma_f]
df[f'(m, depth)'] = None
df.at[0, f'(m, depth)'] = [m, depth]
df.to_excel(f'results/flower/re_factor={re_factor}_num_insert={num_insert}_h={h}_(gammab,f)=({gamma_b}, {gamma_f}).xlsx', index=False)
# df.to_excel(f'results/sunflower/re_factor={re_factor}_num_insert={num_insert}_h={h}_(gammab,f)=({gamma_b}, {gamma_f}).xlsx', index=False)
print('df', df)

'''pde'''
combined_pde_loss = list(itertools.chain(*total_loss_list))
level_end_indices_pde = [len(sublist) for sublist in total_loss_list]
level_end_indices_cumulative_pde = [sum(level_end_indices_pde[:i+1]) - 1 for i in range(len(level_end_indices_pde))]
fig, ax = plt.subplots()
ax.plot(combined_pde_loss, label='PDE Loss')

#mark at the end of level
for i, index in enumerate(level_end_indices_cumulative_pde):
    ax.plot(index, combined_pde_loss[index], 'ro')
    ax.text(index, combined_pde_loss[index], str(i+1), color='black', ha='center')

ax.set_title('PDE Loss Over Refined Levels (recorded every 500 epochs)')
ax.set_xlabel('Epochs')
ax.set_ylabel('PDE Loss')
ax.set_xticks(level_end_indices_cumulative_pde)
ax.set_xticklabels([str((level+1)*epochs) for level in range(len(level_end_indices_pde))])
legend_element = Line2D([0], [0], marker='o', color='w', label='Level End',
                         markerfacecolor='red', markersize=10)
ax.legend(handles=[legend_element])
plt.savefig(f'results/flower/PDE_adapt_flower.pdf',format='pdf')
# plt.savefig(f'results/sunflower/pdeloss_levels.pdf',format='pdf')

'''l2'''
combined_l2 = list(itertools.chain(*total_l2_list))
level_end_indices = [len(sublist) for sublist in total_l2_list]
level_end_indices_cumulative = [sum(level_end_indices[:i+1]) - 1 for i in range(len(level_end_indices))]
fig, ax = plt.subplots()
ax.plot(combined_l2, label='L2 Error')

for i, index in enumerate(level_end_indices_cumulative):
    ax.plot(index, combined_l2[index], 'ro')
    ax.text(index, combined_l2[index], str(i+1), color='black', ha='center')

ax.set_title('Relative L2 Error Over Refined Levels (recorded every 500 epochs)')
ax.set_xlabel('Epochs')
ax.set_ylabel('L2 Error')
ax.set_yscale('log')

ax.set_xticks(level_end_indices_cumulative)
ax.set_xticklabels([str((level+1)*epochs) for level in range(len(level_end_indices))])
legend_element = Line2D([0], [0], marker='o', color='w', label='Level End',
                         markerfacecolor='red', markersize=10)
ax.legend(handles=[legend_element])
plt.savefig(f'results/flower/l2_adapt_flower.pdf',format='pdf')

# plt.savefig(f'results/sunflower/l2_error_levels.pdf',format='pdf')


'''2d'''
'''straight change BOX!!'''
# bp = 0.25
# bm = 0.5
# pde = straight_interface(bp, bm)
# save_path_straight = 'Save_results/straight/'

'''Circle'''
# beta_list = [1, 1000, 100000]
# bm = 1000 #beta1
# bp = 1 #beta2
#
# r0 = 0.5
# save_path_circle = f'Save_results/circle/bp={bp}, bm={bm}/'
# pde = circle_interface(bp, bm, r0)
# ResNet_omega1, ResNet_omega2, err_l2 = train(200000,
#                                          1000, 700,
#                                          1024, 64, 128,
#                                          10, 3, 2)


'''sunflower'''
# bp = 10
# bm = 1
# omega = 20
# save_path_sunflower = 'Save_results/sunflower/'
# pde = sunflower_interface()

# ResNet_omega1, ResNet_omega2, err_l2 = train(50000,
#                                          3000, 500,
#                                          1024, 256, 128,
#                                          10, 3, 2)


# sorted_residuals_list = [res_r_list[i] for i in xr_sorted_indices]
# print('xr_top_10_points', xr_top_10_points)
# for i in range(num_top10_xr):
#     print(f"residual: {sorted_residuals_list[i]}, point: {sorted_points[i]}")