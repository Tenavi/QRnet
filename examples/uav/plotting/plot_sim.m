clear
close all

linewidth = [1.25,2];

architecture = 'LQR';
t_NN = nan;
t_opt = nan;
t_LQR = nan;

if isfile('../results/old_sim_data.mat')
    old_sim_data = load('../results/old_sim_data.mat');
end

load('../results/sim_data.mat')
params = load('../params.mat');

T = 30;%min([max(t_NN), max(t_LQR)]);

U_lb = split_controls(params.U_lb);
U_ub = split_controls(params.U_ub);

plot_opt = 1;
plot_lqr = 1;
plot_nn = 1;
plot_old = 1;

NN_name = architecture;
load('../../plotting/NN_names.mat')
NN_name_idx = strcmp(NN_name, NN_names);
NN_legend_name = legend_names(NN_name_idx);

if isnan(t_opt)
    plot_opt = 0;
end
if isnan(t_NN)
    plot_nn = 0;
end
if isnan(t_LQR)
    plot_lqr = 0;
end
if ~exist('old_sim_data')
    plot_old = 0;
end

X_trim = split_states(t_opt, params.X_bar);
U_trim = split_controls(params.U_bar);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if plot_opt
    X_opt = split_states(t_opt, X_opt);
    U_opt = split_controls(U_opt);
end

if plot_lqr
    X_LQR = split_states(t_LQR, X_LQR);
    U_LQR = split_controls(U_LQR);
end

if plot_nn
    X_NN = split_states(t_NN, X_NN);
    U_NN = split_controls(U_NN);
end

if plot_old
    t_old = old_sim_data.t_NN;
    X_old = split_states(t_old, old_sim_data.X_NN);
    U_old = split_controls(old_sim_data.U_NN);
    old_NN_name_idx = strcmp(old_sim_data.architecture, NN_names);
    old_NN_legend_name = legend_names(old_NN_name_idx);
end

fig1 = figure;
fig1.Position(3:4) = [1200, 500];
fig1_plots = tiledlayout(4,5,'TileSpacing','tight','Padding','compact');

for j=0:3
    if j==0
        x_sel = {'pn','pe','h'};
        labels = {'p_n','p_e','h - h_f'};
        units = {'[m]','[m]','[m]'};
        column_title = 'position';
    elseif j==1
        x_sel = {'u','v','w'};
        labels = x_sel;
        units = {'[m/s]','[m/s]','[m/s]'};
        column_title = 'velocity';
    elseif j==2
        x_sel = {'roll','pitch','yaw'};
        labels = {'\phi','\theta','\psi - \psi_f'};
        units = {'[deg]','[deg]','[deg]'};
        column_title = 'attitude';
    elseif j==3
        x_sel = {'p','q','r'};
        labels = x_sel;
        units = {'[deg/s]','[deg/s]','[deg/s]'};
        column_title = 'angular velocity';
    end

    for i=1:3
        x = x_sel{i};
        nexttile(5*i-(3-j))
        hold on
        box on

        ax = gca;
        ax.FontSize = 12;
        ax.TickLabelInterpreter = 'latex';
        ax.XLim = [0,T];

        if plot_opt
            plot(t_opt, X_opt.(x),'k','linewidth', 1.5)
        end
        if plot_lqr
            plot(t_LQR, X_LQR.(x), ':', 'color', ax.ColorOrder(1,:), 'linewidth', 2)
        end
        if plot_old
            plot(t_old, X_old.(x), '-.', 'color', [.5,.5,.5], 'linewidth', 1.5)
        end
        if plot_nn
            plot(t_NN, X_NN.(x), '--', 'color', ax.ColorOrder(2,:), 'linewidth', 1.5)
        end

        ylabel(['$', labels{i}, '$ ', units{i}],...
            'FontSize',16,'interpreter','latex')
        if i==1
            title(['\textbf{' column_title, '}'],...
                'FontSize',16,'interpreter','latex')
        end
    end
end

x_sel = {'course','alpha','beta'};
labels = {'\chi - \chi_f','\alpha','\beta'};
units = {'[deg]','[deg]','[deg]'};
column_titles = {'course','angle of attack','sideslip'};

for i=1:3
    x = x_sel{i};
    nexttile(16+i)
    hold on
    box on

    ax = gca;
    ax.FontSize = 12;
    ax.TickLabelInterpreter = 'latex';
    ax.XLim = [0,T];

    if plot_opt
        plot(t_opt, X_opt.(x),'k','linewidth', 1.5)
    end
    if plot_lqr
        plot(t_LQR, X_LQR.(x), ':', 'color', ax.ColorOrder(1,:), 'linewidth', 2)
    end
    if plot_old
        plot(t_old, X_old.(x), '-.', 'color', [.5,.5,.5], 'linewidth', 1.5)
    end
    if plot_nn
        plot(t_NN, X_NN.(x), '--', 'color', ax.ColorOrder(2,:), 'linewidth', 1.5)
    end

    xlabel('$t$ [s]', 'FontSize',16,'interpreter','latex')
    ylabel(['$', labels{i}, '$ ', units{i}],...
        'FontSize',16,'interpreter','latex')
    title(['\textbf{', column_titles{i}, '}'],...
        'FontSize',16,'interpreter','latex')
end

u_sel = {'throttle','aileron','elevator','rudder'};
labels = {'\delta_t','\delta_a','\delta_e','\delta_r'};
units = {'','[deg]','[deg]','[deg]'};
for i=1:4
    u = u_sel{i};
    nexttile(5*(i-1)+1);
    hold on
    box on

    ax = gca;
    ax.FontSize = 12;
    ax.TickLabelInterpreter = 'latex';
    ax.XLim = [0,T];
    ax.YLim = [U_lb.(u),U_ub.(u)];

    if plot_opt
        plot(t_opt, U_opt.(u),'k','linewidth', 1.5, 'DisplayName', 'optimal')
    end
    if plot_lqr
        plot(t_LQR, U_LQR.(u), ':', 'color', ax.ColorOrder(1,:), 'linewidth', 2 ,'DisplayName', 'LQR')
    end
    if plot_old
        plot(t_old, U_old.(u), '-.', 'color', [.5,.5,.5], 'linewidth', 1.5, 'DisplayName', old_NN_legend_name)
    end
    if plot_nn
        plot(t_NN, U_NN.(u), '--', 'color', ax.ColorOrder(2,:), 'linewidth', 1.5, 'DisplayName', NN_legend_name)
    end

    ylabel(['$', labels{i}, '$ ', units{i}],...
            'FontSize',16,'interpreter','latex')
    if i==1
        title('\textbf{controls}','FontSize',16,'interpreter','latex')
    elseif i==4
        xlabel('$t$ [s]', 'FontSize',16,'interpreter','latex')
    end
end

lgd = legend;
lgd.FontSize = 16;
lgd.Interpreter = 'latex';
lgd.Layout.Tile = 20;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig2 = figure;
fig2.Position(3:4) = [600, 300];
hold on
grid on
box on

T = max([max(t_opt), max(t_LQR), max(t_NN)]);

if plot_opt
    t_idx = t_opt <= T;
    plot3(X_opt.pe(t_idx), X_opt.pn(t_idx), X_opt.h(t_idx),...
        'k','linewidth', 1.5,'displayname','optimal')
end
if plot_lqr
    t_idx = t_LQR <= T;
    plot3(X_LQR.pe(t_idx), X_LQR.pn(t_idx), X_LQR.h(t_idx),...
        ':', 'color', ax.ColorOrder(1,:), 'linewidth', 2.5, 'displayname', 'LQR')
end
if plot_old
    t_idx = t_old <= T;
    plot3(X_old.pe(t_idx), X_old.pn(t_idx), X_old.h(t_idx),...
        '--', 'color', [.5,.5,.5], 'linewidth', 2, 'displayname', old_NN_legend_name)
end
if plot_nn
    t_idx = t_NN <= T;
    plot3(X_NN.pe(t_idx), X_NN.pn(t_idx), X_NN.h(t_idx),...
        '--', 'color', ax.ColorOrder(2,:), 'linewidth', 2, 'displayname', NN_legend_name)
end

ax = gca;
ax.TickLabelInterpreter = 'latex';
ax.FontSize = 14;

x_size = ax.XLim(2) - ax.XLim(1);
y_size = ax.YLim(2) - ax.YLim(1);
z_size = ax.ZLim(2) - ax.ZLim(1);
max_size = max([x_size,y_size,z_size]);
if x_size < max_size
    size_dif = (max_size - x_size)/2;
    ax.XLim = ax.XLim + size_dif/20*[-1,1];
end
if y_size < max_size
    size_dif = (max_size - y_size)/2;
    ax.YLim = ax.YLim + size_dif/2*[-1,1];
end
if z_size < max_size
    ax.ZLim = ax.ZLim + [-10,10];
end

view(45,45)

xlabel('crossrange $p_e$ [m]','FontSize',18,'interpreter','latex')
ylabel('downrange $p_n$ [m]','FontSize',18,'interpreter','latex')
zlabel('altitude $(h - h_f)$ [m]','FontSize',18,'interpreter','latex')

lgd = legend();
lgd.FontSize = 18;
lgd.Interpreter = 'latex';
lgd.Location = 'northeast';
