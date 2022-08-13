clear
%close all

linewidth = [1.25,2];

W = 0;
architecture = 'LQR';
U_NN = nan;
U_opt = nan;

load('sim_data.mat')
params = load('../params.mat');

U_lim = [params.U_lb, params.U_ub];

plot_bvp = 1;
plot_lqr = 1;
plot_nn = 1;

plot_dvdx = 1;

NN_name = architecture;
NN_name_info = load('../../plotting/NN_names.mat');
NN_name_idx = strcmp(NN_name, NN_name_info.NN_names);
NN_legend_name = NN_name_info.legend_names(NN_name_idx);

if isnan(U_opt)
    plot_bvp = 0;
end
if isnan(U_NN)
    plot_nn = 0;
end

T = min([max(t_NN), max(t_LQR), max(t_opt)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vdp = @(t,x) [x(2); params.mu*(1-x(1)^2)*x(2)-x(1)];

X0 = X_LQR(:,1);
[t_unctrl, X_unctrl] = ode45(vdp, [0,10*T], X0);
X_unctrl = X_unctrl';

fig = figure;
fig.Position(3:4) = [500, 300];

hold on
box on

ax = gca;
ax.FontSize = 12;
ax.TickLabelInterpreter = 'latex';

plot3(X_unctrl(1,:),X_unctrl(2,:),0*t_unctrl,...
    'k','DisplayName', 'uncontrolled', 'color', [0.5,0.5,0.5]);
if plot_bvp
    plot3(X_opt(1,:),X_opt(2,:),U_opt,...
        'DisplayName', 'optimal', 'linewidth', 1.5, 'color', [0,0,0])
end
if plot_lqr
    plot3(X_LQR(1,:),X_LQR(2,:),U_LQR,...
        ':','DisplayName', 'LQR','linewidth', 2, 'color', ax.ColorOrder(1,:))
end
if plot_nn
    plot3(X_NN(1,:),X_NN(2,:),U_NN,...
        '--','DisplayName', NN_legend_name,'linewidth', 2, 'color', ax.ColorOrder(2,:))
end

lgd = legend;
lgd.FontSize = 14;
lgd.Interpreter = 'latex';

title(ax, '\textbf{Closed-loop dynamics}','FontSize',16,'interpreter','latex')
xlabel('$x$', 'FontSize',16,'interpreter','latex')
ylabel('$\dot x$', 'FontSize',16,'interpreter','latex')
zlabel('$u$', 'FontSize', 16, 'Interpreter','latex')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig = figure;
fig.Position(3:4) = [500, 300];

for i = 1:2
    subplot(2,1,i);

    hold on
    box on

    ax = gca;
    ax.FontSize = 12;
    ax.TickLabelInterpreter = 'latex';
    ax.XLim = [0, T];
    
    if plot_bvp
        plot(t_opt,X_opt(i,:),'k','DisplayName', 'optimal', 'linewidth', 1.5)
        %plot(t_nodes, X_nodes(i,:), 'ko', 'Displayname', 'nodes')
    end
    if plot_lqr
        plot(t_LQR,X_LQR(i,:),':','DisplayName', 'LQR','linewidth', 2, 'color', ax.ColorOrder(1,:))
    end
    if plot_nn
        plot(t_NN,X_NN(i,:),'--','DisplayName', NN_legend_name,'linewidth', 2, 'color', ax.ColorOrder(2,:))
    end

    if i == 1
        ylabel('$x$','FontSize',16,'interpreter','latex');
        title('\textbf{Closed-loop dynamics}','FontSize',16,'interpreter','latex')

        lgd = legend;
        lgd.FontSize = 14;
        lgd.Interpreter = 'latex';
    elseif i == 2
        ylabel('$\dot x$','FontSize',16,'interpreter','latex');
    end

    xlabel('$t$', 'FontSize',16,'interpreter','latex')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if plot_dvdx
    fig = figure;
    fig.Position(3:4) = [500, 300];

    for i = 1:2
        subplot(2,1,i);

        hold on
        box on

        ax = gca;
        ax.TickLabelInterpreter = 'latex';
        ax.FontSize = 12;
        ax.XLim = [0, T];

        if plot_bvp
            plot(t_opt,dVdX_opt(i,:),'k','DisplayName', 'optimal', 'linewidth', 1.5)
        end
        if plot_lqr
            plot(t_LQR,dVdX_LQR(i,:),':','DisplayName', 'LQR','linewidth', 2, 'color', ax.ColorOrder(1,:))
        end
        if plot_nn
            plot(t_NN,dVdX_NN(i,:),'--','DisplayName', NN_legend_name,'linewidth', 2, 'color', ax.ColorOrder(2,:))
        end

        if i == 1
            title('\textbf{Costate dynamics}','FontSize',16,'interpreter','latex')

            lgd = legend;
            lgd.FontSize = 14;
            lgd.Interpreter = 'latex';
        end

        xlabel('$t$', 'FontSize',16,'interpreter','latex')
        ylabel(['$\lambda_', num2str(i), '$'],'FontSize',16,'interpreter','latex');
    end
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig = figure;
fig.Position(3:4) = [500, 150];

hold on
box on

ax = gca;
ax.FontSize = 12;
ax.TickLabelInterpreter = 'latex';
ax.XLim = [0, T];
ax.YLim = U_lim;
ax.YTick = [params.U_lb,0,params.U_ub];

if plot_bvp
    plot(t_opt,U_opt,'k','DisplayName', 'optimal', 'linewidth', 1.5)
    %plot(t_nodes, U_nodes, 'ko', 'Displayname', 'nodes')
end
if plot_lqr
    plot(t_LQR,U_LQR,':','DisplayName', 'LQR','linewidth', 2, 'color', ax.ColorOrder(1,:))
end
if plot_nn
    plot(t_NN,U_NN,'--','DisplayName', NN_legend_name,'linewidth', 2, 'color', ax.ColorOrder(2,:))
end

xlabel('$t$','FontSize',16,'interpreter','latex')
ylabel('$u$','FontSize',16,'interpreter','latex');
title('\textbf{Control}','FontSize',16,'interpreter','latex')

lgd = legend;
lgd.FontSize = 14;
lgd.Interpreter = 'latex';
