clear
close all

linewidth = [1.25,2];

W = 0;
architecture = 'LQR';
U_NN = nan;
U_opt = nan;

load('sim_data.mat')
params = load('../params.mat');

U_lb = min(params.U_lb);
U_ub = max(params.U_ub);
w_max = 5;

plot_bvp = 1;
plot_lqr = 1;
plot_nn = 1;

NN_name = architecture;

if isnan(U_opt)
    plot_bvp = 0;
end
if isnan(U_NN)
    plot_nn = 0;
end

if plot_bvp
    X_opt(5:7,:) = rad2deg(X_opt(5:7,:));
end
if plot_nn
    X_NN(5:7,:) = rad2deg(X_NN(5:7,:));
end
if plot_lqr
    X_LQR(5:7,:) = rad2deg(X_LQR(5:7,:));
end

T = max(t);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if plot_nn
    fig1 = figure;
    fig1.Position(3:4) = [350, 300];

    for i=1:3
        subplot(3,1,i);
        hold on

        axis tight
        box on

        ax = gca;
        ax.FontSize = 12;
        ax.XLim = [0,T];

        if i==1
            ax.YLim = [-1,1];

            if plot_bvp
                plot(t, X_opt(1,:),'k-','linewidth', linewidth(1))
                plot(t, X_opt(2,:),'k--','linewidth', linewidth(1))
                plot(t, X_opt(3,:),'k:','linewidth', linewidth(2))
                plot(t, X_opt(4,:),'k-.','linewidth', linewidth(1))
            end

            plot(t, X_NN(1,:),'-','linewidth', linewidth(1))
            plot(t, X_NN(2,:),'--','linewidth', linewidth(1))
            plot(t, X_NN(3,:),':','linewidth', linewidth(2))
            plot(t, X_NN(4,:),'-.','linewidth', linewidth(1))

            title('\textbf{NN}','interpreter','latex','fontsize',16)
            ylabel('$\mathbf q$','FontSize',16,'interpreter','latex')
        elseif i==2
            ax.YLim = (w_max+5)*[-1,1];

            if plot_bvp
                plot(t, X_opt(5,:),'k-','linewidth', linewidth(1))
                plot(t, X_opt(6,:),'k--','linewidth', linewidth(1))
                plot(t, X_opt(7,:),'k:','linewidth', linewidth(2))
            end

            plot(t, X_NN(5,:),'-','linewidth', linewidth(1))
            plot(t, X_NN(6,:),'--','linewidth', linewidth(1))
            plot(t, X_NN(7,:),':','linewidth', linewidth(2))

            ylabel('{\boldmath $\omega$}','FontSize',16,'interpreter','latex')
        elseif i==3
            ax.YLim = [U_lb-.1,U_ub+.1];
            ax.YTick = [U_lb,0,U_ub];

            if plot_bvp
                plot(t, U_opt(1,:),'k-','linewidth', linewidth(1))
                plot(t, U_opt(2,:),'k--','linewidth', linewidth(1))
                plot(t, U_opt(3,:),'k:','linewidth', linewidth(2))
            end

            if any(any(W))
                stairs(t, U_NN(1,:),'-','linewidth', linewidth(1))
                stairs(t, U_NN(2,:),'--','linewidth', linewidth(1))
                stairs(t, U_NN(3,:),':','linewidth', linewidth(2))
            else
                plot(t, U_NN(1,:),'-','linewidth', linewidth(1))
                plot(t, U_NN(2,:),'--','linewidth', linewidth(1))
                plot(t, U_NN(3,:),':','linewidth', linewidth(2))
            end

            xlabel('$t$','FontSize',16,'interpreter','latex')
            ylabel('$\mathbf u$','FontSize',16,'interpreter','latex')
        end
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if plot_lqr

    fig2 = figure;
    fig2.Position(3:4) = [350, 300];

    for i=1:3
        subplot(3,1,i);
        hold on

        axis tight
        box on

        ax = gca;
        ax.FontSize = 12;

        if i==1
            ax.YLim = [-1,1];

            if plot_bvp
                plot(t, X_opt(1,:),'k-','linewidth', linewidth(1))
                plot(t, X_opt(2,:),'k--','linewidth', linewidth(1))
                plot(t, X_opt(3,:),'k:','linewidth', linewidth(2))
                plot(t, X_opt(4,:),'k-.','linewidth', linewidth(1))
            end

            plot(t, X_LQR(1,:),'-','linewidth', linewidth(1))
            plot(t, X_LQR(2,:),'--','linewidth', linewidth(1))
            plot(t, X_LQR(3,:),':','linewidth', linewidth(2))
            plot(t, X_LQR(4,:),'-.','linewidth', linewidth(1))

            title('\textbf{LQR}','interpreter','latex','fontsize',16)
            ylabel('$\mathbf q$','FontSize',16,'interpreter','latex')
        elseif i==2
            ax.YLim = (w_max+5)*[-1,1];

            if plot_bvp
                plot(t, X_opt(5,:),'k-','linewidth', linewidth(1))
                plot(t, X_opt(6,:),'k--','linewidth', linewidth(1))
                plot(t, X_opt(7,:),'k:','linewidth', linewidth(2))
            end

            plot(t, X_LQR(5,:),'-','linewidth', linewidth(1))
            plot(t, X_LQR(6,:),'--','linewidth', linewidth(1))
            plot(t, X_LQR(7,:),':','linewidth', linewidth(2))

            ylabel('{\boldmath $\omega$}','FontSize',16,'interpreter','latex')
        elseif i==3
            ax.YLim = [U_lb-.1,U_ub+.1];
            ax.YTick = [U_lb,0,U_ub];

            if plot_bvp
                plot(t, U_opt(1,:),'k-','linewidth', linewidth(1))
                plot(t, U_opt(2,:),'k--','linewidth', linewidth(1))
                plot(t, U_opt(3,:),'k:','linewidth', linewidth(2))
            end

            if any(any(W))
                stairs(t, U_LQR(1,:),'-','linewidth', linewidth(1))
                stairs(t, U_LQR(2,:),'--','linewidth', linewidth(1))
                stairs(t, U_LQR(3,:),':','linewidth', linewidth(2))
            else
                plot(t, U_LQR(1,:),'-','linewidth', linewidth(1))
                plot(t, U_LQR(2,:),'--','linewidth', linewidth(1))
                plot(t, U_LQR(3,:),':','linewidth', linewidth(2))
            end

            xlabel('$t$','FontSize',16,'interpreter','latex')
            ylabel('$\mathbf u$','FontSize',16,'interpreter','latex')
        end
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if plot_bvp

    fig3 = figure;
    fig3.Position(3:4) = [350, 300];

    if plot_nn
        err_X_NN = X_NN-X_opt;
        err_U_NN = U_NN-U_opt;
    end

    if plot_lqr
        err_X_LQR = X_LQR-X_opt;
        err_U_LQR = U_LQR-U_opt;
    end

    for i=1:3
        subplot(3,1,i);
        hold on

        axis tight
        box on

        ax = gca;
        ax.FontSize = 12;
        ax.YScale = 'log';

        if i==1
            title('errors')

            if plot_lqr
                plot(t, vecnorm(err_X_LQR(1:4,:)),...
                    'linewidth', linewidth(1),'DisplayName', 'LQR')
            end
            if plot_nn
                plot(t, vecnorm(err_X_NN(1:4,:)),...
                    '--', 'linewidth', linewidth(1),'DisplayName', NN_name)
            end

            lgd = legend();
            lgd.FontSize = 14;
            lgd.Location = 'east';

            ylabel('$\Vert \mathbf q - \mathbf q^* \Vert$','FontSize',16,'interpreter','latex')
        elseif i==2
            if plot_lqr
                plot(t, vecnorm(err_X_LQR(4:7,:)), 'linewidth', linewidth(1))
            end
            if plot_nn
                plot(t, vecnorm(err_X_NN(4:7,:)), '--', 'linewidth', linewidth(1))
            end

            ylabel('{\boldmath $\Vert \omega - \omega^* \Vert$}','FontSize',16,'interpreter','latex')
        elseif i==3
            if plot_lqr
                plot(t, vecnorm(err_U_LQR), 'linewidth', linewidth(1))
            end
            if plot_nn
                plot(t, vecnorm(err_U_NN), '--', 'linewidth', linewidth(1))
            end

            xlabel('$t$','FontSize',16,'interpreter','latex')
            ylabel('$\Vert \mathbf u - \mathbf u^* \Vert$','FontSize',16,'interpreter','latex')
        end
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig4 = figure;
fig4.Position(3:4) = [350, 300];

for i=1:2
    subplot(2,1,i);
    hold on

    axis tight
    box on

    ax = gca;
    ax.FontSize = 12;

    if i==1
        if plot_bvp
            plot(t, dVdX_opt(1,:),'k-','linewidth', linewidth(1))
            plot(t, dVdX_opt(2,:),'k--','linewidth', linewidth(1))
            plot(t, dVdX_opt(3,:),'k:','linewidth', linewidth(2))
            plot(t, dVdX_opt(4,:),'k-.','linewidth', linewidth(1))
        end

        if plot_nn
            plot(t, dVdX_NN(1,:),'r-','linewidth', linewidth(1))
            plot(t, dVdX_NN(2,:),'r--','linewidth', linewidth(1))
            plot(t, dVdX_NN(3,:),'r:','linewidth', linewidth(2))
            plot(t, dVdX_NN(4,:),'r-.','linewidth', linewidth(1))
        end

        if plot_lqr
            plot(t, dVdX_LQR(1,:),'b-','linewidth', linewidth(1))
            plot(t, dVdX_LQR(2,:),'b--','linewidth', linewidth(1))
            plot(t, dVdX_LQR(3,:),'b:','linewidth', linewidth(2))
            plot(t, dVdX_LQR(4,:),'b-.','linewidth', linewidth(1))
        end

        title('\textbf{Costates}','FontSize',16,'interpreter','latex')
        ylabel('{\boldmath $\lambda_q$}','FontSize',16,'interpreter','latex')
    elseif i==2
        if plot_bvp
            p1 = plot(t, dVdX_opt(5,:),'k-','linewidth', linewidth(1));
            plot(t, dVdX_opt(6,:),'k--','linewidth', linewidth(1))
            plot(t, dVdX_opt(7,:),'k:','linewidth', linewidth(2))
        end

        if plot_nn
            p2 = plot(t, dVdX_NN(5,:),'r-','linewidth', linewidth(1));
            plot(t, dVdX_NN(6,:),'r--','linewidth', linewidth(1))
            plot(t, dVdX_NN(7,:),'r:','linewidth', linewidth(2))
        end

        if plot_lqr
            p3 = plot(t, dVdX_LQR(5,:),'b-','linewidth', linewidth(1));
            plot(t, dVdX_LQR(6,:),'b--','linewidth', linewidth(1))
            plot(t, dVdX_LQR(7,:),'b:','linewidth', linewidth(2))
        end

        ylabel('{\boldmath $\lambda_\omega$}','FontSize',16,'interpreter','latex')
        xlabel('$t$','FontSize',16,'interpreter','latex')

        if plot_bvp && plot_nn && plot_lqr
            lgd = legend([p1,p2,p3],'optimal',NN_name,'LQR');
        elseif plot_bvp && plot_nn
            lgd = legend([p1,p2],'optimal',NN_name);
        elseif plot_bvp && plot_lqr
            lgd = legend([p1,p3],'optimal','LQR');
        elseif plot_nn && plot_lqr
            lgd = legend([p2,p3],NN_name,'LQR');
        elseif plot_bvp
            lgd = legend(p1,'optimal');
        elseif plot_nn
            lgd = legend(p2,'nn');
        elseif plot_lqr
            lgd = legend(p3,'LQR');
        end
        lgd.Interpreter = 'latex';
        lgd.FontSize = 14;
    end
end
