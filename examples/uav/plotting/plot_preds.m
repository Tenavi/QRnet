clear
close all
data = load('../results/test_pred.mat');

k0 = [find(data.t == 0), length(data.t)+1];
T = 5;%max(data.t) / 2;

fig1 = figure(1);
fig1.Position(3:4) = [1000, 500];

for N=1:data.n_trajectories
    t = data.t(k0(N):k0(N+1)-1);
    U = data.U(:,k0(N):k0(N+1)-1);
    U_pred = data.U_pred(:,k0(N):k0(N+1)-1);

    U_err = split_controls(U_pred - U);
    U_pred = split_controls(U_pred);
    U = split_controls(U);

    u_sel = {'throttle','aileron','elevator','rudder'};
    labels = {'\delta_t','\delta_a','\delta_e','\delta_r'};
    units = {'','[deg]','[deg]','[deg]'};
    for i=1:4
        u = u_sel{i};

        subplot(4,3,3*i-2);
        hold on
        box on

        if N==1 && i==1
            title('\textbf{Predictions}', 'FontSize',16,'interpreter','latex');
        end

        ax = gca;
        ax.FontSize = 12;
        ax.XLim = [0,T];

        plot(t, U_pred.(u))

        ylabel(['$\hat', labels{i}, '$ ', units{i}],...
                'FontSize',16,'interpreter','latex')

        subplot(4,3,3*i-1);
        hold on
        box on

        if N==1 && i==1
            title('\textbf{Data}', 'FontSize',16,'interpreter','latex');
        end

        ax = gca;
        ax.FontSize = 12;
        ax.XLim = [0,T];

        plot(t, U.(u))

        ylabel(['$', labels{i}, '$ ', units{i}],...
                'FontSize',16,'interpreter','latex')

        subplot(4,3,3*i);
        hold on
        box on

        if N==1 && i==1
            title('\textbf{Error}', 'FontSize',16,'interpreter','latex');
        end

        ax = gca;
        ax.FontSize = 12;
        ax.XLim = [0,T];

        plot(t, U_err.(u))

        ylabel(['$\hat', labels{i}, '-', labels{i}, '$ ', units{i}],...
                'FontSize',16,'interpreter','latex')
    end
end
