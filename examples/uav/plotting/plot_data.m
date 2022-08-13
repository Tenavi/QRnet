clear
close all
data = load('../data/train_refined.mat');

k0 = [find(data.t == 0), length(data.t)+1];
T = 10.;%max(data.t) / 2;

fig1 = figure(1);
fig1.Position(3:4) = [1000, 500];

for N=1:data.n_trajectories
    t = data.t(k0(N):k0(N+1)-1);
    X = data.X(:,k0(N):k0(N+1)-1);
    U = data.U(:,k0(N):k0(N+1)-1);

    X = split_states(t, X);
    U = split_controls(U); 

    for j=0:3
        if j==0
            x_sel = {'pn','pe','h'};
            labels = {'p_n','p_e','h - h_f'};
            units = {'[m]','[m]','[m]'};
        elseif j==1
            x_sel = {'u','v','w'};
            labels = x_sel;
            units = {'[m/s]','[m/s]','[m/s]'};
        elseif j==2
            x_sel = {'roll','pitch','yaw'};
            labels = {'\phi','\theta','\psi - \psi_f'};
            units = {'[deg]','[deg]','[deg]'};
        elseif j==3
            x_sel = {'p','q','r'};
            labels = x_sel;
            units = {'[deg/s]','[deg/s]','[deg/s]'};
        end

        for i=1:3
            x = x_sel{i};
            subplot(4,5,i+5*j);
            hold on
            box on

            ax = gca;
            ax.FontSize = 12;
            ax.XLim = [0,T];

            plot(t, X.(x))

            ylabel(['$', labels{i}, '$ ', units{i}],...
                'FontSize',16,'interpreter','latex')
        end
    end

    x_sel = {'Va','alpha','beta','course'};
    labels = {'V_a','\alpha','\beta','\chi - \chi_f'};
    units = {'[m/s]','[deg]','[deg]','[deg]'};

    for i=1:4
        x = x_sel{i};
        subplot(4,5,4 + 5*(i-1));
        hold on
        box on

        ax = gca;
        ax.FontSize = 12;
        ax.XLim = [0,T];

        plot(t, X.(x))

        ylabel(['$', labels{i}, '$ ', units{i}],...
            'FontSize',16,'interpreter','latex')
    end

    u_sel = {'throttle','aileron','elevator','rudder'};
    labels = {'\delta_t','\delta_a','\delta_e','\delta_r'};
    units = {'','[deg]','[deg]','[deg]'};
    for i=1:4
        u = u_sel{i};
        subplot(4,5,5*i);
        hold on
        box on

        ax = gca;
        ax.FontSize = 12;
        ax.XLim = [0,T];
        %ax.YLim = [U_lb.(u),U_ub.(u)];

        plot(t, U.(u))

        ylabel(['$', labels{i}, '$ ', units{i}],...
                'FontSize',16,'interpreter','latex')
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig2 = figure(2);
hold on
grid on
box on
view(-30,30)

for N=1:data.n_trajectories
    t = data.t(k0(N):k0(N+1)-1);
    X = data.X(:,k0(N):k0(N+1)-1);
    U = data.U(:,k0(N):k0(N+1)-1);

    X = split_states(t, X);
    U = split_controls(U);

    plot3(X.pe, X.pn, X.h)
end

xlabel('crossrange [m]','FontSize',16,'interpreter','latex')
ylabel('downrange [m]','FontSize',16,'interpreter','latex')
zlabel('altitude $(h - h_f)$ [m]','FontSize',16,'interpreter','latex')

ax = gca;
x_size = ax.XLim(2) - ax.XLim(1);
y_size = ax.YLim(2) - ax.YLim(1);
z_size = ax.ZLim(2) - ax.ZLim(1);
max_size = max([x_size,y_size,z_size]);
if x_size < max_size
    size_dif = (max_size - x_size)/2;
    ax.XLim = ax.XLim + size_dif/2*[-1,1];
end
if y_size < max_size
    size_dif = (max_size - y_size)/2;
    ax.YLim = ax.YLim + size_dif/2*[-1,1];
end
if z_size < max_size
    ax.ZLim = ax.ZLim + [-10,10];
end