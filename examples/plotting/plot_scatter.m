clear
close all

plot_settings

% Limit optimality plots only to stable NNs
stable_only = 0;

msize = 8;

xlim = nan;

%x_name = 'U_ML2_test'; xlim=[1e-03,1e-00];
x_name = 'U_RML2_test'; xlim=[1e-03,3e-01];
%xlim = [8e-02,5e-01];
%x_name = 'U_ML2_test'; xlim = [0.1,0.5];
%x_name = 'U_maxL2_test'; %xlim = [0,2];

y_names = [
    "max_final_dist";
    %"median_final_dist";
    "median_subopt";
    %"max_final_time";
    %"median_final_time";
    "max_eig_real";
    %"frac_stable";
    %"U_maxL2_test"
];

for y_name = y_names'
    model_info_ = model_info(~isnan(model_info.(y_name)), :);
    if stable_only && ~any([...
            contains(y_name, "median"),...
            contains(y_name, "dist"),...
            strcmp(y_name, "max_eig_real"),...
            contains(y_name, "test"),...
            contains(y_name, "time")...
        ])
        model_info_ = model_info_(model_info_.max_final_dist < fp_tol, :);
    end

    fig = figure;
    fig.Position(3:4) = [600,250];

    ax = gca;
    ax.FontSize = 14;
    ax.TickLabelInterpreter = 'latex';
    if not(isnan(xlim))
        ax.XLim = xlim;
    end
    ax.XScale = 'log';
    ax.ColorOrder = ColorOrder;

    hold on
    box on
    grid on
   
    if isfield(y_settings.(y_name), "yline")
        yline(y_settings.(y_name).yline, 'k', 'linewidth', 1.5, 'Displayname', '');
    end
    if isfield(y_settings.(y_name), "yline_LQR") && y_settings.(y_name).yline_LQR
        yline(LQR_info.(y_name), '--k', 'linewidth', 1.5, 'Displayname', '');
    end

    plots = [];

    plots(1) = plot(LQR_info.(x_name), LQR_info.(y_name),...
        'ko', 'markerfacecolor', 'k', 'markersize', msize,...
        'linewidth', 1.5, 'DisplayName', 'LQR'...
    );
    for j = 1:length(NN_names)
        x = table2array(model_info_(strcmp(model_info_.architecture, NN_names(j)), x_name));
        y = table2array(model_info_(strcmp(model_info_.architecture, NN_names(j)), y_name));
        if size(x,1) == 0
            plots(end+1) = plot(...
                nan, nan, markers(j), 'markersize', msize, 'linewidth', 1.5,...
                'DisplayName', legend_names(j), 'color', ax.ColorOrder(j,:)...
            );
        else
            plots(end+1) = plot(...
                x, y, markers(j), 'markersize', msize, 'linewidth', 1.5,...
                'DisplayName', legend_names(j), 'color', ax.ColorOrder(j,:)...
            );
        end
    end

    xlabel(y_settings.(x_name).label, 'interpreter','latex','fontsize',16)
    
    if isfield(y_settings.(y_name), "label")
        ylabel(y_settings.(y_name).label,'interpreter','latex','fontsize',16)
    end

    if isfield(y_settings.(y_name), "ylim")
        for i=1:2
            if ~isnan(y_settings.(y_name).ylim(i))
                ax.YLim(i) = y_settings.(y_name).ylim(i);
            end
        end
    end

    title(join(['\textbf{', problem_name, ': ', y_settings.(y_name).title, '}'], ''),...
        'interpreter','latex','fontsize',16) 

    if isfield(y_settings.(y_name), "scale")
        ax.YScale = y_settings.(y_name).scale;
    end
    
    if strcmp(y_name,'train_time') && any(mod(ax.YTick, 1) ~= 0)
        ax.YAxis.TickLabelFormat = '%.1f';
    end

    lgd = legend(plots);
    lgd.Interpreter = 'latex';
    lgd.Location = 'northwest';
    %lgd.Location = 'eastoutside';
    lgd.FontSize = 14;
end