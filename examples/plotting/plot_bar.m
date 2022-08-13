clear
close all

plot_settings

y_names = [
    "train_time";
    %"U_ML2_test";
    "U_RML2_test";
    %"U_maxL2_test";
    %"max_eig_real"
    %"frac_stable"
];

model_info = model_info(~strcmp(model_info.architecture, 'LQR'), :);

Nd = unique(model_info.n_trajectories_train);

for y_name = y_names'

    avgs = nan*ones(length(Nd), length(NN_names));
    p25s = avgs;
    p75s = avgs;
    mins = avgs;
    maxs = avgs;
    
    for j = 1:length(NN_names)
        NN_idx = strcmp(model_info.architecture, NN_names(j));

        NN_data = model_info(NN_idx, {y_name{1}, 'n_trajectories_train'});

        for i=1:length(Nd)
            Nd_idx = NN_data.n_trajectories_train == Nd(i);
            Nd_data = table2array(NN_data(Nd_idx, y_name));

            if height(Nd_data) > 0
                avgs(i,j) = median(Nd_data);
                p25s(i,j) = prctile(Nd_data,25);
                p75s(i,j) = prctile(Nd_data,75);
                mins(i,j) = min(Nd_data);
                maxs(i,j) = max(Nd_data);
            end
        end
    end

    fig = figure;
    fig.Position(3:4) = [600,300];
    
    ax = gca;
    ax.ColorOrder = ColorOrder;

    hold on
    box on
    grid on

    barwidth = 0.8;
    b = bar(avgs, 'grouped', 'barwidth', barwidth);

    ngroups = size(avgs, 1);
    nbars = size(avgs, 2);
    % Calculate the width for each bar group
    groupwidth = min(0.8, nbars/(nbars + 1.5));
    % Set the position of each error bar in the centre of the main bar
    % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
    x = zeros(nbars,ngroups);
    for i = 1:nbars
        % Calculate center of each bar
        x(i,:) = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
        errorbar(x(i,:), avgs(:,i), avgs(:,i) - p25s(:,i), p75s(:,i) - avgs(:,i),...
            'k', 'linestyle', 'none', 'linewidth', 1.5, 'capsize', 10);
        
        scatter(x(i,:), maxs(:,i), 'k^', 'filled')
        scatter(x(i,:), mins(:,i), 'kv', 'filled')
    end
        
    ax.XTick = mean(x,1);
    ax.XTickLabel = Nd;
    ax.FontSize = 14;
    ax.TickLabelInterpreter = 'latex';
    
    if isfield(y_settings.(y_name), "yline")
        yline(y_settings.(y_name).yline, 'k', 'linewidth', 2);
    end
    if isfield(y_settings.(y_name), "yline_LQR") && y_settings.(y_name).yline_LQR
        LQR_plot = yline(LQR_info.(y_name), '--k', 'linewidth', 2);
    end
    
    xlabel('number of trajectories','interpreter','latex','fontsize',16)
    
    if isfield(y_settings.(y_name), "label")
        ylabel(y_settings.(y_name).label,'interpreter','latex','fontsize',16)
    end

    title(join(['\textbf{', problem_name, ':', y_settings.(y_name).title, '}']),...
        'interpreter','latex','fontsize',16)
    
    if isfield(y_settings.(y_name), "scale")
        ax.YScale = y_settings.(y_name).scale;
    end
    
    if isfield(y_settings.(y_name), "ylim")
        for i=1:2
            if ~isnan(y_settings.(y_name).ylim(i))
                ax.YLim(i) = y_settings.(y_name).ylim(i);
            end
        end
    end

    if strcmp(y_name,'train_time') && any(mod(ax.YTick, 1) ~= 0)
        ax.YAxis.TickLabelFormat = '%.1f';
    end

    if 0%exist('LQR_plot', 'var')
       lgd = legend([b,LQR_plot], [legend_names;"LQR"]);
    else
       lgd = legend(b, legend_names);
    end
    %lgd.Location = 'eastoutside';
    %lgd.Location = 'northwest'; lgd.NumColumns = 2; lgd.Orientation = 'horizontal';
    %lgd.Location = 'northeast'; lgd.Orientation = 'horizontal';
    lgd.Location = 'northoutside'; lgd.NumColumns = 6;
    lgd.Interpreter = 'latex';
    lgd.FontSize = 14;
    
    clear LQR_plot
end
